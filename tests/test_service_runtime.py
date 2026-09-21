"""Exercise service startup policy without allocating models or using a GPU."""

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from qev.api import MediaBudget, SystemOneRequest, decode_question_options, decode_state
from qev.service_runtime import service_runtime

GIB = 1024 ** 3
MIB = 1024 ** 2


class MemoryRuntime:
    gpu = "gpu"

    def __init__(self, active=3 * GIB, recommended=12 * GIB, total=16 * GIB, old=0, device="gpu",
                 cache=3 * GIB, cache_limit=12 * GIB):
        self.active, self.recommended, self.total = active, recommended, total
        self.cache, self.cache_limit = cache, cache_limit
        self.limit, self.device = old, SimpleNamespace(type=device)
        self.metal = SimpleNamespace(is_available=lambda: True)
        self.calls = []

    def default_device(self):
        return self.device

    def get_active_memory(self):
        self.calls.append("active")
        return self.active

    def get_cache_memory(self):
        self.calls.append("cached")
        return self.cache

    def set_cache_limit(self, limit):
        self.calls.append(("cache", limit))
        previous, self.cache_limit = self.cache_limit, limit
        return previous

    def clear_cache(self):
        self.calls.append("clear_cache")
        self.cache = 0

    def device_info(self):
        return {"memory_size": self.total, "max_recommended_working_set_size": self.recommended}

    def set_wired_limit(self, limit):
        self.calls.append(("wire", limit))
        previous, self.limit = self.limit, limit
        return previous

    def synchronize(self):
        self.calls.append("synchronize")


def install(monkeypatch, runtime):
    monkeypatch.setattr("qev.service_runtime.import_module", lambda name: runtime)
    return SimpleNamespace(backend="mlx", model=SimpleNamespace(_device=runtime.device))


@pytest.mark.parametrize("active,recommended,total,expected", [
    (GIB, 12 * GIB, 16 * GIB, GIB + 512 * MIB),
    (3 * GIB, 12 * GIB, 16 * GIB, 3 * GIB + (3 * GIB) // 5),
    (7 * GIB, 12 * GIB, 16 * GIB, 8 * GIB),
    (7 * GIB, 6 * GIB, 16 * GIB, 6 * GIB),
])
def test_memory_budget_is_bounded_and_restored(monkeypatch, active, recommended, total, expected):
    runtime = MemoryRuntime(active, recommended, total, old=123)
    agent = install(monkeypatch, runtime)
    with service_runtime(agent, warmup=False, stream=io.StringIO()):
        assert runtime.limit == expected
    assert runtime.limit == 123
    assert runtime.calls[-3:] == ["synchronize", ("cache", 12 * GIB), ("wire", 123)]


@pytest.mark.parametrize("old_cache_limit", [0, 64 * MIB, 256 * MIB, 12 * GIB])
def test_idle_cache_is_bounded_cleared_before_wiring_and_original_limit_restored(monkeypatch, old_cache_limit):
    runtime = MemoryRuntime(cache_limit=old_cache_limit)
    output = io.StringIO()
    with service_runtime(install(monkeypatch, runtime), warmup=False, stream=output):
        assert runtime.cache_limit == min(old_cache_limit, 256 * MIB)
        assert runtime.cache == 0 and runtime.active == 3 * GIB
        wired_at = next(i for i, call in enumerate(runtime.calls) if isinstance(call, tuple) and call[0] == "wire")
        assert runtime.calls.index("clear_cache") < wired_at
        assert "allocator before: 3072 MiB active, 3072 MiB cached" in output.getvalue()
        assert "allocator after: 3072 MiB active, 0 MiB cached" in output.getvalue()
    assert runtime.cache_limit == old_cache_limit


@pytest.mark.parametrize("operation", ["get_cache_memory", "set_cache_limit", "clear_cache"])
def test_unsupported_cache_management_still_applies_and_restores_residency(monkeypatch, operation):
    runtime = MemoryRuntime(old=123)

    def unavailable(*args):
        raise AttributeError("cache API unavailable")

    setattr(runtime, operation, unavailable)
    output = io.StringIO()
    with service_runtime(install(monkeypatch, runtime), warmup=False, stream=output):
        assert runtime.limit > runtime.active
        assert "cache management unavailable" in output.getvalue()
    assert runtime.limit == 123 and runtime.cache_limit == 12 * GIB


def test_larger_existing_user_limit_is_preserved(monkeypatch):
    runtime = MemoryRuntime(old=9 * GIB)
    agent = install(monkeypatch, runtime)
    output = io.StringIO()
    with service_runtime(agent, warmup=False, stream=output):
        assert runtime.limit == 9 * GIB
        assert "preserved existing limit" in output.getvalue()
    assert runtime.limit == 9 * GIB


@pytest.mark.parametrize("backend,enabled", [("torch", True), ("mlx", False)])
def test_non_mlx_or_opt_out_never_imports_or_changes_mlx(monkeypatch, backend, enabled):
    def reject_import(name):
        raise AssertionError("MLX must not be touched")
    monkeypatch.setattr("qev.service_runtime.import_module", reject_import)
    with service_runtime(SimpleNamespace(backend=backend), wired_memory=enabled, warmup=False, stream=io.StringIO()):
        pass


def test_cpu_mlx_does_not_change_memory_settings(monkeypatch):
    runtime = MemoryRuntime(device="cpu")
    with service_runtime(install(monkeypatch, runtime), warmup=False, stream=io.StringIO()):
        assert not runtime.calls
    assert not runtime.calls


@pytest.mark.parametrize("failure", [AttributeError("old MLX API"), RuntimeError("wiring unsupported")])
def test_missing_or_unsupported_memory_api_does_not_block_service(monkeypatch, failure):
    runtime = MemoryRuntime()
    def unavailable(limit):
        raise failure
    runtime.set_wired_limit = unavailable
    output = io.StringIO()
    with service_runtime(install(monkeypatch, runtime), warmup=False, stream=output):
        assert "unavailable" in output.getvalue()
    assert runtime.limit == 0


@pytest.mark.parametrize("during_warmup", [False, True])
def test_startup_or_server_failure_restores_old_residency(monkeypatch, during_warmup):
    runtime = MemoryRuntime(old=GIB)
    agent = install(monkeypatch, runtime)
    def fail(**request):
        raise ValueError("injected warmup failure")
    agent.predict = fail
    output = io.StringIO()
    with pytest.raises(ValueError, match="injected"), service_runtime(agent, warmup=during_warmup, stream=output):
        raise ValueError("injected server failure")
    assert runtime.limit == GIB
    assert runtime.cache_limit == 12 * GIB
    assert runtime.calls[-3:] == ["synchronize", ("cache", 12 * GIB), ("wire", GIB)]


@pytest.mark.parametrize("failure", ["synchronize", "cache_restore"])
def test_shutdown_memory_failures_do_not_prevent_other_setting_restoration(monkeypatch, failure):
    runtime = MemoryRuntime(old=123)
    agent = install(monkeypatch, runtime)
    output = io.StringIO()

    def unavailable(*args):
        raise RuntimeError("injected shutdown failure")

    with service_runtime(agent, warmup=False, stream=output):
        if failure == "synchronize":
            runtime.synchronize = unavailable
        else:
            runtime.set_cache_limit = unavailable
    assert runtime.limit == 123
    assert "injected shutdown failure" in output.getvalue()
    if failure == "synchronize":
        assert runtime.cache_limit == 12 * GIB


def test_warmup_runs_real_predict_shapes_once_and_does_not_cache_user_requests():
    calls = []
    def predict(**request):
        calls.append(request)
        return {"usage": {"input_tokens": len(calls)}, "answers": {"unique": len(calls)}}
    agent = SimpleNamespace(backend="torch", predict=predict)
    with service_runtime(agent, stream=io.StringIO()):
        assert len(calls) == 3
        modalities = []
        for request in calls:
            validated = SystemOneRequest(**request)
            budget = MediaBudget()
            state = decode_state(validated.state, budget)
            options = [decode_question_options(question, budget) for question in validated.questions.values()]
            modalities.append((len(state.images), sum(len(option.images) for group in options for option in group)))
        assert modalities == [(0, 0), (1, 0), (0, 3)]
        # An identical later input still calls predict, with a fresh answer.
        repeated = agent.predict(**calls[0])
        assert repeated["answers"]["unique"] == 4
    assert len(calls) == 4


def test_cli_opt_out_flags_wrap_the_server_lifetime(monkeypatch):
    from qev.cli import main

    calls = []
    agent = SimpleNamespace(backend="torch")
    monkeypatch.setattr("qev.inference.Agent", lambda *args, **kwargs: agent)
    monkeypatch.setattr("qev.serve.create_app", lambda actual, request_log=None: calls.append(("app", actual, request_log)) or "app")
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append(("serve", app, kwargs)))
    main(["serve", "--model", "unused", "--no-warmup", "--no-wired-memory", "--no-request-log", "--port", "8010"])
    assert calls == [("app", agent, None), ("serve", "app", {"host": "127.0.0.1", "port": 8010})]


def test_mlx_warmup_decisions_alias_and_native_share_one_persistent_thread():
    calls, threads = [], []

    def predict(**request):
        calls.append("predict")
        threads.append(threading.current_thread())
        return {"usage": {"input_tokens": 7}, "answers": {"serial": len(calls)},
                "latency_ms": 0., "qev": {"timings_ms": {"queue": 0., "inference": 0., "total": 0.}}}

    def native(request):
        calls.append("native")
        threads.append(threading.current_thread())
        return {"choices": [], "qev": {"latency_ms": 0., "decision_adapter_enabled": False}}

    agent = SimpleNamespace(backend="mlx", config={"name": "test"}, predict=predict,
                            system_one=predict, chat_completions=native)
    with service_runtime(agent, wired_memory=False, stream=io.StringIO()) as serving:
        assert serving.backend == "mlx" and serving.config is agent.config
        assert len(calls) == 3
        first = serving.predict(state="same", questions={})
        repeated = serving.system_one(state="same", questions={})
        assert first["answers"] != repeated["answers"]
        result = serving.chat_completions({})
        assert result["qev"]["latency_ms"] >= 0
        assert "timings_ms" not in result["qev"]
        assert len(set(threads)) == 1 and threads[0] is not threading.current_thread()
    assert not threads[0].is_alive()
    with pytest.raises(RuntimeError, match="shutdown"):
        serving.predict(state="after shutdown", questions={})


def test_concurrent_requests_run_serially_and_report_executor_queue(monkeypatch):
    started, release, second_submitted = threading.Event(), threading.Event(), threading.Event()
    calls, thread_ids = [], []

    def predict(state, **kwargs):
        calls.append(state)
        thread_ids.append(threading.get_ident())
        if state == "first":
            started.set()
            assert release.wait(timeout=2)
        return {"latency_ms": 0., "qev": {"timings_ms": {"queue": 0., "inference": 0., "total": 0.}}}

    agent = SimpleNamespace(backend="mlx", predict=predict)
    with service_runtime(agent, wired_memory=False, warmup=False, stream=io.StringIO()) as serving:
        original_submit = serving._executor.submit

        def submit(*args, **kwargs):
            future = original_submit(*args, **kwargs)
            if started.is_set():
                second_submitted.set()
            return future

        monkeypatch.setattr(serving._executor, "submit", submit)
        with ThreadPoolExecutor(max_workers=2) as clients:
            first = clients.submit(serving.predict, state="first", questions={})
            assert started.wait(timeout=2)
            second_submitted.clear()
            second = clients.submit(serving.predict, state="second", questions={})
            assert second_submitted.wait(timeout=2)
            try:
                time.sleep(0.04)
                assert calls == ["first"]
            finally:
                release.set()
            first.result(timeout=2)
            response = second.result(timeout=2)
    assert calls == ["first", "second"] and len(set(thread_ids)) == 1
    timings = response["qev"]["timings_ms"]
    assert timings["queue"] >= 30
    assert response["latency_ms"] == timings["total"] >= timings["queue"]


def test_inference_failure_does_not_replace_or_break_the_worker():
    threads = []

    def predict(state, **kwargs):
        threads.append(threading.get_ident())
        if state == "fail":
            raise ValueError("invalid request")
        return {"answers": {"ok": True}}

    agent = SimpleNamespace(backend="mlx", predict=predict)
    with service_runtime(agent, wired_memory=False, warmup=False, stream=io.StringIO()) as serving:
        with pytest.raises(ValueError, match="invalid request"):
            serving.predict(state="fail")
        assert serving.predict(state="good")["answers"]["ok"]
    assert len(set(threads)) == 1


def test_shutdown_joins_active_worker_before_restoring_memory(monkeypatch):
    runtime = MemoryRuntime(old=GIB)
    agent = install(monkeypatch, runtime)
    started, release = threading.Event(), threading.Event()
    original_set = runtime.set_wired_limit

    def set_limit(limit):
        if limit == GIB:
            assert "work finished" in runtime.calls
        return original_set(limit)

    runtime.set_wired_limit = set_limit

    def predict(**kwargs):
        started.set()
        assert release.wait(timeout=2)
        runtime.calls.append("work finished")
        return {"answers": {}}

    agent.predict = predict
    with ThreadPoolExecutor(max_workers=1) as client:
        with service_runtime(agent, warmup=False, stream=io.StringIO()) as serving:
            pending = client.submit(serving.predict)
            assert started.wait(timeout=2)
            timer = threading.Timer(0.04, release.set)
            timer.start()
        pending.result(timeout=2)
        timer.join(timeout=2)
    assert runtime.limit == GIB
