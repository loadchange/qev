"""Bounded MLX memory management and warmup for a long-lived HTTP service."""

from __future__ import annotations

import base64
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib import import_module

_MIB = 1024 * 1024
_CACHE_LIMIT = 256 * _MIB
_MEMORY_ERRORS = (ImportError, AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError)


class _ThreadedAgent:
    """Keep MLX's per-thread compiler cache warm without caching answers."""

    def __init__(self, agent, executor):
        self._agent, self._executor = agent, executor

    def __getattr__(self, name):
        return getattr(self._agent, name)

    def _call(self, method, *args, **kwargs):
        submitted = time.perf_counter()

        def invoke():
            queue_ms = (time.perf_counter() - submitted) * 1000
            return getattr(self._agent, method)(*args, **kwargs), queue_ms

        response, queue_ms = self._executor.submit(invoke).result()
        elapsed_ms = round((time.perf_counter() - submitted) * 1000, 2)
        result = dict(response)
        qev = dict(result.get("qev", {}))
        if method != "chat_completions" or "timings_ms" in qev:
            timings = dict(qev.get("timings_ms", {}))
            timings["queue"] = round(timings.get("queue", 0.) + queue_ms, 2)
            timings["total"] = elapsed_ms
            qev["timings_ms"] = timings
        if method == "chat_completions":
            qev["latency_ms"] = elapsed_ms
        else:
            result["latency_ms"] = elapsed_ms
        result["qev"] = qev
        return result

    def predict(self, *args, **kwargs):
        return self._call("predict", *args, **kwargs)

    system_one = predict

    def chat_completions(self, *args, **kwargs):
        return self._call("chat_completions", *args, **kwargs)


def _warmup_requests():
    from PIL import Image, ImageDraw

    def picture(color, shape):
        image = Image.new("RGB", (96, 96), "white")
        draw = ImageDraw.Draw(image)
        if shape == "circle":
            draw.ellipse((20, 20, 76, 76), fill=color)
        else:
            draw.rectangle((20, 20, 76, 76), fill=color)
        output = io.BytesIO()
        image.save(output, "PNG")
        return {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii"),
        }}

    images = [picture("red", "square"), picture("blue", "circle"), picture("yellow", "square")]
    return [
        ("text", {
            "state": "A customer was charged twice for the same order and requests a refund.",
            "questions": {"team": {"type": "choice", "instructions": "Choose the department that should handle this request.",
                                    "criteria": {"billing": "Billing and refunds", "technical": "Technical support", "sales": "Sales"}}},
        }),
        ("state_image", {
            "state": [{"type": "text", "text": "Look at the shape in the attached image."}, images[0]],
            "questions": {"shape": {"type": "choice", "instructions": "Which description matches the image?",
                                     "criteria": {"A": "A red square", "B": "A blue circle", "C": "A yellow square"}}},
        }),
        ("option_images", {
            "state": "Select the candidate image containing a blue circle. Compare the pictures in all three options.",
            "questions": {"image": {"type": "choice", "instructions": "Choose the option whose picture matches the requested shape and color.",
                                     "criteria": {name: {"content": [{"type": "text", "text": f"Candidate {name}"}, image]}
                                                  for name, image in zip("ABC", images, strict=True)}}},
        }),
    ]


def _warmup(agent, emit):
    for name, request in _warmup_requests():
        start = time.perf_counter()
        # Execute normal inference and discard its answer. Each subsequent HTTP
        # request still encodes its inputs and runs a fresh model forward.
        try:
            result = agent.predict(**request)
        except Exception as exc:
            emit(f"warmup {name} failed: {type(exc).__name__}: {exc}")
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        tokens = result.get("usage", {}).get("input_tokens", "unknown")
        emit(f"warmup {name}: {elapsed_ms:.1f} ms, {tokens} input tokens (answer discarded)")


@contextmanager
def service_runtime(agent, *, warmup=True, wired_memory=True, stream=None):
    """Bound idle MLX GPU buffers and keep the model resident while serving.

    MLX settings are process-wide. This context belongs around the single CLI
    server lifetime, after loading/materializing its model and before requests
    start; it must not be entered or exited per HTTP request. No system setting
    is changed. Torch, CPU, and older runtimes remain usable. Active model
    buffers are unaffected by the allocator's free-cache limit.
    MLX requests and warmup use one persistent inference thread, since compiled
    function caches belong to their calling thread, independently of streams.
    """
    stream = sys.stderr if stream is None else stream

    def emit(message):
        print(f"Qev service: {message}", file=stream, flush=True)

    mx, previous_limit, previous_cache_limit, executor = None, None, None, None
    try:
        if not wired_memory:
            emit("automatic MLX memory management disabled")
        elif agent.backend != "mlx":
            emit(f"automatic MLX memory management unchanged for {agent.backend} backend")
        else:
            try:
                mx = import_module("mlx.core")
                device = getattr(agent.model, "_device", None)
                device = mx.default_device() if device is None else device
                if not mx.metal.is_available() or device.type != mx.gpu:
                    emit("automatic MLX memory management unchanged for MLX CPU")
                else:
                    try:
                        active_before, cache_before = int(mx.get_active_memory()), int(mx.get_cache_memory())
                        emit(f"MLX allocator before: {active_before / _MIB:.0f} MiB active, {cache_before / _MIB:.0f} MiB cached")
                        previous_cache_limit = mx.set_cache_limit(_CACHE_LIMIT)
                        cache_limit = min(_CACHE_LIMIT, previous_cache_limit)
                        if previous_cache_limit < _CACHE_LIMIT:
                            mx.set_cache_limit(previous_cache_limit)
                        # Release loading buffers before selecting residency;
                        # clear_cache never frees active model weights.
                        mx.clear_cache()
                        active_after, cache_after = int(mx.get_active_memory()), int(mx.get_cache_memory())
                        emit(f"MLX allocator after: {active_after / _MIB:.0f} MiB active, {cache_after / _MIB:.0f} MiB cached; "
                             f"free-cache limit {cache_limit / _MIB:.0f} MiB")
                    except _MEMORY_ERRORS as exc:
                        emit(f"MLX allocator cache management unavailable; continuing ({type(exc).__name__}: {exc})")
                    active = int(mx.get_active_memory())
                    info = mx.device_info()
                    cap = min(int(info["max_recommended_working_set_size"]), int(info["memory_size"]) // 2)
                    if active < 0 or cap <= 0:
                        raise ValueError("invalid MLX memory accounting")
                    requested = min(active + max(512 * _MIB, active // 5), cap)
                    previous_limit = mx.set_wired_limit(requested)
                    # There is no wired-limit getter. The setter returns the
                    # former value, so immediately reinstate a larger user
                    # setting before warmup rather than overriding that choice.
                    effective = max(requested, previous_limit)
                    if previous_limit > requested:
                        mx.set_wired_limit(previous_limit)
                    policy = "preserved existing limit" if previous_limit > requested else "automatic limit"
                    emit(f"MLX memory residency: {effective / _MIB:.0f} MiB ({policy}; {active / _MIB:.0f} MiB active)")
                    if requested < active and effective < active:
                        emit("residency cap is below active model memory; some buffers may remain pageable")
            except _MEMORY_ERRORS as exc:
                emit(f"MLX memory residency unavailable; continuing ({type(exc).__name__}: {exc})")
        service_agent = agent
        if agent.backend == "mlx":
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qev-inference")
            service_agent = _ThreadedAgent(agent, executor)
            emit("MLX warmup and requests share one persistent inference thread")
        if warmup:
            _warmup(service_agent, emit)
        else:
            emit("startup warmup disabled")
        yield service_agent
    finally:
        try:
            if executor is not None:
                executor.shutdown(wait=True)
        finally:
            if previous_limit is not None or previous_cache_limit is not None:
                # Request streams synchronize before returning. Finish all
                # executor work before restoring process-wide memory settings.
                try:
                    mx.synchronize()
                except _MEMORY_ERRORS as exc:
                    emit(f"MLX shutdown synchronization unavailable ({type(exc).__name__}: {exc})")
                for name, previous in (("cache", previous_cache_limit), ("wired", previous_limit)):
                    if previous is not None:
                        try:
                            getattr(mx, f"set_{name}_limit")(previous)
                        except _MEMORY_ERRORS as exc:
                            emit(f"MLX {name} limit restoration failed ({type(exc).__name__}: {exc})")
