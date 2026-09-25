"""Keep tests independent of the developer's installed models and running server."""
import pytest


@pytest.fixture(autouse=True)
def isolated_qev_environment(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("QEV_HOME", str(tmp_path_factory.mktemp("qev-home")))
    # Port 9 (discard) refuses connections at once, so server discovery never
    # finds a real local qev serve during tests.
    monkeypatch.setenv("QEV_SERVER", "http://127.0.0.1:9")
    monkeypatch.delenv("QEV_MODEL", raising=False)
    monkeypatch.delenv("QEV_PROCESSOR", raising=False)
