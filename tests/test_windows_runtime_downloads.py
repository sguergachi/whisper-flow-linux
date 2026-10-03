import io
import ssl
import urllib.error
from pathlib import Path

import pytest

from whisper_flow import backend as bm


@pytest.fixture
def windows(tmp_path, monkeypatch):
    monkeypatch.setattr(bm.sys, "platform", "win32")
    monkeypatch.setattr(bm.platform, "system", lambda: "Windows")
    monkeypatch.setenv("WINDIR", str(tmp_path / "Windows"))
    monkeypatch.setattr(bm, "bundled_dir", lambda: tmp_path / "bundle")
    return tmp_path


def test_app_local_runtime_works_without_system_runtime_or_elevation(windows, monkeypatch):
    source = windows / "bundle" / "native-runtime"
    source.mkdir(parents=True)
    for name in bm._MSVC_REQUIRED + ("msvcp140_atomic_wait.dll",):
        (source / name).write_bytes(b"official redistributable fixture")
    target = windows / "runtime" / "cuda"
    monkeypatch.setattr(bm.subprocess, "run", lambda *a, **k: pytest.fail("started runtime installer"))
    bm._prepare_engine_runtime(target)
    assert {p.name for p in target.iterdir()} == {p.name for p in source.iterdir()}
    assert not (windows / "Windows").exists()


def test_32_bit_or_partial_runtime_is_not_a_64_bit_runtime(windows):
    wow = windows / "Windows" / "SysWOW64"
    wow.mkdir(parents=True)
    for name in bm._MSVC_REQUIRED:
        (wow / name).touch()
    assert bm._is_vcredist_available() is False
    system = windows / "Windows" / "System32"
    system.mkdir()
    (system / "vcruntime140.dll").touch()
    assert bm._is_vcredist_available() is False
    with pytest.raises(RuntimeError, match="latest whisper-flow package"):
        bm._prepare_engine_runtime(windows / "engine")


class Response:
    def __init__(self, data, headers, status=200, fail=False):
        self.stream = io.BytesIO(data)
        self.headers, self.status, self.fail = headers, status, fail
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def read(self, size):
        data = self.stream.read(size)
        if not data and self.fail:
            raise OSError("connection reset")
        return data


def responses(monkeypatch, values):
    requests = []
    values = iter(values)
    def open_url(request, **kwargs):
        requests.append(request)
        result = next(values)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(bm.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(bm.time, "sleep", lambda *_: None)
    return requests


def test_interrupted_model_download_resumes_with_validator(tmp_path, monkeypatch):
    requests = responses(monkeypatch, [
        Response(b"abc", {"Content-Length": "6", "ETag": '"model-v1"'}, fail=True),
        Response(b"def", {"Content-Range": "bytes 3-5/6"}, 206),
    ])
    dest = tmp_path / "model.bin"
    bm._download("https://example.invalid/model", dest)
    assert dest.read_bytes() == b"abcdef"
    assert requests[1].get_header("Range") == "bytes=3-"
    assert requests[1].get_header("If-range") == '"model-v1"'
    assert not Path(str(dest) + ".part").exists()


def test_changed_model_or_ignored_range_replaces_partial_bytes(tmp_path, monkeypatch):
    responses(monkeypatch, [
        Response(b"abc", {"Content-Length": "6", "ETag": '"old"'}, fail=True),
        Response(b"newfile", {"Content-Length": "7", "ETag": '"new"'}),
    ])
    dest = tmp_path / "model.bin"
    bm._download("https://example.invalid/model", dest)
    assert dest.read_bytes() == b"newfile"


def test_short_download_never_replaces_existing_model(tmp_path, monkeypatch):
    responses(monkeypatch, [Response(b"short", {"Content-Length": "10"}) for _ in range(3)])
    dest = tmp_path / "model.bin"
    dest.write_bytes(b"previous complete model")
    with pytest.raises(OSError, match="Incomplete download"):
        bm._download("https://example.invalid/model", dest)
    assert dest.read_bytes() == b"previous complete model"


@pytest.mark.parametrize("error", [
    urllib.error.HTTPError("https://example.invalid", 403, "denied", {}, None),
    urllib.error.URLError(ssl.SSLCertVerificationError("untrusted certificate")),
])
def test_download_respects_denied_access_and_tls_verification(tmp_path, monkeypatch, error):
    requests = responses(monkeypatch, [error])
    with pytest.raises(type(error)):
        bm._download("https://example.invalid/model", tmp_path / "model.bin")
    assert len(requests) == 1
