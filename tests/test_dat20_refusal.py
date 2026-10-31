"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

import grounding_dino_detection_pipeline as gd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/grounding_dino_detection_colab.ipynb"
BYOD_CELL = "grounding_dino_detection-17"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


STEM = "grounding_dino_detection"
EARLIER = {"records": "old", "splits": "old", "test_records": "old", "frozen_test": "old", "adapt_result": "old", "input_manifest": "old", "byod": "old", "vocabulary": "old"}


def _png(i: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (i * 7 % 256, i * 13 % 256, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip_bytes(entries: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _good(n: int = 16) -> dict:
    rows = "file,x0,y0,x1,y1,phrase\n" + "".join(f"p{i}.png,4,4,24,24,red disc\np{i}.png,30,20,50,44,blue square\n" for i in range(n))
    return {"boxes.csv": rows.encode("utf-8"), **{f"p{i}.png": _png(i) for i in range(n)}}


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    (out / f"{STEM}_adapter").mkdir(parents=True)
    for name in ("train.csv", "input_manifest.json", "evaluation_report.json", "result.json", "scene_frozen.png"):
        (out / f"{STEM}_{name}").write_text("x", encoding="utf-8")


def _namespace(extra: dict | None = None) -> dict:
    ns = {name: getattr(gd, name) for name in gd.__all__}
    ns.update({"os": os, "Path": Path, "json": json, **EARLIER, "pipe": "kept", **(extra or {})})
    return ns


def _section4(monkeypatch, tmp_path, path: str, *, use_byod: bool = True, source: str | None = None, extra: dict | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    _seed_outputs(tmp_path)
    source = source or _cell(BYOD_CELL)
    source = source[: source.index("train_records, val_records, test_records = splits")]
    if use_byod:
        source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1)
    source = source.replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1)
    ns = _namespace(extra)
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime-host\n", encoding="utf-8")
    (tmp_path / "no_csv.zip").write_bytes(_zip_bytes({"p0.png": _png(0)}))
    (tmp_path / "no_x0.zip").write_bytes(_zip_bytes({"boxes.csv": b"file,y0,x1,y1,phrase\np0.png,4,24,24,red disc\n", "p0.png": _png(0)}))
    (tmp_path / "text_coord.zip").write_bytes(_zip_bytes({"boxes.csv": b"file,x0,y0,x1,y1,phrase\np0.png,left,4,24,24,red disc\n", "p0.png": _png(0)}))
    (tmp_path / "missing_image.zip").write_bytes(_zip_bytes({"boxes.csv": b"file,x0,y0,x1,y1,phrase\np1.png,4,4,24,24,red disc\n", "p0.png": _png(0)}))
    (tmp_path / "not_image.zip").write_bytes(_zip_bytes({"boxes.csv": b"file,x0,y0,x1,y1,phrase\np0.png,4,4,24,24,red disc\n", "p0.png": b"text"}))
    (tmp_path / "too_few.zip").write_bytes(_zip_bytes(_good(3)))
    return {
        "not a zip or folder (the scripted /etc/hostname run)": ("hostname", "is neither a directory nor a zip file"),
        "missing": ("nope.zip", "does not exist in this runtime"),
        "no boxes.csv": ("no_csv.zip", "BYOD needs a boxes.csv"),
        "boxes.csv without an x0 column (KeyError)": ("no_x0.zip", "KeyError: 'x0'"),
        "a coordinate that is not a number": ("text_coord.zip", "could not convert string to float"),
        "row naming a missing image": ("missing_image.zip", "which is not among the uploaded files"),
        "undecodable image": ("not_image.zip", "not a decodable image: p0.png"),
        "too few images": ("too_few.zip", "ValueError"),
    }


CASES = [
    "not a zip or folder (the scripted /etc/hostname run)", "missing", "no boxes.csv", "boxes.csv without an x0 column (KeyError)",
    "a coordinate that is not a number", "row naming a missing image", "undecodable image", "too few images",
]


@pytest.mark.parametrize("case", CASES)
def test_byod_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    file_name, rule = _cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / file_name))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []


def test_refusal_clears_the_session(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.chdir(tmp_path)
    source = _cell(BYOD_CELL)
    source = source[: source.index("train_records, val_records, test_records = splits")].replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", "BYOD_PATH = 'nope.zip'", 1)
    ns = _namespace()
    with pytest.raises(ValueError, match="^nope.zip: BYOD_PATH"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["pipe"] == "kept"


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith("Refused: (no file): the upload dialog needs Google Colab")


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "received 0"), ({"a.zip": b"1", "b.zip": b"2"}, "a.zip, b.zip", "received 2")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: upload exactly one .zip file") and rule in line


def test_valid_zip_still_splits(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "good.zip").write_bytes(_zip_bytes(_good()))
    ns = _section4(monkeypatch, tmp_path, str(tmp_path / "good.zip"))
    assert ns["data_source"] == "BYOD (good.zip)" and sum(len(v) for v in ns["splits"].values()) == 16


def test_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "good.zip").write_bytes(_zip_bytes(_good()))

    def broken(*args, **kwargs):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "good.zip"), extra={"drop_duplicate_images": broken})
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is removed for an internal error


def test_sample_corpus_errors_are_not_refusals(monkeypatch, tmp_path):
    def offline(*args, **kwargs):
        raise OSError("no network")

    with pytest.raises(OSError, match="no network") as caught:
        _section4(monkeypatch, tmp_path, "", use_byod=False, extra={"fetch_corpus": offline})
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/etc/hostname': a file present in every fresh runtime."""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "/etc/hostname")
    assert _refusal_line(caught.value).startswith("Refused: hostname: ValueError: /etc/hostname is neither a directory nor a zip file")
