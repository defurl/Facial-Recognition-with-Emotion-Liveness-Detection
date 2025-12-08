import sys
from pathlib import Path

# Ensure project root and src are on sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

from src.pipeline.processing import process_frame_shell


def test_process_frame_shell_runs_callbacks_in_order():
    calls = []

    def detect_cb(frame):
        calls.append("detect")
        return {"faces": [(1, 2, 3, 4)], "meta": "ok"}

    def process_cb(frame, face_idx, bbox, ctx):
        calls.append(("process", face_idx, bbox, ctx.get("meta")))
        return {"face_idx": face_idx, "bbox": bbox}

    def update_cb(frame, results, ctx):
        calls.append(("update", len(results), ctx.get("meta")))

    frame = object()
    results = process_frame_shell(frame, detect_faces_cb=detect_cb, process_face_cb=process_cb, update_ui_cb=update_cb)

    assert len(results) == 1
    assert calls[0] == "detect"
    assert calls[1][0] == "process"
    assert calls[2][0] == "update"


def test_process_frame_shell_handles_plain_face_list_and_errors():
    def detect_cb(_frame):
        return [(0, 0, 10, 10)]

    def process_cb(_frame, face_idx, bbox, _ctx):
        raise ValueError("boom")

    captured = {}

    def update_cb(_frame, results, ctx):
        captured["results"] = results
        captured["ctx"] = ctx

    process_frame_shell(object(), detect_faces_cb=detect_cb, process_face_cb=process_cb, update_ui_cb=update_cb)

    assert captured["ctx"]["faces"] == [(0, 0, 10, 10)]
    assert len(captured["results"]) == 1
    assert "error" in captured["results"][0]
