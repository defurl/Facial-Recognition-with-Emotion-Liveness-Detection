"""Runtime settings helpers (e.g., GUI threshold persistence)."""
from __future__ import annotations

import json
from pathlib import Path

from src.config import OUTPUT_DIR

_GUI_THRESH_FILE = OUTPUT_DIR / "gui_threshold.json"


def load_gui_threshold(default_val: float) -> float:
    """Load GUI threshold from disk, falling back to default on error."""
    try:
        if _GUI_THRESH_FILE.exists():
            data = json.loads(_GUI_THRESH_FILE.read_text())
            return float(data.get("threshold", default_val))
    except Exception as e:
        print(f"Warning: could not load GUI threshold: {e}")
    return default_val


def save_gui_threshold(value: float) -> None:
    """Persist GUI threshold to disk (best-effort)."""
    try:
        OUTPUT_DIR.mkdir(exist_ok=True)
        _GUI_THRESH_FILE.write_text(json.dumps({"threshold": float(value)}))
    except Exception as e:
        print(f"Warning: could not save GUI threshold: {e}")
