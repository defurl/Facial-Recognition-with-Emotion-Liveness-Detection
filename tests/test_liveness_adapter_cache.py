import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root and src are on sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

from src.pipeline.liveness_adapter import LivenessAdapter


def _make_dummy_face(size=112):
    return np.zeros((size, size, 3), dtype=np.uint8)


def _val_transform(pil_image):
    # Minimal transform: convert PIL to CHW float tensor in [0,1]
    arr = np.array(pil_image).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


class DummyModel:
    def __call__(self, tensor, mode="metric"):
        return torch.zeros((1, 128))


def test_early_identity_hits_cache_then_pick_cached_identity():
    adapter = LivenessAdapter(use_blink_only=True)
    face = _make_dummy_face()
    model = DummyModel()
    employee_db = {"alice": torch.zeros((1, 128))}

    early_id = adapter.compute_early_identity(
        face,
        model,
        _val_transform,
        employee_db,
        current_threshold=0.7,
        relaxed_multiplier=1.5,
        confidence_floor=1.0,
        use_multi_embedding=False,
    )
    assert early_id == "alice"

    cache_hit = adapter.pick_cached_identity(early_id, {"alice"}, employee_db)
    assert cache_hit == "alice"


def test_pick_cached_identity_fallback_without_early_match():
    adapter = LivenessAdapter(use_blink_only=True)
    employee_db = {"bob": torch.ones((1, 128))}
    cache_hit = adapter.pick_cached_identity(None, {"bob"}, employee_db)
    assert cache_hit == "bob"


def test_pick_cached_identity_no_cache():
    adapter = LivenessAdapter(use_blink_only=True)
    employee_db = {"carol": torch.ones((1, 128))}
    cache_hit = adapter.pick_cached_identity(None, set(), employee_db)
    assert cache_hit is None
