"""Verification utilities: indexed distance lookup and embedding accessors."""
from __future__ import annotations

from typing import List, Tuple

import torch

# Module-level storage to keep API compatibility while app owns the data lifecycle
_employee_index = None
_employee_offsets: List[Tuple[str, int, int]] = []


def set_embedding_index(index: torch.Tensor, offsets: List[Tuple[str, int, int]]):
    """Cache the prebuilt embedding index and offsets for fast lookups."""
    global _employee_index, _employee_offsets
    _employee_index = index
    _employee_offsets = offsets or []


def verify_embedding_fast(trial_embedding: torch.Tensor):
    """Fast per-identity min-distance lookup using the cached index.

    Returns (best_name, best_distance, [(name, min_distance), ...]).
    If no index is available, returns (None, inf, []).
    """
    if _employee_index is None or not _employee_offsets:
        return None, float("inf"), []

    if trial_embedding.device != _employee_index.device:
        trial_embedding = trial_embedding.to(_employee_index.device, non_blocking=True)

    dist_vec = torch.cdist(trial_embedding, _employee_index, p=2)[0]

    results = []
    best_name = None
    best_dist = float("inf")
    for name, start, end in _employee_offsets:
        if end <= start:
            continue
        slice_min = dist_vec[start:end].min().item()
        results.append((name, slice_min))
        if slice_min < best_dist:
            best_dist = slice_min
            best_name = name

    return best_name, best_dist, results
