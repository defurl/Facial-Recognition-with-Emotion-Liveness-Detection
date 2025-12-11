"""Benchmark verification paths (fast cdist vs naive loop) on the current employee DB.

Usage:
    python artifacts/scripts/bench_verification.py --trials 50 --device cuda

The script loads `artifacts/outputs/employee_db.pt`, builds the embedding index, and times
both the vectorized cdist path and a naive per-identity loop. It also reports
accept rates using the GUI threshold (with small-DB adjustment).
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from src.config import (
    DEVICE,
    EMPLOYEE_DB_PATH,
    OPTIMAL_THRESHOLD_GUI,
    UNRECOGNIZED_DISTANCE_MULTIPLIER,
    USE_MULTI_EMBEDDING,
)
from src.runtime.db import build_embedding_index, load_employee_db


def _select_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return device


def _flatten_embeddings(db: Dict[str, torch.Tensor]) -> Iterable[Tuple[str, torch.Tensor]]:
    """Yield (name, embedding_tensor) for each stored embedding."""
    for name, stored in db.items():
        if USE_MULTI_EMBEDDING and isinstance(stored, list):
            for emb in stored:
                yield name, emb
        else:
            emb = stored[0] if isinstance(stored, list) else stored
            yield name, emb


def _prepare_trials(db: Dict[str, torch.Tensor], trials: int) -> List[Tuple[str, torch.Tensor]]:
    """Create a mix of near-positives and random negatives for benchmarking."""
    embeddings = list(_flatten_embeddings(db))
    if not embeddings:
        return []

    random.shuffle(embeddings)
    trials_list: List[Tuple[str, torch.Tensor]] = []

    # Near-positives: perturb existing embeddings slightly
    for name, emb in embeddings[: max(1, trials // 2)]:
        noise = torch.randn_like(emb) * 0.02
        trials_list.append((name, emb + noise))

    # Negatives: random vectors
    for _ in range(trials - len(trials_list)):
        rnd = torch.randn_like(embeddings[0][1])
        trials_list.append(("__negative__", rnd))

    random.shuffle(trials_list)
    return trials_list


def _verify_fast(trial: torch.Tensor, index: torch.Tensor, offsets: List[Tuple[str, int, int]]):
    """Vectorized min-distance lookup using prebuilt index."""
    dist_vec = torch.cdist(trial.unsqueeze(0), index, p=2)[0]
    best_name = None
    best_dist = float("inf")
    for name, start, end in offsets:
        if end <= start:
            continue
        slice_min = dist_vec[start:end].min().item()
        if slice_min < best_dist:
            best_dist = slice_min
            best_name = name
    return best_name, best_dist


def _verify_naive(trial: torch.Tensor, db: Dict[str, torch.Tensor]):
    best_name = None
    best_dist = float("inf")
    target_device = trial.device
    for name, stored in db.items():
        if USE_MULTI_EMBEDDING and isinstance(stored, list):
            dists = []
            for emb in stored:
                emb_dev = emb.to(target_device, non_blocking=True)
                dists.append(torch.nn.functional.pairwise_distance(trial, emb_dev).item())
            dist = min(dists) if dists else float("inf")
        else:
            emb = stored[0] if isinstance(stored, list) else stored
            emb_dev = emb.to(target_device, non_blocking=True)
            dist = torch.nn.functional.pairwise_distance(trial, emb_dev).item()
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name, best_dist


def main():
    parser = argparse.ArgumentParser(description="Benchmark verification speed and thresholds")
    parser.add_argument("--db", type=Path, default=EMPLOYEE_DB_PATH, help="Path to employee_db.pt")
    parser.add_argument("--device", type=str, default=str(DEVICE), help="Device to benchmark on (cpu or cuda)")
    parser.add_argument("--trials", type=int, default=50, help="Number of trial verifications")
    args = parser.parse_args()

    device = _select_device(args.device)
    db = load_employee_db(args.db)

    if not db:
        print(f"[INFO] Database empty at {args.db}; nothing to benchmark.")
        return

    index, offsets = build_embedding_index(db, device=device)
    if index is None:
        print("[INFO] No embeddings found after indexing; nothing to benchmark.")
        return

    trials = _prepare_trials(db, args.trials)
    if not trials:
        print("[INFO] No trial embeddings could be prepared; aborting.")
        return

    fast_times: List[float] = []
    naive_times: List[float] = []
    agree = 0
    accept_count = 0

    adjusted_threshold = OPTIMAL_THRESHOLD_GUI
    if len(db) <= 2:
        adjusted_threshold *= UNRECOGNIZED_DISTANCE_MULTIPLIER

    for expected, trial in trials:
        trial = trial.detach().to(device)

        # Fast path timing
        t0 = time.perf_counter()
        best_fast, dist_fast = _verify_fast(trial, index, offsets)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fast_times.append((time.perf_counter() - t0) * 1000)

        # Naive path timing
        t0 = time.perf_counter()
        best_naive, dist_naive = _verify_naive(trial, db)
        if device.type == "cuda":
            torch.cuda.synchronize()
        naive_times.append((time.perf_counter() - t0) * 1000)

        if best_fast == best_naive:
            agree += 1

        if dist_fast < adjusted_threshold:
            accept_count += 1

    print("\n=== Verification Benchmark ===")
    print(f"DB size: {len(db)} identities, device: {device}")
    print(f"Trials: {len(trials)} (near-positives + random negatives)")
    print(f"Adjusted threshold: {adjusted_threshold:.4f} (GUI {OPTIMAL_THRESHOLD_GUI:.4f}, multiplier {UNRECOGNIZED_DISTANCE_MULTIPLIER})")
    print(f"Agreement (fast vs naive): {agree}/{len(trials)}")
    print(f"Accept rate (fast, dist < threshold): {accept_count}/{len(trials)}")
    print(f"Fast avg: {sum(fast_times)/len(fast_times):.3f} ms")
    print(f"Naive avg: {sum(naive_times)/len(naive_times):.3f} ms")
    speedup = (sum(naive_times) / sum(fast_times)) if sum(fast_times) > 0 else float("inf")
    print(f"Speedup (naive/fast): {speedup:.2f}x")


if __name__ == "__main__":
    main()
