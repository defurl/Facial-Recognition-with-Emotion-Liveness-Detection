"""Super-small helper functions for Deep kNN on embeddings."""


from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def fit_knn(embeddings: np.ndarray, metric: str = "cosine", k: int = 5) -> NearestNeighbors:
    """Train a scikit-learn kNN model on the provided embeddings."""

    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("Embeddings must have shape [num_samples, dim]")

    knn = NearestNeighbors(metric=metric, n_neighbors=k)
    knn.fit(emb)
    return knn


def knn_search(
    knn: NearestNeighbors,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    k: Optional[int] = None,
    exclude_self: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return neighbour distances, labels, and indices for each query vector."""

    gallery_embeddings = np.asarray(gallery_embeddings, dtype=np.float32)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    gallery_labels = np.asarray(gallery_labels, dtype=np.int64)

    neighbours = k or knn.n_neighbors
    extra = 1 if exclude_self else 0
    distances, indices = knn.kneighbors(query_embeddings, n_neighbors=min(neighbours + extra, len(gallery_embeddings)))

    if exclude_self:
        cleaned_distances = []
        cleaned_labels = []
        cleaned_indices = []
        for dist_row, idx_row in zip(distances, indices):
            keep_dist = []
            keep_lab = []
            keep_idx = []
            for dist, idx in zip(dist_row, idx_row):
                if len(keep_dist) >= neighbours:
                    break
                if dist <= 1e-6:
                    continue
                keep_dist.append(float(dist))
                keep_lab.append(int(gallery_labels[idx]))
                keep_idx.append(int(idx))
            if not keep_dist:  # identical vectors, fallback
                keep_dist = [float(dist_row[0])]
                keep_lab = [int(gallery_labels[idx_row[0]])]
                keep_idx = [int(idx_row[0])]
            cleaned_distances.append(keep_dist)
            cleaned_labels.append(keep_lab)
            cleaned_indices.append(keep_idx)
        distances = np.asarray(cleaned_distances, dtype=np.float32)
        neighbour_labels = np.asarray(cleaned_labels, dtype=np.int64)
        neighbour_indices = np.asarray(cleaned_indices, dtype=np.int64)
    else:
        distances = distances[:, :neighbours].astype(np.float32)
        truncated_indices = indices[:, :neighbours]
        neighbour_labels = gallery_labels[truncated_indices]
        neighbour_indices = truncated_indices.astype(np.int64)

    return distances, neighbour_labels, neighbour_indices


def evaluate_topk(
    neighbour_labels: np.ndarray,
    query_labels: np.ndarray,
    ks: Iterable[int] = (1, 3, 5),
) -> dict:
    """Compute simple Top-K accuracies given neighbour labels."""

    query_labels = np.asarray(query_labels, dtype=np.int64)
    ks = sorted({int(k) for k in ks if k > 0})
    results = {}
    if neighbour_labels.shape[0] != query_labels.shape[0]:
        raise ValueError("Query label count does not match neighbour predictions")

    for k in ks:
        k_labels = neighbour_labels[:, :k]
        in_topk = np.array([truth in row for truth, row in zip(query_labels, k_labels)])
        results[f"top{k}_accuracy"] = float(np.mean(in_topk))

    results["top1_accuracy"] = float(np.mean(neighbour_labels[:, 0] == query_labels))
    return results


__all__ = ["fit_knn", "knn_search", "evaluate_topk"]
