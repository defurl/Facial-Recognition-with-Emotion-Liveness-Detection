"""
Deep kNN for Face Recognition with Uncertainty Quantification

Enhanced helper functions for kNN-based face retrieval with:
- Confidence scoring based on neighbor agreement
- Uncertainty quantification
- Local outlier detection
- Detailed explanations for decisions

Author: Enhanced for Hackathon
Date: November 2025
"""


from typing import Iterable, Optional, Sequence, Tuple, Dict, List
from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor


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


def knn_confidence_score(
    neighbor_labels: np.ndarray,
    neighbor_distances: np.ndarray,
    predicted_label: int
) -> Dict[str, float]:
    """
    Compute confidence score based on kNN neighbor consistency.
    
    High confidence when:
    - Most neighbors agree on the same label
    - Distances are small and consistent (low variance)
    - Clear separation between matching and non-matching neighbors
    
    Args:
        neighbor_labels: Array of k neighbor labels
        neighbor_distances: Array of k neighbor distances
        predicted_label: The predicted identity label
    
    Returns:
        Dictionary with confidence metrics:
            - confidence: Overall confidence score [0, 1]
            - agreement: Fraction of neighbors that match predicted label
            - avg_distance: Mean distance to neighbors
            - distance_variance: Variance in neighbor distances
            - separation: Distance gap between matching and non-matching neighbors
    """
    neighbor_labels = np.asarray(neighbor_labels)
    neighbor_distances = np.asarray(neighbor_distances)
    
    # 1. Agreement: fraction of neighbors matching predicted label
    matches = (neighbor_labels == predicted_label)
    agreement = matches.sum() / len(neighbor_labels)
    
    # 2. Average distance
    avg_distance = neighbor_distances.mean()
    
    # 3. Distance variance (lower is better)
    distance_variance = neighbor_distances.var()
    
    # 4. Separation: gap between matching and non-matching neighbors
    if matches.any() and (~matches).any():
        avg_match_dist = neighbor_distances[matches].mean()
        avg_nonmatch_dist = neighbor_distances[~matches].mean()
        separation = avg_nonmatch_dist - avg_match_dist
        separation_normalized = 1 / (1 + np.exp(-5 * separation))  # Sigmoid
    else:
        separation = 0.0
        separation_normalized = 1.0 if agreement == 1.0 else 0.0
    
    # 5. Combined confidence score
    # High agreement + low distance + low variance + good separation = high confidence
    distance_score = np.exp(-2 * avg_distance)  # Exponential decay
    variance_score = 1 / (1 + distance_variance)  # Inverse
    
    confidence = (
        0.4 * agreement +  # Most important
        0.25 * distance_score +
        0.2 * variance_score +
        0.15 * separation_normalized
    )
    
    return {
        'confidence': float(confidence),
        'agreement': float(agreement),
        'avg_distance': float(avg_distance),
        'distance_variance': float(distance_variance),
        'separation': float(separation),
        'num_matches': int(matches.sum()),
        'num_neighbors': len(neighbor_labels)
    }


def knn_predict_with_confidence(
    neighbor_labels: np.ndarray,
    neighbor_distances: np.ndarray,
    return_explanation: bool = False
) -> Tuple[int, float, Optional[Dict]]:
    """
    Predict identity with confidence score based on kNN neighbors.
    
    Args:
        neighbor_labels: Array of k neighbor labels (shape: [k])
        neighbor_distances: Array of k neighbor distances (shape: [k])
        return_explanation: Whether to return detailed explanation
    
    Returns:
        Tuple of (predicted_label, confidence, explanation_dict)
        - predicted_label: Most common label among neighbors
        - confidence: Confidence score [0, 1]
        - explanation_dict: Optional detailed breakdown
    """
    neighbor_labels = np.asarray(neighbor_labels)
    neighbor_distances = np.asarray(neighbor_distances)
    
    # Majority vote
    label_counts = Counter(neighbor_labels)
    predicted_label = label_counts.most_common(1)[0][0]
    
    # Compute confidence
    conf_dict = knn_confidence_score(neighbor_labels, neighbor_distances, predicted_label)
    confidence = conf_dict['confidence']
    
    explanation = None
    if return_explanation:
        explanation = {
            **conf_dict,
            'predicted_label': int(predicted_label),
            'vote_distribution': dict(label_counts),
            'neighbor_labels': neighbor_labels.tolist(),
            'neighbor_distances': neighbor_distances.tolist(),
            'confidence_level': _classify_confidence(confidence)
        }
    
    return int(predicted_label), float(confidence), explanation


def _classify_confidence(confidence: float) -> str:
    """Classify confidence into human-readable levels."""
    if confidence >= 0.8:
        return "Very High"
    elif confidence >= 0.6:
        return "High"
    elif confidence >= 0.4:
        return "Moderate"
    elif confidence >= 0.2:
        return "Low"
    else:
        return "Very Low"


def detect_outliers_lof(
    embeddings: np.ndarray,
    contamination: float = 0.1,
    n_neighbors: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outlier embeddings using Local Outlier Factor.
    
    Useful for identifying faces that don't belong to any registered identity.
    
    Args:
        embeddings: Array of embeddings (shape: [n_samples, embedding_dim])
        contamination: Expected proportion of outliers
        n_neighbors: Number of neighbors for LOF
    
    Returns:
        Tuple of (outlier_predictions, outlier_scores)
        - outlier_predictions: 1 for inliers, -1 for outliers
        - outlier_scores: Negative outlier factor (lower = more outlier-like)
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    
    if len(embeddings) < n_neighbors:
        n_neighbors = max(2, len(embeddings) - 1)
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_predictions = lof.fit_predict(embeddings)
    outlier_scores = lof.negative_outlier_factor_
    
    return outlier_predictions, outlier_scores


def is_query_outlier(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    threshold_percentile: float = 10
) -> Tuple[bool, float, str]:
    """
    Check if a query embedding is an outlier compared to gallery.
    
    Args:
        query_embedding: Single query embedding (shape: [embedding_dim])
        gallery_embeddings: Gallery embeddings (shape: [n_samples, embedding_dim])
        threshold_percentile: Percentile threshold for outlier detection
    
    Returns:
        Tuple of (is_outlier, outlier_score, message)
    """
    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    gallery_embeddings = np.asarray(gallery_embeddings, dtype=np.float32)
    
    # Compute distances to all gallery embeddings
    from scipy.spatial.distance import cdist
    distances = cdist(query_embedding, gallery_embeddings, metric='cosine')[0]
    
    # Statistics
    min_distance = distances.min()
    mean_distance = distances.mean()
    percentile_threshold = np.percentile(distances, threshold_percentile)
    
    # Outlier if nearest neighbor is far
    is_outlier = min_distance > percentile_threshold
    outlier_score = float(min_distance)
    
    if is_outlier:
        message = f"Face appears unfamiliar (min distance: {min_distance:.3f} > threshold: {percentile_threshold:.3f})"
    else:
        message = f"Face matches database (min distance: {min_distance:.3f})"
    
    return is_outlier, outlier_score, message


def get_knn_explanation_text(explanation: Dict) -> str:
    """
    Generate human-readable text from kNN explanation dictionary.
    
    Args:
        explanation: Dictionary from knn_predict_with_confidence
    
    Returns:
        Formatted explanation string
    """
    lines = []
    lines.append(f"🎯 Predicted Identity: {explanation['predicted_label']}")
    lines.append(f"📊 Confidence: {explanation['confidence']:.1%} ({explanation['confidence_level']})")
    lines.append(f"\n📈 Neighbor Analysis:")
    lines.append(f"   • Agreement: {explanation['num_matches']}/{explanation['num_neighbors']} neighbors match")
    lines.append(f"   • Average distance: {explanation['avg_distance']:.3f}")
    lines.append(f"   • Distance variance: {explanation['distance_variance']:.4f}")
    lines.append(f"   • Separation score: {explanation['separation']:.3f}")
    
    lines.append(f"\n🗳️ Vote Distribution:")
    for label, count in sorted(explanation['vote_distribution'].items(), key=lambda x: -x[1]):
        percent = (count / explanation['num_neighbors']) * 100
        lines.append(f"   • Identity {label}: {count} votes ({percent:.1f}%)")
    
    # Interpretation
    lines.append(f"\n💡 Interpretation:")
    if explanation['confidence'] >= 0.8:
        lines.append("   ✓ Very high confidence - strong consensus among neighbors")
    elif explanation['confidence'] >= 0.6:
        lines.append("   ✓ Good confidence - majority agreement with low distance")
    elif explanation['confidence'] >= 0.4:
        lines.append("   ⚠ Moderate confidence - some uncertainty in matching")
    else:
        lines.append("   ⚠ Low confidence - unreliable prediction, manual review recommended")
    
    return '\n'.join(lines)


__all__ = [
    "fit_knn", 
    "knn_search", 
    "evaluate_topk",
    "knn_confidence_score",
    "knn_predict_with_confidence",
    "detect_outliers_lof",
    "is_query_outlier",
    "get_knn_explanation_text"
]


if __name__ == "__main__":
    print("✓ Enhanced Deep kNN module loaded")
    print("\nNew features:")
    print("  • kNN confidence scoring")
    print("  • Uncertainty quantification")
    print("  • Outlier detection (LOF)")
    print("  • Detailed explanations")
