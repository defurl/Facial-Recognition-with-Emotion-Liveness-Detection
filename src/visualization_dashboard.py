"""
Real-time t-SNE Visualization Dashboard for Face Recognition

Interactive dashboard for visualizing embedding space with:
- Real-time t-SNE projection of registered faces
- Query face highlighting with nearest neighbors
- Cluster quality metrics
- Embedding drift tracking
- Interactive face exploration

Author: Enhanced for Hackathon
Date: November 2025
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional
import cv2
from datetime import datetime
from collections import defaultdict


class EmbeddingDashboard:
    """
    Real-time dashboard for visualizing face embeddings.
    """
    
    def __init__(self, 
                 perplexity: int = 30,
                 n_components: int = 2,
                 random_state: int = 42,
                 use_pca: bool = True,
                 pca_components: int = 50):
        """
        Initialize the dashboard.
        
        Args:
            perplexity: t-SNE perplexity parameter
            n_components: Number of t-SNE dimensions (usually 2)
            random_state: Random seed for reproducibility
            use_pca: Whether to apply PCA before t-SNE
            pca_components: Number of PCA components
        """
        self.perplexity = perplexity
        self.n_components = n_components
        self.random_state = random_state
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        # Components
        self.pca = None
        self.tsne = None
        
        # Data storage
        self.embeddings = None
        self.labels = None
        self.names = None
        self.embeddings_2d = None
        
        # History for drift tracking
        self.embedding_history = defaultdict(list)
        self.timestamp_history = defaultdict(list)
        
        # Cluster metrics
        self.cluster_metrics = {}
    
    def fit(self, embeddings: np.ndarray, labels: np.ndarray, 
            names: Optional[List[str]] = None):
        """
        Fit t-SNE on gallery embeddings.
        
        Args:
            embeddings: Array of embeddings (n_samples, embedding_dim)
            labels: Array of identity labels (n_samples,)
            names: Optional list of identity names
        """
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.names = names or [f"ID_{i}" for i in range(len(np.unique(labels)))]
        
        # Apply PCA first if specified
        if self.use_pca and self.embeddings.shape[1] > self.pca_components:
            print(f"Applying PCA: {self.embeddings.shape[1]} → {self.pca_components} dims")
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            embeddings_reduced = self.pca.fit_transform(self.embeddings)
        else:
            embeddings_reduced = self.embeddings
        
        # Fit t-SNE
        print(f"Fitting t-SNE with perplexity={self.perplexity}...")
        self.tsne = TSNE(
            n_components=self.n_components,
            perplexity=min(self.perplexity, len(self.embeddings) - 1),
            random_state=self.random_state,
            n_iter=1000,
            verbose=0
        )
        self.embeddings_2d = self.tsne.fit_transform(embeddings_reduced)
        
        # Compute cluster metrics
        self._compute_cluster_metrics()
        
        print(f"✓ Dashboard fitted with {len(self.embeddings)} embeddings")
    
    def transform_query(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Transform a new query embedding to 2D space.
        
        Note: t-SNE doesn't have a native transform method, so we approximate
        by finding nearest neighbors in the original space and interpolating.
        
        Args:
            query_embedding: Single embedding vector (embedding_dim,)
        
        Returns:
            2D coordinates (2,)
        """
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Apply PCA if used
        if self.pca is not None:
            query_reduced = self.pca.transform(query_embedding)
        else:
            query_reduced = query_embedding
        
        # Find k nearest neighbors in original space
        k = min(5, len(self.embeddings))
        distances = cdist(query_reduced, 
                         self.pca.transform(self.embeddings) if self.pca else self.embeddings,
                         metric='euclidean')[0]
        nearest_indices = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_indices]
        
        # Inverse distance weighting for interpolation
        weights = 1 / (nearest_distances + 1e-6)
        weights = weights / weights.sum()
        
        # Weighted average of nearest neighbors' 2D positions
        query_2d = (weights[:, None] * self.embeddings_2d[nearest_indices]).sum(axis=0)
        
        return query_2d
    
    def _compute_cluster_metrics(self):
        """Compute quality metrics for embedding clusters."""
        if len(np.unique(self.labels)) < 2:
            return
        
        try:
            # Silhouette score (higher is better, range [-1, 1])
            silhouette = silhouette_score(self.embeddings_2d, self.labels)
            
            # Davies-Bouldin index (lower is better)
            davies_bouldin = davies_bouldin_score(self.embeddings_2d, self.labels)
            
            # Intra-cluster vs inter-cluster distances
            unique_labels = np.unique(self.labels)
            intra_distances = []
            inter_distances = []
            
            for label in unique_labels:
                mask = self.labels == label
                cluster_embeddings = self.embeddings_2d[mask]
                
                if len(cluster_embeddings) > 1:
                    # Intra: distances within cluster
                    intra_dist = cdist(cluster_embeddings, cluster_embeddings).mean()
                    intra_distances.append(intra_dist)
                
                # Inter: distances to other clusters
                other_embeddings = self.embeddings_2d[~mask]
                if len(other_embeddings) > 0:
                    inter_dist = cdist(cluster_embeddings, other_embeddings).mean()
                    inter_distances.append(inter_dist)
            
            avg_intra = np.mean(intra_distances) if intra_distances else 0
            avg_inter = np.mean(inter_distances) if inter_distances else 0
            separation_ratio = avg_inter / (avg_intra + 1e-6)
            
            self.cluster_metrics = {
                'silhouette_score': float(silhouette),
                'davies_bouldin_index': float(davies_bouldin),
                'avg_intra_distance': float(avg_intra),
                'avg_inter_distance': float(avg_inter),
                'separation_ratio': float(separation_ratio),
                'num_clusters': int(len(unique_labels)),
                'num_samples': int(len(self.embeddings))
            }
        except Exception as e:
            print(f"Warning: Could not compute cluster metrics: {e}")
            self.cluster_metrics = {}
    
    def plot(self, 
             query_embedding: Optional[np.ndarray] = None,
             query_name: Optional[str] = None,
             query_label: Optional[int] = None,
             neighbor_indices: Optional[List[int]] = None,
             figsize: Tuple[int, int] = (12, 10),
             title: str = "Face Embedding Space (t-SNE)",
             show_metrics: bool = True,
             save_path: Optional[str] = None) -> np.ndarray:
        """
        Plot the embedding space with optional query highlighting.
        
        Args:
            query_embedding: Optional query embedding to highlight
            query_name: Name of query identity
            query_label: Label of query identity
            neighbor_indices: Indices of nearest neighbors to highlight
            figsize: Figure size
            title: Plot title
            show_metrics: Whether to show cluster metrics
            save_path: Optional path to save plot
        
        Returns:
            Plot as numpy array (H, W, 3)
        """
        fig = Figure(figsize=figsize)
        canvas = FigureCanvasAgg(fig)
        
        if show_metrics:
            # Create subplots: main plot + metrics
            ax_main = fig.add_subplot(1, 2, 1)
            ax_metrics = fig.add_subplot(1, 2, 2)
        else:
            ax_main = fig.add_subplot(1, 1, 1)
            ax_metrics = None
        
        # Main scatter plot
        unique_labels = np.unique(self.labels)
        colors = plt.cm.get_cmap('tab20', len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            ax_main.scatter(
                self.embeddings_2d[mask, 0],
                self.embeddings_2d[mask, 1],
                c=[colors(i)],
                label=self.names[label] if label < len(self.names) else f"ID_{label}",
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidths=0.5
            )
        
        # Highlight query if provided
        if query_embedding is not None:
            query_2d = self.transform_query(query_embedding)
            ax_main.scatter(
                query_2d[0], query_2d[1],
                c='red',
                marker='*',
                s=500,
                label=f'Query: {query_name or "Unknown"}',
                edgecolors='black',
                linewidths=2,
                zorder=10
            )
            
            # Draw lines to nearest neighbors
            if neighbor_indices is not None:
                for idx in neighbor_indices:
                    ax_main.plot(
                        [query_2d[0], self.embeddings_2d[idx, 0]],
                        [query_2d[1], self.embeddings_2d[idx, 1]],
                        'r--',
                        alpha=0.5,
                        linewidth=1.5,
                        zorder=5
                    )
                # Highlight neighbors
                neighbor_coords = self.embeddings_2d[neighbor_indices]
                ax_main.scatter(
                    neighbor_coords[:, 0],
                    neighbor_coords[:, 1],
                    facecolors='none',
                    edgecolors='red',
                    s=200,
                    linewidths=3,
                    zorder=8
                )
        
        ax_main.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax_main.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax_main.set_title(title, fontsize=14, fontweight='bold')
        ax_main.legend(loc='best', fontsize=8, ncol=2)
        ax_main.grid(True, alpha=0.3)
        
        # Metrics panel
        if show_metrics and ax_metrics is not None:
            ax_metrics.axis('off')
            
            metrics_text = "📊 Cluster Quality Metrics\n"
            metrics_text += "=" * 40 + "\n\n"
            
            if self.cluster_metrics:
                metrics_text += f"Silhouette Score: {self.cluster_metrics['silhouette_score']:.3f}\n"
                metrics_text += f"  (Range: [-1, 1], higher=better)\n\n"
                
                metrics_text += f"Davies-Bouldin Index: {self.cluster_metrics['davies_bouldin_index']:.3f}\n"
                metrics_text += f"  (Lower is better)\n\n"
                
                metrics_text += f"Avg Intra-cluster Distance: {self.cluster_metrics['avg_intra_distance']:.3f}\n"
                metrics_text += f"Avg Inter-cluster Distance: {self.cluster_metrics['avg_inter_distance']:.3f}\n"
                metrics_text += f"Separation Ratio: {self.cluster_metrics['separation_ratio']:.2f}\n"
                metrics_text += f"  (Higher=better separation)\n\n"
                
                metrics_text += f"Number of Identities: {self.cluster_metrics['num_clusters']}\n"
                metrics_text += f"Total Samples: {self.cluster_metrics['num_samples']}\n\n"
                
                # Interpretation
                silhouette = self.cluster_metrics['silhouette_score']
                metrics_text += "💡 Interpretation:\n"
                if silhouette > 0.5:
                    metrics_text += "✓ Excellent clustering quality\n"
                elif silhouette > 0.3:
                    metrics_text += "✓ Good clustering quality\n"
                elif silhouette > 0.1:
                    metrics_text += "⚠ Moderate clustering quality\n"
                else:
                    metrics_text += "⚠ Poor clustering - consider re-training\n"
            else:
                metrics_text += "Metrics not available\n"
            
            ax_metrics.text(0.1, 0.5, metrics_text, 
                          fontsize=10, family='monospace',
                          verticalalignment='center')
        
        fig.tight_layout()
        
        # Convert to numpy array
        canvas.draw()
        buf = canvas.buffer_rgba()
        img_array = np.asarray(buf)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        return img_array
    
    def track_embedding_drift(self, 
                             identity_name: str,
                             new_embedding: np.ndarray,
                             timestamp: Optional[datetime] = None):
        """
        Track embedding changes over time for drift detection.
        
        Args:
            identity_name: Name of the identity
            new_embedding: New embedding vector
            timestamp: Optional timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.embedding_history[identity_name].append(new_embedding)
        self.timestamp_history[identity_name].append(timestamp)
    
    def analyze_drift(self, identity_name: str, 
                     window_size: int = 10) -> Dict:
        """
        Analyze embedding drift for a specific identity.
        
        Args:
            identity_name: Name of the identity
            window_size: Number of recent embeddings to analyze
        
        Returns:
            Dictionary with drift analysis
        """
        if identity_name not in self.embedding_history:
            return {'error': 'No history for this identity'}
        
        history = self.embedding_history[identity_name]
        timestamps = self.timestamp_history[identity_name]
        
        if len(history) < 2:
            return {'error': 'Insufficient history (need at least 2 points)'}
        
        # Get recent embeddings
        recent_embeddings = np.array(history[-window_size:])
        recent_timestamps = timestamps[-window_size:]
        
        # Compute pairwise distances
        distances = cdist(recent_embeddings, recent_embeddings, metric='cosine')
        
        # Statistics
        avg_distance = distances[np.triu_indices_from(distances, k=1)].mean()
        max_distance = distances.max()
        
        # Drift score: compare recent vs original
        original_embedding = history[0]
        drift_distances = cdist([original_embedding], recent_embeddings, metric='cosine')[0]
        avg_drift = drift_distances.mean()
        max_drift = drift_distances.max()
        
        # Classification
        if max_drift < 0.1:
            drift_level = "Stable"
            recommendation = "No action needed"
        elif max_drift < 0.2:
            drift_level = "Minor Drift"
            recommendation = "Monitor closely"
        elif max_drift < 0.3:
            drift_level = "Moderate Drift"
            recommendation = "Consider re-registration soon"
        else:
            drift_level = "Significant Drift"
            recommendation = "Re-registration recommended"
        
        return {
            'identity': identity_name,
            'num_samples': len(history),
            'window_size': len(recent_embeddings),
            'avg_distance': float(avg_distance),
            'max_distance': float(max_distance),
            'avg_drift_from_original': float(avg_drift),
            'max_drift_from_original': float(max_drift),
            'drift_level': drift_level,
            'recommendation': recommendation,
            'timestamps': [str(t) for t in recent_timestamps]
        }
    
    def plot_drift_timeline(self, 
                           identity_name: str,
                           figsize: Tuple[int, int] = (10, 6)) -> np.ndarray:
        """
        Plot embedding drift over time.
        
        Args:
            identity_name: Name of the identity
            figsize: Figure size
        
        Returns:
            Plot as numpy array
        """
        if identity_name not in self.embedding_history:
            # Return placeholder
            fig = Figure(figsize=figsize)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"No drift history for {identity_name}",
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            canvas.draw()
            buf = canvas.buffer_rgba()
            return np.asarray(buf)[:, :, :3]
        
        history = np.array(self.embedding_history[identity_name])
        timestamps = self.timestamp_history[identity_name]
        
        # Compute distances from original
        original = history[0]
        distances = cdist([original], history, metric='cosine')[0]
        
        fig = Figure(figsize=figsize)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot drift
        time_indices = list(range(len(timestamps)))
        ax.plot(time_indices, distances, marker='o', linewidth=2, markersize=8)
        
        # Color zones
        ax.axhspan(0, 0.1, alpha=0.2, color='green', label='Stable')
        ax.axhspan(0.1, 0.2, alpha=0.2, color='yellow', label='Minor Drift')
        ax.axhspan(0.2, 0.3, alpha=0.2, color='orange', label='Moderate Drift')
        ax.axhspan(0.3, 1.0, alpha=0.2, color='red', label='Significant Drift')
        
        ax.set_xlabel('Sample Index (Time →)', fontsize=12)
        ax.set_ylabel('Cosine Distance from Original', fontsize=12)
        ax.set_title(f'Embedding Drift Timeline: {identity_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        fig.tight_layout()
        canvas.draw()
        buf = canvas.buffer_rgba()
        img_array = np.asarray(buf)[:, :, :3]
        
        return img_array
    
    def get_metrics_summary(self) -> str:
        """Get formatted string summary of cluster metrics."""
        if not self.cluster_metrics:
            return "No metrics available"
        
        lines = []
        lines.append("=" * 50)
        lines.append("EMBEDDING SPACE QUALITY METRICS")
        lines.append("=" * 50)
        lines.append(f"Silhouette Score: {self.cluster_metrics['silhouette_score']:.3f}")
        lines.append(f"Davies-Bouldin Index: {self.cluster_metrics['davies_bouldin_index']:.3f}")
        lines.append(f"Separation Ratio: {self.cluster_metrics['separation_ratio']:.2f}")
        lines.append(f"Number of Identities: {self.cluster_metrics['num_clusters']}")
        lines.append(f"Total Embeddings: {self.cluster_metrics['num_samples']}")
        lines.append("=" * 50)
        
        return '\n'.join(lines)


def create_embedding_dashboard(employee_db: Dict, 
                               model,
                               device='cpu',
                               transform=None) -> EmbeddingDashboard:
    """
    Convenience function to create dashboard from employee database.
    
    Args:
        employee_db: Dictionary mapping names to embeddings
        model: Face embedding model (not used if embeddings are pre-computed)
        device: torch device
        transform: Image transform (not used if embeddings are pre-computed)
    
    Returns:
        Fitted EmbeddingDashboard instance
    """
    # Extract embeddings and labels
    all_embeddings = []
    all_labels = []
    all_names = []
    
    label_map = {name: idx for idx, name in enumerate(sorted(employee_db.keys()))}
    
    for name, data in employee_db.items():
        label = label_map[name]
        
        # Handle both single embedding and multi-embedding formats
        if isinstance(data, torch.Tensor):
            embeddings = [data]
        elif isinstance(data, list):
            embeddings = data
        else:
            continue
        
        for emb in embeddings:
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            all_embeddings.append(emb)
            all_labels.append(label)
    
    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)
    all_names = list(label_map.keys())
    
    # Create and fit dashboard
    dashboard = EmbeddingDashboard(perplexity=min(30, len(all_embeddings) - 1))
    dashboard.fit(all_embeddings, all_labels, all_names)
    
    return dashboard


if __name__ == "__main__":
    print("✓ t-SNE Visualization Dashboard module loaded")
    print("\nFeatures:")
    print("  • Real-time t-SNE projection")
    print("  • Query highlighting with neighbors")
    print("  • Cluster quality metrics")
    print("  • Embedding drift tracking")
    print("  • Interactive exploration")
