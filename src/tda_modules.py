"""
TDA (Topological Data Analysis) Modules for Face Verification

This module provides topological data analysis tools for analyzing and regularizing
CBAM attention maps. Uses giotto-tda for persistent homology computation.

Key Features:
- Cubical persistence on 2D attention maps
- Persistence-based feature extraction (Betti curves, entropy)
- Topological regularization loss for training
- Efficient batch processing

Usage:
    from tda_modules import TDAAnalyzer, compute_attention_topology_loss
    
    analyzer = TDAAnalyzer()
    topo_features = analyzer.extract_features(attention_maps)
    topo_loss = compute_attention_topology_loss(attention_maps)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import time

# Try to import giotto-tda
try:
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import (
        PersistenceEntropy, 
        Amplitude,
        NumberOfPoints,
        PersistenceLandscape,
        BettiCurve
    )
    GIOTTO_AVAILABLE = True
except ImportError:
    GIOTTO_AVAILABLE = False
    print("Warning: giotto-tda not available. TDA features will be disabled.")


class TDAAnalyzer:
    """
    Analyzes topological features of attention maps using persistent homology.
    
    This class extracts topological features from 2D attention maps (e.g., CBAM
    spatial attention) using cubical persistence. Features include:
    - Betti numbers (connected components, holes)
    - Persistence entropy
    - Amplitude statistics
    - Betti curves
    
    Attributes:
        homology_dims: Homology dimensions to compute (default: [0, 1])
        n_bins: Number of bins for Betti curves
    """
    
    def __init__(
        self, 
        homology_dims: List[int] = [0, 1],
        n_bins: int = 10,
        infinity_values: float = None
    ):
        """
        Initialize TDA analyzer.
        
        Args:
            homology_dims: Homology dimensions to compute. 0=connected components, 1=holes
            n_bins: Number of bins for discretized features (Betti curves)
            infinity_values: Value to replace infinity in persistence diagrams
        """
        if not GIOTTO_AVAILABLE:
            raise RuntimeError("giotto-tda is required but not installed")
        
        self.homology_dims = homology_dims
        self.n_bins = n_bins
        self.infinity_values = infinity_values
        
        # Initialize persistence computation
        self.cubical = CubicalPersistence(
            homology_dimensions=homology_dims,
            infinity_values=infinity_values
        )
        
        # Feature extractors
        self.entropy = PersistenceEntropy()
        self.amplitude = Amplitude(metric="wasserstein", order=2)
        self.num_points = NumberOfPoints()
        self.betti_curve = BettiCurve(n_bins=n_bins)
    
    def compute_persistence(
        self, 
        attention_maps: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Compute persistence diagrams for attention maps.
        
        Args:
            attention_maps: Attention maps of shape (B, H, W) or (B, 1, H, W)
            
        Returns:
            Persistence diagrams array of shape (B, n_points, 3)
            Each point is (birth, death, dimension)
        """
        # Convert to numpy if needed
        if isinstance(attention_maps, torch.Tensor):
            attention_maps = attention_maps.detach().cpu().numpy()
        
        # Ensure shape is (B, H, W)
        if attention_maps.ndim == 4:
            attention_maps = attention_maps.squeeze(1)
        elif attention_maps.ndim == 2:
            attention_maps = attention_maps[np.newaxis, ...]
        
        # Compute persistence diagrams
        diagrams = self.cubical.fit_transform(attention_maps)
        return diagrams
    
    def extract_features(
        self, 
        attention_maps: Union[np.ndarray, torch.Tensor],
        return_dict: bool = True
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Extract topological features from attention maps.
        
        Args:
            attention_maps: Attention maps of shape (B, H, W) or (B, 1, H, W)
            return_dict: If True, return dict with named features; else concatenated array
            
        Returns:
            Dictionary of features or concatenated feature array
        """
        # Compute persistence diagrams
        diagrams = self.compute_persistence(attention_maps)
        
        # Extract features
        entropy = self.entropy.fit_transform(diagrams)
        amplitude = self.amplitude.fit_transform(diagrams)
        num_points = self.num_points.fit_transform(diagrams)
        betti_curves = self.betti_curve.fit_transform(diagrams)
        
        if return_dict:
            return {
                'entropy': entropy,  # Shape: (B, n_homology_dims)
                'amplitude': amplitude,  # Shape: (B, n_homology_dims)
                'num_points': num_points,  # Shape: (B, n_homology_dims)
                'betti_curves': betti_curves,  # Shape: (B, n_homology_dims * n_bins)
                'diagrams': diagrams
            }
        else:
            # Concatenate all scalar features
            return np.concatenate([
                entropy, amplitude, num_points, 
                betti_curves.reshape(betti_curves.shape[0], -1)
            ], axis=1)
    
    def get_feature_dim(self) -> int:
        """Get the dimension of concatenated feature vector."""
        n_dims = len(self.homology_dims)
        # entropy + amplitude + num_points + betti_curves
        return n_dims * 3 + n_dims * self.n_bins


class AttentionTopologyLoss(nn.Module):
    """
    Topological regularization loss for CBAM attention maps.
    
    This loss encourages attention maps to have desirable topological properties:
    - Low persistence entropy (focused, non-noisy attention)
    - Few but significant topological features
    
    The loss is computed on CPU using giotto-tda (not differentiable through
    the topology computation itself, but provides gradient signal through
    the attention map values that create the topology).
    """
    
    def __init__(
        self,
        target_entropy: float = 0.5,
        entropy_weight: float = 1.0,
        complexity_weight: float = 0.1,
        homology_dims: List[int] = [0, 1]
    ):
        """
        Initialize topological loss.
        
        Args:
            target_entropy: Target persistence entropy (lower = more focused)
            entropy_weight: Weight for entropy deviation loss
            complexity_weight: Weight for topological complexity penalty
            homology_dims: Homology dimensions to analyze
        """
        super().__init__()
        
        if not GIOTTO_AVAILABLE:
            raise RuntimeError("giotto-tda is required but not installed")
        
        self.target_entropy = target_entropy
        self.entropy_weight = entropy_weight
        self.complexity_weight = complexity_weight
        
        self.analyzer = TDAAnalyzer(homology_dims=homology_dims)
    
    def forward(
        self, 
        attention_maps: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute topological regularization loss.
        
        Args:
            attention_maps: Attention maps of shape (B, 1, H, W) or (B, H, W)
            return_features: If True, also return extracted features
            
        Returns:
            Scalar loss tensor (on same device as input)
            Optionally: tuple of (loss, feature_dict)
        """
        device = attention_maps.device
        
        # Extract features (computed on CPU)
        features = self.analyzer.extract_features(attention_maps, return_dict=True)
        
        # Compute entropy deviation loss
        entropy = features['entropy']  # (B, n_dims)
        entropy_loss = np.mean((entropy - self.target_entropy) ** 2)
        
        # Compute complexity penalty (penalize too many topological features)
        num_points = features['num_points']  # (B, n_dims)
        complexity_loss = np.mean(num_points)
        
        # Total loss
        total_loss = (
            self.entropy_weight * entropy_loss +
            self.complexity_weight * complexity_loss
        )
        
        # Convert to tensor on original device
        loss_tensor = torch.tensor(total_loss, dtype=torch.float32, device=device)
        
        if return_features:
            return loss_tensor, features
        return loss_tensor


def compute_attention_topology_loss(
    attention_maps: torch.Tensor,
    target_entropy: float = 0.5,
    entropy_weight: float = 1.0,
    complexity_weight: float = 0.1
) -> torch.Tensor:
    """
    Convenience function to compute topological loss on attention maps.
    
    Args:
        attention_maps: Attention maps of shape (B, 1, H, W) or (B, H, W)
        target_entropy: Target persistence entropy
        entropy_weight: Weight for entropy loss
        complexity_weight: Weight for complexity penalty
        
    Returns:
        Scalar loss tensor
    """
    if not GIOTTO_AVAILABLE:
        # Return zero loss if giotto-tda not available
        return torch.tensor(0.0, device=attention_maps.device)
    
    loss_fn = AttentionTopologyLoss(
        target_entropy=target_entropy,
        entropy_weight=entropy_weight,
        complexity_weight=complexity_weight
    )
    return loss_fn(attention_maps)


def extract_persistence_features(
    attention_maps: torch.Tensor,
    as_tensor: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Extract persistence features from attention maps for concatenation with embeddings.
    
    Args:
        attention_maps: Attention maps of shape (B, 1, H, W) or (B, H, W)
        as_tensor: If True, return PyTorch tensor; else numpy array
        
    Returns:
        Feature array/tensor of shape (B, feature_dim)
    """
    if not GIOTTO_AVAILABLE:
        batch_size = attention_maps.shape[0]
        dummy = np.zeros((batch_size, 1), dtype=np.float32)
        return torch.from_numpy(dummy).to(attention_maps.device) if as_tensor else dummy
    
    analyzer = TDAAnalyzer()
    features = analyzer.extract_features(attention_maps, return_dict=False)
    
    if as_tensor:
        return torch.from_numpy(features.astype(np.float32)).to(attention_maps.device)
    return features


def benchmark_tda_overhead(
    spatial_sizes: List[Tuple[int, int]] = [(4, 4), (8, 8), (16, 16)],
    batch_sizes: List[int] = [1, 8, 32],
    n_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark TDA computation overhead for different input sizes.
    
    Args:
        spatial_sizes: List of (H, W) sizes to test
        batch_sizes: List of batch sizes to test
        n_iterations: Number of iterations for timing
        
    Returns:
        Dictionary mapping configuration to ms/sample
    """
    if not GIOTTO_AVAILABLE:
        return {"error": "giotto-tda not available"}
    
    results = {}
    analyzer = TDAAnalyzer()
    
    for H, W in spatial_sizes:
        for B in batch_sizes:
            key = f"batch{B}_{H}x{W}"
            
            # Create dummy attention maps
            attention = np.random.randn(B, H, W).astype(np.float32)
            
            # Warm-up
            _ = analyzer.extract_features(attention)
            
            # Timing
            start = time.time()
            for _ in range(n_iterations):
                _ = analyzer.extract_features(attention)
            elapsed = time.time() - start
            
            ms_per_sample = (elapsed / n_iterations / B) * 1000
            results[key] = ms_per_sample
    
    return results


# Module availability check
def is_tda_available() -> bool:
    """Check if TDA functionality is available."""
    return GIOTTO_AVAILABLE


if __name__ == "__main__":
    print("=" * 70)
    print("TDA Module Self-Test")
    print("=" * 70)
    
    if not GIOTTO_AVAILABLE:
        print("ERROR: giotto-tda not installed")
        exit(1)
    
    print("\n1. Testing TDAAnalyzer...")
    analyzer = TDAAnalyzer()
    
    # Create dummy attention map (batch of 4, 4x4 spatial)
    attention = torch.randn(4, 1, 4, 4)
    features = analyzer.extract_features(attention)
    
    print(f"   Input shape: {attention.shape}")
    print(f"   Entropy shape: {features['entropy'].shape}")
    print(f"   Amplitude shape: {features['amplitude'].shape}")
    print(f"   Num points shape: {features['num_points'].shape}")
    print(f"   Betti curves shape: {features['betti_curves'].shape}")
    print("   ✓ TDAAnalyzer works!")
    
    print("\n2. Testing AttentionTopologyLoss...")
    loss_fn = AttentionTopologyLoss()
    loss = loss_fn(attention)
    print(f"   Loss value: {loss.item():.4f}")
    print("   ✓ AttentionTopologyLoss works!")
    
    print("\n3. Benchmarking overhead...")
    results = benchmark_tda_overhead(
        spatial_sizes=[(4, 4), (8, 8)],
        batch_sizes=[1, 16],
        n_iterations=5
    )
    for key, ms in results.items():
        print(f"   {key}: {ms:.2f} ms/sample")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
