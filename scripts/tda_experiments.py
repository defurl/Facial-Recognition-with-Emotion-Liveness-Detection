"""
TDA Enhancement Experiments

This script systematically tests different TDA configurations to find optimal settings:
1. Different filtration functions (sublevel, superlevel, radial, distance transform)
2. Different persistence image resolutions (20x20, 32x32, 64x64)
3. Different bandwidths and weight functions
4. H0 only vs H0+H1 (multi-dimensional homology)

Usage:
    conda run -n face_recog python scripts/tda_experiments.py --smoke-test
    conda run -n face_recog python scripts/tda_experiments.py --full-benchmark
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# GUDHI imports
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations import PersistenceImage, DiagramSelector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Image processing
from PIL import Image
from scipy import ndimage

# ============================================================================
# WEIGHT FUNCTIONS (for Persistence Image)
# ============================================================================

def weight_persistence_squared(x):
    """Standard: weight = (death - birth)^2 = persistence^2"""
    return (x[1] - x[0]) ** 2

def weight_death_squared(x):
    """Current: weight = death^2 (original implementation)"""
    return x[1] ** 2

def weight_linear_persistence(x):
    """Linear: weight = death - birth = persistence"""
    return x[1] - x[0]

def weight_log_persistence(x):
    """Log: weight = log(1 + persistence)"""
    return np.log1p(x[1] - x[0])

def weight_arctan_persistence(x):
    """Arctan: weight = arctan(persistence) - bounded, smooth"""
    return np.arctan(x[1] - x[0])


# ============================================================================
# FILTRATION FUNCTIONS (Image Preprocessing)
# ============================================================================

def filtration_sublevel(image: np.ndarray) -> np.ndarray:
    """
    Standard sublevel filtration: use pixel intensity directly.
    Bright pixels appear late, dark pixels appear early.
    """
    return image.astype(np.float64)


def filtration_superlevel(image: np.ndarray) -> np.ndarray:
    """
    Superlevel filtration: invert the image.
    Bright pixels appear early, dark pixels appear late.
    Better for detecting bright features (eyes, highlights).
    """
    return 255.0 - image.astype(np.float64)


def filtration_radial(image: np.ndarray) -> np.ndarray:
    """
    Radial filtration: distance from center of image.
    Captures face structure which is typically centered.
    Combined with intensity for meaningful topology.
    """
    h, w = image.shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    
    # Normalized radial distance [0, 1]
    radial = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    radial = radial / radial.max()
    
    # Combine radial with intensity (weighted average)
    intensity = image.astype(np.float64) / 255.0
    combined = 0.3 * radial + 0.7 * intensity
    
    return (combined * 255.0).astype(np.float64)


def filtration_distance_transform(image: np.ndarray) -> np.ndarray:
    """
    Distance transform filtration: distance to nearest edge.
    Captures face contours and structure boundaries.
    """
    # Edge detection using Sobel
    from scipy.ndimage import sobel, distance_transform_edt
    
    # Compute gradient magnitude
    dx = sobel(image.astype(np.float64), axis=0)
    dy = sobel(image.astype(np.float64), axis=1)
    gradient = np.sqrt(dx**2 + dy**2)
    
    # Threshold to get edges
    threshold = np.percentile(gradient, 80)
    edges = gradient > threshold
    
    # Distance transform from edges
    dist = distance_transform_edt(~edges)
    
    # Normalize to [0, 255]
    if dist.max() > 0:
        dist = (dist / dist.max()) * 255.0
    
    return dist


def filtration_gaussian_blur(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Gaussian blur filtration: smoothed version reduces noise.
    """
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(image.astype(np.float64), sigma=sigma)
    return blurred


# ============================================================================
# ENHANCED TDA FEATURE EXTRACTOR
# ============================================================================

@dataclass
class TDAConfig:
    """Configuration for TDA feature extraction."""
    name: str
    homology_dims: List[int]  # [0] for H0, [0, 1] for H0+H1
    resolution: Tuple[int, int]
    bandwidth: float
    weight_fn: Callable
    filtration_fn: Callable
    stack_channels: bool = True  # If True, stack H0/H1 as channels; else concat
    
    def __repr__(self):
        h_str = "H0" if self.homology_dims == [0] else "H0+H1"
        return f"{self.name}({h_str}, {self.resolution[0]}x{self.resolution[1]}, bw={self.bandwidth})"


class EnhancedTDAExtractor:
    """
    Enhanced TDA feature extractor with configurable filtrations and homology.
    """
    
    def __init__(self, config: TDAConfig, n_jobs: int = -2):
        self.config = config
        self.n_jobs = n_jobs
        self.pipelines = {}  # One pipeline per homology dimension
        self.im_ranges = {}
        self.is_fitted = False
        
    @property
    def feature_dim(self) -> int:
        """Calculate total feature dimension."""
        res = self.config.resolution[0] * self.config.resolution[1]
        n_dims = len(self.config.homology_dims)
        return res * n_dims
    
    def _prepare_images(self, images: np.ndarray) -> np.ndarray:
        """Apply filtration and prepare images."""
        if images.ndim == 4:  # (N, H, W, C) -> grayscale
            images = np.dot(images[..., :3], [0.299, 0.587, 0.114])
        
        # Ensure [0, 255] range
        if images.max() <= 1.0:
            images = images * 255.0
        
        # Apply filtration function
        filtered = np.array([self.config.filtration_fn(img) for img in images])
        
        # Flatten for cubical persistence
        return filtered.reshape(len(filtered), -1)
    
    def _build_pipeline_for_dim(self, h_dim: int, im_range: List[float]) -> Pipeline:
        """Build pipeline for a specific homology dimension."""
        return Pipeline([
            ("cub_pers", CubicalPersistence(
                homology_dimensions=h_dim,
                n_jobs=self.n_jobs
            )),
            ("finite_diags", DiagramSelector(use=True, point_type="finite")),
            ("pers_img", PersistenceImage(
                bandwidth=self.config.bandwidth,
                weight=self.config.weight_fn,
                im_range=im_range,
                resolution=list(self.config.resolution)
            )),
            ("scaler", MinMaxScaler()),
        ])
    
    def fit(self, images: np.ndarray) -> 'EnhancedTDAExtractor':
        """Fit the extractor on sample images."""
        X = self._prepare_images(images)
        
        # Compute persistence to find ranges for each dimension
        sample_size = min(100, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        sample = X[sample_indices]
        
        for h_dim in self.config.homology_dims:
            # Get persistence diagrams
            cub_pers = CubicalPersistence(homology_dimensions=h_dim, n_jobs=self.n_jobs)
            diagrams = cub_pers.fit_transform(sample)
            
            # Find min/max birth and death values
            all_points = []
            for diag in diagrams:
                if len(diag) > 0:
                    finite_mask = np.isfinite(diag[:, 1])
                    if np.any(finite_mask):
                        all_points.append(diag[finite_mask])
            
            if all_points:
                all_points = np.vstack(all_points)
                birth_min = max(0, all_points[:, 0].min() - 10)
                birth_max = all_points[:, 0].max() + 10
                death_min = max(0, all_points[:, 1].min() - 10)
                death_max = all_points[:, 1].max() + 10
            else:
                birth_min, birth_max = 0, 256
                death_min, death_max = 0, 256
            
            im_range = [birth_min, birth_max, death_min, death_max]
            self.im_ranges[h_dim] = im_range
            
            # Build and fit pipeline
            pipeline = self._build_pipeline_for_dim(h_dim, im_range)
            pipeline.fit(X)
            self.pipelines[h_dim] = pipeline
        
        self.is_fitted = True
        return self
    
    def transform(self, images: np.ndarray) -> np.ndarray:
        """Extract TDA features from images."""
        if not self.is_fitted:
            raise RuntimeError("Extractor must be fitted first")
        
        X = self._prepare_images(images)
        
        features_list = []
        for h_dim in self.config.homology_dims:
            feats = self.pipelines[h_dim].transform(X)
            features_list.append(feats)
        
        if len(features_list) == 1:
            return features_list[0]
        
        if self.config.stack_channels:
            # Stack as multi-channel image (for potential conv processing)
            # Shape: (N, n_dims, res, res) but flatten for now
            return np.hstack(features_list)
        else:
            # Simple concatenation
            return np.hstack(features_list)
    
    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(images)
        return self.transform(images)


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

def get_experiment_configs() -> List[TDAConfig]:
    """Define all experimental configurations to test."""
    configs = []
    
    # === BASELINE (current implementation) ===
    configs.append(TDAConfig(
        name="baseline",
        homology_dims=[0],
        resolution=(20, 20),
        bandwidth=50.0,
        weight_fn=weight_death_squared,
        filtration_fn=filtration_sublevel,
    ))
    
    # === RESOLUTION EXPERIMENTS ===
    configs.append(TDAConfig(
        name="res_32x32",
        homology_dims=[0],
        resolution=(32, 32),
        bandwidth=50.0,
        weight_fn=weight_death_squared,
        filtration_fn=filtration_sublevel,
    ))
    
    configs.append(TDAConfig(
        name="res_64x64",
        homology_dims=[0],
        resolution=(64, 64),
        bandwidth=50.0,
        weight_fn=weight_death_squared,
        filtration_fn=filtration_sublevel,
    ))
    
    # === BANDWIDTH EXPERIMENTS ===
    configs.append(TDAConfig(
        name="bw_25",
        homology_dims=[0],
        resolution=(20, 20),
        bandwidth=25.0,
        weight_fn=weight_death_squared,
        filtration_fn=filtration_sublevel,
    ))
    
    configs.append(TDAConfig(
        name="bw_100",
        homology_dims=[0],
        resolution=(20, 20),
        bandwidth=100.0,
        weight_fn=weight_death_squared,
        filtration_fn=filtration_sublevel,
    ))
    
    # === WEIGHT FUNCTION EXPERIMENTS ===
    configs.append(TDAConfig(
        name="weight_persistence_sq",
        homology_dims=[0],
        resolution=(20, 20),
        bandwidth=50.0,
        weight_fn=weight_persistence_squared,
        filtration_fn=filtration_sublevel,
    ))
    
    configs.append(TDAConfig(
        name="weight_linear",
        homology_dims=[0],
        resolution=(20, 20),
        bandwidth=50.0,
        weight_fn=weight_linear_persistence,
        filtration_fn=filtration_sublevel,
    ))
    
    # === FILTRATION EXPERIMENTS ===
    configs.append(TDAConfig(
        name="filt_superlevel",
        homology_dims=[0],
        resolution=(20, 20),
        bandwidth=50.0,
        weight_fn=weight_death_squared,
        filtration_fn=filtration_superlevel,
    ))
    
    configs.append(TDAConfig(
        name="filt_radial",
        homology_dims=[0],
        resolution=(20, 20),
        bandwidth=50.0,
        weight_fn=weight_death_squared,
        filtration_fn=filtration_radial,
    ))
    
    configs.append(TDAConfig(
        name="filt_distance",
        homology_dims=[0],
        resolution=(20, 20),
        bandwidth=50.0,
        weight_fn=weight_death_squared,
        filtration_fn=filtration_distance_transform,
    ))
    
    # === H0+H1 EXPERIMENTS ===
    configs.append(TDAConfig(
        name="h0h1_20x20",
        homology_dims=[0, 1],
        resolution=(20, 20),
        bandwidth=50.0,
        weight_fn=weight_death_squared,
        filtration_fn=filtration_sublevel,
        stack_channels=True,
    ))
    
    configs.append(TDAConfig(
        name="h0h1_32x32",
        homology_dims=[0, 1],
        resolution=(32, 32),
        bandwidth=50.0,
        weight_fn=weight_death_squared,
        filtration_fn=filtration_sublevel,
        stack_channels=True,
    ))
    
    # === COMBINED BEST (hypothesis) ===
    configs.append(TDAConfig(
        name="combined_best",
        homology_dims=[0, 1],
        resolution=(32, 32),
        bandwidth=40.0,
        weight_fn=weight_persistence_squared,
        filtration_fn=filtration_sublevel,
        stack_channels=True,
    ))
    
    return configs


# ============================================================================
# SMOKE TEST & BENCHMARKING
# ============================================================================

def load_sample_images(data_dir: Path, n_samples: int = 100) -> np.ndarray:
    """Load sample images for testing."""
    images = []
    image_paths = list(data_dir.rglob("*.png")) + list(data_dir.rglob("*.jpg"))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    sample_paths = np.random.choice(image_paths, min(n_samples, len(image_paths)), replace=False)
    
    for path in sample_paths:
        img = Image.open(path).convert('L')
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        images.append(np.array(img))
    
    return np.array(images)


def run_smoke_test(config: TDAConfig, images: np.ndarray, verbose: bool = True) -> Dict:
    """
    Run smoke test for a single configuration.
    
    Returns timing and feature statistics.
    """
    n_samples = len(images)
    
    try:
        extractor = EnhancedTDAExtractor(config)
        
        # Time fitting
        start_fit = time.time()
        extractor.fit(images[:min(50, n_samples)])
        fit_time = time.time() - start_fit
        
        # Time transform
        start_transform = time.time()
        features = extractor.transform(images)
        transform_time = time.time() - start_transform
        
        # Feature statistics
        result = {
            'config': str(config),
            'name': config.name,
            'feature_dim': extractor.feature_dim,
            'n_samples': n_samples,
            'fit_time_sec': round(fit_time, 3),
            'transform_time_sec': round(transform_time, 3),
            'time_per_image_ms': round(transform_time / n_samples * 1000, 2),
            'feature_mean': round(float(features.mean()), 4),
            'feature_std': round(float(features.std()), 4),
            'feature_min': round(float(features.min()), 4),
            'feature_max': round(float(features.max()), 4),
            'feature_sparsity': round(float((features == 0).mean()), 4),
            'status': 'success',
        }
        
        if verbose:
            print(f"  ✓ {config.name}: {extractor.feature_dim}D, "
                  f"{result['time_per_image_ms']:.1f}ms/img, "
                  f"mean={result['feature_mean']:.3f}, std={result['feature_std']:.3f}")
        
    except Exception as e:
        result = {
            'config': str(config),
            'name': config.name,
            'status': 'error',
            'error': str(e),
        }
        if verbose:
            print(f"  ✗ {config.name}: ERROR - {e}")
    
    return result


def run_full_benchmark(configs: List[TDAConfig], data_dir: Path, output_dir: Path):
    """Run full benchmark on all configurations."""
    print("=" * 70)
    print("TDA CONFIGURATION BENCHMARK")
    print("=" * 70)
    
    # Load sample images
    print(f"\nLoading sample images from {data_dir}...")
    images = load_sample_images(data_dir, n_samples=500)
    print(f"Loaded {len(images)} images (64x64 grayscale)")
    
    results = []
    
    print("\nRunning experiments...")
    print("-" * 70)
    
    for config in configs:
        result = run_smoke_test(config, images, verbose=True)
        results.append(result)
    
    print("-" * 70)
    
    # Sort by time per image
    successful = [r for r in results if r['status'] == 'success']
    successful.sort(key=lambda x: x['time_per_image_ms'])
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY (sorted by speed)")
    print("=" * 70)
    print(f"{'Config':<25} {'Dim':>6} {'Time/img':>10} {'Mean':>8} {'Std':>8}")
    print("-" * 70)
    
    for r in successful:
        print(f"{r['name']:<25} {r['feature_dim']:>6} {r['time_per_image_ms']:>8.1f}ms "
              f"{r['feature_mean']:>8.4f} {r['feature_std']:>8.4f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "tda_benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if successful:
        fastest = successful[0]
        
        # Find best feature diversity (highest std that's not noise)
        diverse = [r for r in successful if r['feature_std'] > 0.1]
        if diverse:
            diverse.sort(key=lambda x: -x['feature_std'])
            best_diverse = diverse[0]
        else:
            best_diverse = fastest
        
        print(f"\nFastest: {fastest['name']} ({fastest['time_per_image_ms']:.1f}ms/img)")
        print(f"Most diverse features: {best_diverse['name']} (std={best_diverse['feature_std']:.4f})")
        
        # Estimate full dataset times
        n_train = 380638  # Training set size
        print(f"\nEstimated precomputation time for {n_train:,} training images:")
        for r in successful[:5]:
            est_hours = r['time_per_image_ms'] * n_train / 1000 / 3600
            print(f"  {r['name']:<25}: ~{est_hours:.1f} hours")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TDA Enhancement Experiments")
    parser.add_argument('--smoke-test', action='store_true', 
                        help="Run quick smoke test (100 images)")
    parser.add_argument('--full-benchmark', action='store_true',
                        help="Run full benchmark (500 images)")
    parser.add_argument('--config', type=str, default=None,
                        help="Test specific config by name")
    parser.add_argument('--n-samples', type=int, default=100,
                        help="Number of samples for testing")
    args = parser.parse_args()
    
    # Paths
    data_dir = PROJECT_ROOT / "dataset" / "classification_data" / "train_data"
    output_dir = PROJECT_ROOT / "outputs" / "tda_experiments"
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    configs = get_experiment_configs()
    
    if args.config:
        # Test specific config
        configs = [c for c in configs if c.name == args.config]
        if not configs:
            print(f"ERROR: Config '{args.config}' not found")
            print(f"Available: {[c.name for c in get_experiment_configs()]}")
            sys.exit(1)
    
    if args.smoke_test:
        print("=" * 70)
        print("SMOKE TEST (Quick validation)")
        print("=" * 70)
        
        print(f"\nLoading {args.n_samples} sample images...")
        images = load_sample_images(data_dir, n_samples=args.n_samples)
        print(f"Loaded {len(images)} images\n")
        
        for config in configs:
            run_smoke_test(config, images, verbose=True)
        
        print("\n✓ Smoke test complete!")
        
    elif args.full_benchmark:
        run_full_benchmark(configs, data_dir, output_dir)
        
    else:
        parser.print_help()
        print("\nAvailable configurations:")
        for c in configs:
            print(f"  - {c.name}: {c}")


if __name__ == "__main__":
    main()
