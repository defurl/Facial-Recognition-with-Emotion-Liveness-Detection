"""
TDA Feature Extraction using GUDHI

This module implements Topological Data Analysis (TDA) feature extraction
using Cubical Persistence and Persistence Images for face verification.

Workflow:
1. Convert grayscale image to cubical complex
2. Compute persistent homology (H0 = connected components)
3. Convert persistence diagram to persistence image (stable vectorization)
4. Output: fixed-size feature vector for each image

Based on lecturer's MNIST TDA implementation, adapted for 64x64 face images.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union
import pickle
from tqdm import tqdm
import warnings

# GUDHI imports
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations import PersistenceImage, DiagramSelector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Image processing
from PIL import Image


def _persistence_weight(x):
    """
    Weight function for persistence image.
    Uses death value squared: points with higher persistence get more weight.
    This is a module-level function to enable pickling.
    """
    return x[1] ** 2


class TDAFeatureExtractor:
    """
    Extract topological features from images using persistent homology.
    
    Uses Cubical Persistence to compute H0 (connected components) features
    and converts them to stable Persistence Image representation.
    """
    
    def __init__(
        self,
        homology_dimensions: int = 0,  # H0 only for speed
        resolution: Tuple[int, int] = (20, 20),  # Output 400-dim vector
        bandwidth: float = 50.0,
        im_range: Optional[List[float]] = None,  # Auto-detect if None
        n_jobs: int = -2,  # Use all but one CPU
    ):
        """
        Initialize TDA feature extractor.
        
        Args:
            homology_dimensions: 0 for H0 (connected components), 
                                 [0,1] for H0+H1 (components + holes)
            resolution: Persistence image resolution (default 20x20 = 400 features)
            bandwidth: Gaussian smoothing bandwidth for persistence image
            im_range: Range [birth_min, birth_max, death_min, death_max]
                      If None, will be auto-detected from data
            n_jobs: Number of parallel jobs (-2 = all but one CPU)
        """
        self.homology_dimensions = homology_dimensions
        self.resolution = resolution
        self.bandwidth = bandwidth
        self.im_range = im_range
        self.n_jobs = n_jobs
        
        self.pipeline = None
        self.is_fitted = False
        self._feature_dim = resolution[0] * resolution[1]
        
    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of TDA features."""
        return self._feature_dim
    
    def _build_pipeline(self, im_range: List[float]) -> Pipeline:
        """Build the TDA pipeline with specified range."""
        return Pipeline([
            # Step 1: Compute cubical persistence
            ("cub_pers", CubicalPersistence(
                homology_dimensions=self.homology_dimensions,
                n_jobs=self.n_jobs
            )),
            
            # Step 2: Keep only finite persistence points
            ("finite_diags", DiagramSelector(use=True, point_type="finite")),
            
            # Step 3: Convert to persistence image
            ("pers_img", PersistenceImage(
                bandwidth=self.bandwidth,
                weight=_persistence_weight,  # Use module-level function for pickling
                im_range=im_range,
                resolution=list(self.resolution)
            )),
            
            # Step 4: Normalize to [0, 1]
            ("scaler", MinMaxScaler()),
        ])
    
    def _prepare_images(self, images: np.ndarray) -> np.ndarray:
        """
        Prepare images for cubical persistence.
        
        Args:
            images: Array of shape (N, H, W) or (N, H, W, C)
                    Values should be in [0, 255] or [0, 1]
        
        Returns:
            Array of shape (N, H*W) flattened grayscale images
        """
        if images.ndim == 4:  # (N, H, W, C) - convert to grayscale
            # Use luminosity method: 0.299*R + 0.587*G + 0.114*B
            images = np.dot(images[..., :3], [0.299, 0.587, 0.114])
        
        # Ensure values are in [0, 255] for persistence
        if images.max() <= 1.0:
            images = images * 255.0
        
        # Flatten each image
        N = images.shape[0]
        return images.reshape(N, -1)
    
    def fit(self, images: np.ndarray) -> 'TDAFeatureExtractor':
        """
        Fit the TDA pipeline on sample images to determine value ranges.
        
        Args:
            images: Sample images array (N, H, W) or (N, H, W, C)
        
        Returns:
            self
        """
        X = self._prepare_images(images)
        
        # Determine im_range from data if not specified
        if self.im_range is None:
            # Compute persistence on a sample to find range
            sample_size = min(100, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            sample = X[sample_indices]
            
            # Get persistence diagrams
            cub_pers = CubicalPersistence(
                homology_dimensions=self.homology_dimensions,
                n_jobs=self.n_jobs
            )
            diagrams = cub_pers.fit_transform(sample)
            
            # Find min/max birth and death values
            all_points = []
            for diag in diagrams:
                if len(diag) > 0:
                    finite_mask = np.isfinite(diag[:, 1])
                    all_points.append(diag[finite_mask])
            
            if all_points:
                all_points = np.vstack(all_points)
                birth_min = max(0, all_points[:, 0].min() - 10)
                birth_max = all_points[:, 0].max() + 10
                death_min = max(0, all_points[:, 1].min() - 10)
                death_max = all_points[:, 1].max() + 10
            else:
                # Default range for [0, 255] images
                birth_min, birth_max = 0, 256
                death_min, death_max = 0, 256
            
            self.im_range = [birth_min, birth_max, death_min, death_max]
            print(f"Auto-detected im_range: {self.im_range}")
        
        # Build and fit the pipeline
        self.pipeline = self._build_pipeline(self.im_range)
        self.pipeline.fit(X)
        self.is_fitted = True
        
        return self
    
    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Extract TDA features from images.
        
        Args:
            images: Images array (N, H, W) or (N, H, W, C)
        
        Returns:
            TDA features array (N, feature_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("TDAFeatureExtractor must be fitted before transform")
        
        X = self._prepare_images(images)
        return self.pipeline.transform(X)
    
    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(images)
        return self.transform(images)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted extractor to disk."""
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump({
                'homology_dimensions': self.homology_dimensions,
                'resolution': self.resolution,
                'bandwidth': self.bandwidth,
                'im_range': self.im_range,
                'n_jobs': self.n_jobs,
                'is_fitted': self.is_fitted,
                'pipeline': self.pipeline,
            }, f)
        print(f"TDA extractor saved to: {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TDAFeatureExtractor':
        """Load a fitted extractor from disk."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        extractor = cls(
            homology_dimensions=data['homology_dimensions'],
            resolution=data['resolution'],
            bandwidth=data['bandwidth'],
            im_range=data['im_range'],
            n_jobs=data['n_jobs'],
        )
        extractor.pipeline = data['pipeline']
        extractor.is_fitted = data['is_fitted']
        
        return extractor


def load_image_as_array(image_path: Path, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Load a single image as numpy array."""
    img = Image.open(image_path).convert('L')  # Grayscale
    img = img.resize(size, Image.Resampling.LANCZOS)
    return np.array(img)


def precompute_tda_features(
    data_dir: Path,
    output_path: Path,
    extractor: Optional[TDAFeatureExtractor] = None,
    image_size: Tuple[int, int] = (64, 64),
    batch_size: int = 500,
    fit_on_sample: bool = True,
    sample_size: int = 1000,
) -> Tuple[np.ndarray, List[str], TDAFeatureExtractor]:
    """
    Precompute TDA features for all images in a directory.
    
    Args:
        data_dir: Directory containing identity subdirectories with images
        output_path: Path to save the precomputed features
        extractor: TDAFeatureExtractor instance (will create if None)
        image_size: Size to resize images to
        batch_size: Number of images to process at once
        fit_on_sample: Whether to fit the extractor on a sample first
        sample_size: Number of images to use for fitting
    
    Returns:
        Tuple of (features array, image paths list, fitted extractor)
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    
    # Collect all image paths
    print(f"Scanning {data_dir}...")
    image_paths = []
    for identity_dir in sorted(data_dir.iterdir()):
        if identity_dir.is_dir():
            for img_file in identity_dir.glob('*.jpg'):
                image_paths.append(img_file)
            for img_file in identity_dir.glob('*.png'):
                image_paths.append(img_file)
    
    print(f"Found {len(image_paths):,} images")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    # Create extractor if not provided
    if extractor is None:
        extractor = TDAFeatureExtractor(
            homology_dimensions=0,  # H0 only
            resolution=(20, 20),    # 400-dim features
            bandwidth=50.0,
        )
    
    # Fit on sample if needed
    if fit_on_sample and not extractor.is_fitted:
        print(f"Fitting TDA extractor on {sample_size} sample images...")
        sample_indices = np.random.choice(
            len(image_paths), 
            min(sample_size, len(image_paths)), 
            replace=False
        )
        sample_images = np.array([
            load_image_as_array(image_paths[i], image_size) 
            for i in tqdm(sample_indices, desc="Loading samples")
        ])
        extractor.fit(sample_images)
    
    # Process all images in batches
    all_features = []
    path_strings = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Computing TDA features"):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load batch images
        batch_images = np.array([
            load_image_as_array(p, image_size) for p in batch_paths
        ])
        
        # Extract features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = extractor.transform(batch_images)
        
        all_features.append(features)
        path_strings.extend([str(p) for p in batch_paths])
    
    # Concatenate all features
    all_features = np.vstack(all_features)
    print(f"TDA features shape: {all_features.shape}")
    
    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=all_features,
        paths=np.array(path_strings),
    )
    print(f"Saved TDA features to: {output_path}")
    
    return all_features, path_strings, extractor


class TDAFeatureCache:
    """
    Cache for fast lookup of precomputed TDA features by image path.
    """
    
    def __init__(self, cache_path: Path):
        """Load precomputed TDA features from disk."""
        self.cache_path = Path(cache_path)
        
        data = np.load(self.cache_path, allow_pickle=True)
        self.features = data['features']
        self.paths = data['paths']
        
        # Build path -> index mapping
        self.path_to_idx = {path: idx for idx, path in enumerate(self.paths)}
        
        print(f"Loaded TDA cache: {len(self.paths):,} images, {self.features.shape[1]}-dim features")
    
    def get(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Get TDA features for an image path."""
        path_str = str(image_path)
        idx = self.path_to_idx.get(path_str)
        if idx is not None:
            return self.features[idx]
        return None
    
    def get_batch(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        """Get TDA features for multiple image paths."""
        features = []
        for path in image_paths:
            feat = self.get(path)
            if feat is None:
                raise KeyError(f"Image not in cache: {path}")
            features.append(feat)
        return np.array(features)
    
    @property
    def feature_dim(self) -> int:
        return self.features.shape[1]


if __name__ == "__main__":
    # Test the TDA feature extractor
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from config import TRAIN_DIR, VAL_DIR, OUTPUT_DIR
    
    print("=" * 60)
    print("TDA Feature Extraction Test")
    print("=" * 60)
    
    # Test on a few images
    test_dir = TRAIN_DIR
    sample_images = []
    for i, identity_dir in enumerate(sorted(test_dir.iterdir())[:5]):
        if identity_dir.is_dir():
            for img_file in list(identity_dir.glob('*.jpg'))[:2]:
                sample_images.append(load_image_as_array(img_file))
    
    sample_images = np.array(sample_images)
    print(f"Sample images shape: {sample_images.shape}")
    
    # Extract features
    extractor = TDAFeatureExtractor()
    features = extractor.fit_transform(sample_images)
    print(f"TDA features shape: {features.shape}")
    print(f"Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
    
    print("\n✓ TDA feature extraction test passed!")
