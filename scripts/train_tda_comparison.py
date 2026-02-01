"""
TDA Configuration Comparison Training

Tests the most promising TDA configurations with actual model training
to compare validation accuracy improvements.

Promising configs from benchmark:
1. h0h1_20x20: H0+H1 features (800D) - captures both components AND holes
2. bw_25: Lower bandwidth (more detail) with higher feature diversity
3. filt_distance: Distance transform filtration (fastest, edge-focused)
4. combined: H0+H1 with 32x32 resolution (2048D)

Usage:
    conda run -n face_recog python scripts/train_tda_comparison.py --config h0h1_20x20 --epochs 10
    conda run -n face_recog python scripts/train_tda_comparison.py --config all --epochs 5
"""

import os
import sys
import time
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image

# Import from experiments
from scripts.tda_experiments import (
    EnhancedTDAExtractor, TDAConfig,
    weight_death_squared, weight_persistence_squared, weight_linear_persistence,
    filtration_sublevel, filtration_distance_transform, filtration_radial,
)

# ============================================================================
# CONFIGURATIONS TO TEST
# ============================================================================

def get_test_configs() -> Dict[str, TDAConfig]:
    """Get configurations to compare."""
    return {
        'baseline': TDAConfig(
            name="baseline",
            homology_dims=[0],
            resolution=(20, 20),
            bandwidth=50.0,
            weight_fn=weight_death_squared,
            filtration_fn=filtration_sublevel,
        ),
        'h0h1_20x20': TDAConfig(
            name="h0h1_20x20",
            homology_dims=[0, 1],
            resolution=(20, 20),
            bandwidth=50.0,
            weight_fn=weight_death_squared,
            filtration_fn=filtration_sublevel,
            stack_channels=True,
        ),
        'bw_25': TDAConfig(
            name="bw_25",
            homology_dims=[0],
            resolution=(20, 20),
            bandwidth=25.0,
            weight_fn=weight_death_squared,
            filtration_fn=filtration_sublevel,
        ),
        'filt_distance': TDAConfig(
            name="filt_distance",
            homology_dims=[0],
            resolution=(20, 20),
            bandwidth=50.0,
            weight_fn=weight_death_squared,
            filtration_fn=filtration_distance_transform,
        ),
        'h0h1_32x32': TDAConfig(
            name="h0h1_32x32",
            homology_dims=[0, 1],
            resolution=(32, 32),
            bandwidth=50.0,
            weight_fn=weight_death_squared,
            filtration_fn=filtration_sublevel,
            stack_channels=True,
        ),
        'weight_pers_sq': TDAConfig(
            name="weight_pers_sq",
            homology_dims=[0],
            resolution=(20, 20),
            bandwidth=50.0,
            weight_fn=weight_persistence_squared,
            filtration_fn=filtration_sublevel,
        ),
    }


# ============================================================================
# DATASET WITH TDA
# ============================================================================

class TripletDatasetWithEnhancedTDA(Dataset):
    """Dataset that loads images and computes TDA features on-the-fly or from cache."""
    
    def __init__(
        self, 
        data_dir: Path, 
        tda_extractor: EnhancedTDAExtractor,
        transform=None,
        cache_path: Optional[Path] = None,
    ):
        self.data_dir = Path(data_dir)
        self.tda_extractor = tda_extractor
        self.transform = transform
        
        # Build identity mapping
        self.identity_to_images = {}
        self.all_images = []
        
        for identity_dir in sorted(self.data_dir.iterdir()):
            if identity_dir.is_dir():
                identity = identity_dir.name
                images = list(identity_dir.glob("*.png")) + list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.jpeg"))
                if len(images) >= 2:  # Need at least 2 for positive pair
                    self.identity_to_images[identity] = images
                    for img_path in images:
                        self.all_images.append((img_path, identity))
        
        self.identities = list(self.identity_to_images.keys())
        print(f"Loaded {len(self.all_images)} images from {len(self.identities)} identities")
        
        # Load or compute TDA cache
        self.tda_cache = {}
        if cache_path and cache_path.exists():
            data = np.load(cache_path)
            for key in data.files:
                self.tda_cache[key] = data[key]
            print(f"Loaded TDA cache: {len(self.tda_cache)} features")
    
    def __len__(self):
        return len(self.all_images)
    
    def _load_image_gray(self, path: Path) -> np.ndarray:
        """Load image as grayscale numpy array."""
        img = Image.open(path).convert('L')
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        return np.array(img)
    
    def _get_tda_features(self, path: Path) -> torch.Tensor:
        """Get TDA features for an image (cached or computed)."""
        key = str(path)
        if key in self.tda_cache:
            return torch.tensor(self.tda_cache[key], dtype=torch.float32)
        
        # Compute on-the-fly
        img_gray = self._load_image_gray(path)
        features = self.tda_extractor.transform(img_gray[np.newaxis, ...])[0]
        self.tda_cache[key] = features
        return torch.tensor(features, dtype=torch.float32)
    
    def _load_image_rgb(self, path: Path) -> torch.Tensor:
        """Load and transform RGB image."""
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    
    def __getitem__(self, idx):
        anchor_path, anchor_identity = self.all_images[idx]
        
        # Positive: same identity, different image
        positive_candidates = [p for p in self.identity_to_images[anchor_identity] if p != anchor_path]
        positive_path = np.random.choice(positive_candidates)
        
        # Negative: different identity
        negative_identity = np.random.choice([i for i in self.identities if i != anchor_identity])
        negative_path = np.random.choice(self.identity_to_images[negative_identity])
        
        # Load images
        anchor_img = self._load_image_rgb(anchor_path)
        positive_img = self._load_image_rgb(positive_path)
        negative_img = self._load_image_rgb(negative_path)
        
        # Get TDA features
        anchor_tda = self._get_tda_features(anchor_path)
        positive_tda = self._get_tda_features(positive_path)
        negative_tda = self._get_tda_features(negative_path)
        
        return (
            (anchor_img, anchor_tda),
            (positive_img, positive_tda),
            (negative_img, negative_tda),
        )


# ============================================================================
# FLEXIBLE DUAL-STREAM MODEL
# ============================================================================

class FlexibleDualStreamNet(nn.Module):
    """
    Dual-stream model that adapts to different TDA feature dimensions.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 256,
        tda_dim: int = 400,
        tda_hidden_dim: int = 128,
        use_cbam: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.tda_dim = tda_dim
        self.use_cbam = use_cbam
        
        # CNN Backbone (same as original)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        cnn_out = 256
        
        # TDA Branch - adapts to feature dimension
        self.tda_branch = nn.Sequential(
            nn.Linear(tda_dim, tda_hidden_dim),
            nn.BatchNorm1d(tda_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        fusion_dim = cnn_out + tda_hidden_dim
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Embedding
        self.embedding = nn.Sequential(
            nn.Linear(fusion_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
    
    def forward(self, x, tda_features):
        # CNN stream
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x).flatten(1)  # (B, 256)
        
        # TDA stream
        tda = self.tda_branch(tda_features)  # (B, tda_hidden)
        
        # Fusion
        fused = torch.cat([x, tda], dim=1)
        fused = self.fusion(fused)
        
        # Embedding
        emb = self.embedding(fused)
        return F.normalize(emb, p=2, dim=1)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        (anchor_img, anchor_tda), (pos_img, pos_tda), (neg_img, neg_tda) = batch
        
        anchor_img = anchor_img.to(device)
        anchor_tda = anchor_tda.to(device)
        pos_img = pos_img.to(device)
        pos_tda = pos_tda.to(device)
        neg_img = neg_img.to(device)
        neg_tda = neg_tda.to(device)
        
        optimizer.zero_grad()
        
        anchor_emb = model(anchor_img, anchor_tda)
        pos_emb = model(pos_img, pos_tda)
        neg_emb = model(neg_img, neg_tda)
        
        loss = criterion(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy: positive closer than negative
        pos_dist = F.pairwise_distance(anchor_emb, pos_emb)
        neg_dist = F.pairwise_distance(anchor_emb, neg_emb)
        correct += (pos_dist < neg_dist).sum().item()
        total += anchor_img.size(0)
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Validating", leave=False):
        (anchor_img, anchor_tda), (pos_img, pos_tda), (neg_img, neg_tda) = batch
        
        anchor_img = anchor_img.to(device)
        anchor_tda = anchor_tda.to(device)
        pos_img = pos_img.to(device)
        pos_tda = pos_tda.to(device)
        neg_img = neg_img.to(device)
        neg_tda = neg_tda.to(device)
        
        anchor_emb = model(anchor_img, anchor_tda)
        pos_emb = model(pos_img, pos_tda)
        neg_emb = model(neg_img, neg_tda)
        
        loss = criterion(anchor_emb, pos_emb, neg_emb)
        total_loss += loss.item()
        
        pos_dist = F.pairwise_distance(anchor_emb, pos_emb)
        neg_dist = F.pairwise_distance(anchor_emb, neg_emb)
        correct += (pos_dist < neg_dist).sum().item()
        total += anchor_img.size(0)
    
    return total_loss / len(loader), correct / total


def run_comparison_training(
    config_name: str,
    config: TDAConfig,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    subset_size: Optional[int] = None,
):
    """Train model with specific TDA config and return metrics."""
    print(f"\n{'='*70}")
    print(f"Training with: {config_name}")
    print(f"Config: {config}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Paths
    train_dir = PROJECT_ROOT / "dataset" / "classification_data" / "train_data"
    val_dir = PROJECT_ROOT / "dataset" / "classification_data" / "val_data"
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Create TDA extractor
    print("\nInitializing TDA extractor...")
    extractor = EnhancedTDAExtractor(config)
    
    # Fit on sample images
    print("Fitting TDA extractor on sample images...")
    sample_images = []
    for i, img_path in enumerate(train_dir.rglob("*.*")):
        if i >= 200:
            break
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            img = Image.open(img_path).convert('L').resize((64, 64))
            sample_images.append(np.array(img))
    if len(sample_images) == 0:
        raise ValueError(f"No images found in {train_dir}")
    extractor.fit(np.array(sample_images))
    
    tda_dim = extractor.feature_dim
    print(f"TDA feature dimension: {tda_dim}")
    
    # Create datasets (use subset for faster comparison)
    print("\nCreating datasets...")
    train_dataset = TripletDatasetWithEnhancedTDA(train_dir, extractor, train_transform)
    val_dataset = TripletDatasetWithEnhancedTDA(val_dir, extractor, val_transform)
    
    if subset_size:
        train_indices = np.random.choice(len(train_dataset), min(subset_size, len(train_dataset)), replace=False)
        val_indices = np.random.choice(len(val_dataset), min(subset_size // 4, len(val_dataset)), replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Create model
    model = FlexibleDualStreamNet(
        embedding_dim=256,
        tda_dim=tda_dim,
        tda_hidden_dim=min(128, tda_dim // 2),  # Scale hidden dim with input
        dropout=0.3,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    history = []
    best_val_acc = 0
    
    print("\nStarting training...")
    for epoch in range(epochs):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'time': epoch_time,
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        gap = train_acc - val_acc
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
              f"Val Acc={val_acc*100:.2f}%, Gap={gap*100:+.2f}%, "
              f"Time={epoch_time:.1f}s")
    
    result = {
        'config_name': config_name,
        'config': str(config),
        'tda_dim': tda_dim,
        'best_val_acc': best_val_acc,
        'final_train_acc': history[-1]['train_acc'],
        'final_val_acc': history[-1]['val_acc'],
        'final_gap': history[-1]['train_acc'] - history[-1]['val_acc'],
        'total_params': total_params,
        'history': history,
    }
    
    print(f"\n✓ {config_name} complete: Best Val Acc = {best_val_acc*100:.2f}%")
    
    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TDA Configuration Comparison Training")
    parser.add_argument('--config', type=str, default='all',
                        help="Config to test: baseline, h0h1_20x20, bw_25, filt_distance, h0h1_32x32, weight_pers_sq, or 'all'")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs per config")
    parser.add_argument('--subset', type=int, default=10000,
                        help="Subset size for quick comparison (0 = full dataset)")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="Batch size")
    args = parser.parse_args()
    
    all_configs = get_test_configs()
    
    if args.config == 'all':
        configs_to_test = all_configs
    elif args.config in all_configs:
        configs_to_test = {args.config: all_configs[args.config]}
    else:
        print(f"Unknown config: {args.config}")
        print(f"Available: {list(all_configs.keys())} or 'all'")
        sys.exit(1)
    
    print("=" * 70)
    print("TDA CONFIGURATION COMPARISON TRAINING")
    print("=" * 70)
    print(f"Configs to test: {list(configs_to_test.keys())}")
    print(f"Epochs per config: {args.epochs}")
    print(f"Subset size: {args.subset if args.subset > 0 else 'Full dataset'}")
    
    results = []
    
    for name, config in configs_to_test.items():
        result = run_comparison_training(
            config_name=name,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            subset_size=args.subset if args.subset > 0 else None,
        )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'TDA Dim':>8} {'Best Val':>10} {'Final Gap':>10}")
    print("-" * 70)
    
    results.sort(key=lambda x: -x['best_val_acc'])
    for r in results:
        print(f"{r['config_name']:<20} {r['tda_dim']:>8} {r['best_val_acc']*100:>9.2f}% {r['final_gap']*100:>+9.2f}%")
    
    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "tda_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "comparison_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Winner
    if results:
        winner = results[0]
        print(f"\n🏆 BEST CONFIG: {winner['config_name']} with {winner['best_val_acc']*100:.2f}% validation accuracy")


if __name__ == "__main__":
    main()
