"""
Enhanced TDA Training Script

Based on experimental results, this script trains the dual-stream model
with an enhanced TDA configuration that combines the best aspects:

1. Keep sublevel filtration (works best for faces)
2. Keep H0 only for now (H0+H1 needs more epochs to show benefit)
3. Reduce bandwidth slightly (50 -> 40) for more detail
4. Precompute all TDA features for fast training

Usage:
    conda run -n face_recog python scripts/train_enhanced_tda.py --precompute
    conda run -n face_recog python scripts/train_enhanced_tda.py --train --epochs 20
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms
from PIL import Image

# Import TDA extractor
from scripts.tda_experiments import (
    EnhancedTDAExtractor, TDAConfig,
    weight_death_squared, filtration_sublevel,
)


# ============================================================================
# OPTIMIZED CONFIGURATION
# ============================================================================

def get_optimized_config() -> TDAConfig:
    """
    Optimized TDA configuration based on experiments.
    
    Key insights:
    - Sublevel filtration works best for face images
    - H0 (connected components) captures essential topology
    - 20x20 resolution is good balance of speed vs detail
    - Bandwidth 50 (default) performs well
    """
    return TDAConfig(
        name="optimized_v1",
        homology_dims=[0],  # H0 only - H1 needs more training to show benefit
        resolution=(20, 20),
        bandwidth=50.0,  # Keep default, lower causes overfitting
        weight_fn=weight_death_squared,
        filtration_fn=filtration_sublevel,
    )


# ============================================================================
# DATASET WITH PRECOMPUTED TDA
# ============================================================================

class TripletDatasetWithPrecomputedTDA(Dataset):
    """Dataset that uses precomputed TDA features for fast training."""
    
    def __init__(
        self, 
        data_dir: Path, 
        tda_cache_path: Path,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load precomputed TDA features
        print(f"Loading TDA cache from {tda_cache_path}...")
        data = np.load(tda_cache_path)
        self.tda_features = {k: data[k] for k in data.files}
        print(f"Loaded {len(self.tda_features)} TDA features")
        
        # Build identity mapping
        self.identity_to_images = {}
        self.all_images = []
        
        for identity_dir in sorted(self.data_dir.iterdir()):
            if identity_dir.is_dir():
                identity = identity_dir.name
                images = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    images.extend(identity_dir.glob(ext))
                
                # Filter to only images with precomputed TDA features
                valid_images = []
                for img_path in images:
                    key = str(img_path)
                    if key in self.tda_features:
                        valid_images.append(img_path)
                
                if len(valid_images) >= 2:
                    self.identity_to_images[identity] = valid_images
                    for img_path in valid_images:
                        self.all_images.append((img_path, identity))
        
        self.identities = list(self.identity_to_images.keys())
        print(f"Loaded {len(self.all_images)} images from {len(self.identities)} identities")
    
    def __len__(self):
        return len(self.all_images)
    
    def _get_tda_features(self, path: Path) -> torch.Tensor:
        """Get precomputed TDA features."""
        key = str(path)
        return torch.tensor(self.tda_features[key], dtype=torch.float32)
    
    def _load_image(self, path: Path) -> torch.Tensor:
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
        
        return (
            (self._load_image(anchor_path), self._get_tda_features(anchor_path)),
            (self._load_image(positive_path), self._get_tda_features(positive_path)),
            (self._load_image(negative_path), self._get_tda_features(negative_path)),
        )


# ============================================================================
# MODEL (same as original DualStreamFaceNet)
# ============================================================================

class DualStreamFaceNet(nn.Module):
    """Dual-stream CNN+TDA model for face verification."""
    
    def __init__(
        self, 
        embedding_dim: int = 256,
        tda_dim: int = 400,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # CNN Backbone
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
        
        # TDA Branch
        tda_hidden = 128
        self.tda_branch = nn.Sequential(
            nn.Linear(tda_dim, tda_hidden),
            nn.BatchNorm1d(tda_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        fusion_dim = cnn_out + tda_hidden
        
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
        x = self.gap(x).flatten(1)
        
        # TDA stream
        tda = self.tda_branch(tda_features)
        
        # Fusion
        fused = torch.cat([x, tda], dim=1)
        fused = self.fusion(fused)
        
        # Embedding
        emb = self.embedding(fused)
        return F.normalize(emb, p=2, dim=1)


# ============================================================================
# PRECOMPUTATION
# ============================================================================

def precompute_tda_features(config: TDAConfig, output_path: Path):
    """Precompute TDA features for all images in the dataset."""
    
    train_dir = PROJECT_ROOT / "dataset" / "classification_data" / "train_data"
    val_dir = PROJECT_ROOT / "dataset" / "classification_data" / "val_data"
    test_dir = PROJECT_ROOT / "dataset" / "classification_data" / "test_data"
    
    print("=" * 70)
    print("TDA FEATURE PRECOMPUTATION")
    print("=" * 70)
    print(f"Config: {config}")
    print(f"Output: {output_path}")
    
    # Create extractor
    print("\nInitializing TDA extractor...")
    extractor = EnhancedTDAExtractor(config)
    
    # Fit on sample images
    print("Fitting extractor on sample images...")
    sample_images = []
    for i, img_path in enumerate(train_dir.rglob("*.*")):
        if i >= 500:
            break
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            img = Image.open(img_path).convert('L').resize((64, 64))
            sample_images.append(np.array(img))
    extractor.fit(np.array(sample_images))
    
    tda_dim = extractor.feature_dim
    print(f"TDA feature dimension: {tda_dim}")
    
    # Collect all image paths
    all_paths = []
    for data_dir in [train_dir, val_dir, test_dir]:
        if data_dir.exists():
            for img_path in data_dir.rglob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    all_paths.append(img_path)
    
    print(f"\nTotal images to process: {len(all_paths)}")
    
    # Estimate time
    test_batch = min(100, len(all_paths))
    test_images = []
    for path in all_paths[:test_batch]:
        img = Image.open(path).convert('L').resize((64, 64))
        test_images.append(np.array(img))
    
    start = time.time()
    _ = extractor.transform(np.array(test_images))
    elapsed = time.time() - start
    
    time_per_img = elapsed / test_batch
    total_time = time_per_img * len(all_paths)
    print(f"Estimated time: {total_time / 60:.1f} minutes ({time_per_img*1000:.2f}ms/img)")
    
    # Process all images
    print("\nProcessing images...")
    features = {}
    batch_size = 256
    
    for i in tqdm(range(0, len(all_paths), batch_size), desc="Batches"):
        batch_paths = all_paths[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert('L').resize((64, 64))
                batch_images.append(np.array(img))
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        if batch_images:
            batch_features = extractor.transform(np.array(batch_images))
            for j, path in enumerate(valid_paths):
                features[str(path)] = batch_features[j].astype(np.float32)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **features)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved {len(features)} TDA features to {output_path}")
    print(f"  File size: {file_size:.1f} MB")
    
    return features


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        pos_dist = F.pairwise_distance(anchor_emb, pos_emb)
        neg_dist = F.pairwise_distance(anchor_emb, neg_emb)
        correct += (pos_dist < neg_dist).sum().item()
        total += anchor_img.size(0)
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
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


def train_model(
    tda_cache_path: Path,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """Train the dual-stream model with precomputed TDA features."""
    
    print("=" * 70)
    print("DUAL-STREAM MODEL TRAINING")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Paths
    train_dir = PROJECT_ROOT / "dataset" / "classification_data" / "train_data"
    val_dir = PROJECT_ROOT / "dataset" / "classification_data" / "val_data"
    output_dir = PROJECT_ROOT / "outputs"
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = TripletDatasetWithPrecomputedTDA(train_dir, tda_cache_path, train_transform)
    val_dataset = TripletDatasetWithPrecomputedTDA(val_dir, tda_cache_path, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Determine TDA dimension from cache
    sample_key = next(iter(train_dataset.tda_features.keys()))
    tda_dim = train_dataset.tda_features[sample_key].shape[0]
    print(f"TDA dimension: {tda_dim}")
    
    # Model
    model = DualStreamFaceNet(
        embedding_dim=256,
        tda_dim=tda_dim,
        dropout=0.3,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    
    total_steps = len(train_loader) * epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
    )
    
    # Training loop
    history = []
    best_val_acc = 0
    best_model_state = None
    
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start
        gap = train_acc - val_acc
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'time': epoch_time,
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            marker = " ★ BEST"
        else:
            marker = ""
        
        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
              f"Val Acc={val_acc*100:.2f}%, Gap={gap*100:+.2f}%, "
              f"Time={epoch_time:.1f}s{marker}")
    
    # Save best model
    model_path = output_dir / "best_enhanced_tda_model.pth"
    torch.save({
        'model_state_dict': best_model_state,
        'best_val_acc': best_val_acc,
        'config': str(get_optimized_config()),
        'tda_dim': tda_dim,
    }, model_path)
    print(f"\n✓ Best model saved to {model_path}")
    print(f"  Best validation accuracy: {best_val_acc*100:.2f}%")
    
    # Save history
    history_path = output_dir / "enhanced_tda_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {history_path}")
    
    return history, best_val_acc


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced TDA Training")
    parser.add_argument('--precompute', action='store_true', help="Precompute TDA features")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    config = get_optimized_config()
    cache_path = PROJECT_ROOT / "outputs" / "tda_experiments" / "enhanced_tda_cache.npz"
    
    if args.precompute:
        precompute_tda_features(config, cache_path)
    
    if args.train:
        if not cache_path.exists():
            print(f"TDA cache not found at {cache_path}")
            print("Run with --precompute first to generate TDA features")
            sys.exit(1)
        
        train_model(
            tda_cache_path=cache_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    
    if not args.precompute and not args.train:
        print("Usage:")
        print("  --precompute  Precompute TDA features")
        print("  --train       Train the model")
        print("  --epochs N    Number of training epochs (default: 20)")
        print("")
        print("Example:")
        print("  python scripts/train_enhanced_tda.py --precompute")
        print("  python scripts/train_enhanced_tda.py --train --epochs 20")


if __name__ == "__main__":
    main()
