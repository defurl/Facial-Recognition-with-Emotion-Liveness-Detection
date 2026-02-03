#!/usr/bin/env python3
"""
Training script with TDA Consistency Regularization

Key Insight: TDA features have poor discriminative power for faces (gap=0.016).
All human faces have similar topology (2 eyes, nose, mouth = same structure).

NEW APPROACH: Instead of using TDA as input features (which hurts performance),
use TDA as a CONSISTENCY REGULARIZATION loss term:

Loss = TripletLoss(CNN_embeddings) + λ * TDAConsistencyLoss

Where TDAConsistencyLoss encourages:
  - Same-person images to have similar TDA features
  - The CNN to learn pose/lighting invariant features

This way CNN remains the primary discriminator (94.15% baseline)
while TDA provides auxiliary structural consistency.

Usage:
    conda run -n face_recog python scripts/train_tda_regularized.py
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np

try:
    from config import (
        DEVICE, LEARNING_RATE, RANDOM_SEED, OUTPUT_DIR,
        WEIGHT_DECAY, BATCH_SIZE, TRAIN_DIR, VAL_DIR
    )
except ImportError:
    from .config import (
        DEVICE, LEARNING_RATE, RANDOM_SEED, OUTPUT_DIR,
        WEIGHT_DECAY, BATCH_SIZE, TRAIN_DIR, VAL_DIR
    )

from models import FaceEmbeddingCNN, get_loss_functions, count_parameters
from data_loader import load_classification_data, get_transforms


# Training configuration
NUM_EPOCHS = 25
EMBEDDING_DIM = 256
DROPOUT = 0.3

# TDA Regularization hyperparameters
TDA_LAMBDA = 0.1  # Weight for TDA consistency loss (start small)
TDA_WARMUP_EPOCHS = 3  # Don't apply TDA loss for first N epochs
TDA_CACHE_PATH = OUTPUT_DIR / "tda_cache" / "tda_train.npz"

# Output paths
MODEL_NAME = "best_tda_regularized_model.pth"
MODEL_SAVE_PATH = OUTPUT_DIR / MODEL_NAME
HISTORY_SAVE_PATH = OUTPUT_DIR / "tda_regularized_history.json"


class TDAConsistencyLoss(nn.Module):
    """
    TDA Consistency Regularization Loss
    
    Encourages same-person images to have similar TDA features,
    providing structural consistency without using TDA as direct input.
    """
    
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, tda_anchor, tda_positive, tda_negative):
        """
        Triplet-style loss on TDA features
        
        Args:
            tda_anchor: TDA features of anchor images (B, D)
            tda_positive: TDA features of positive images (B, D)
            tda_negative: TDA features of negative images (B, D)
        
        Returns:
            TDA consistency loss value
        """
        # Cosine similarity (TDA features are already normalized)
        pos_sim = F.cosine_similarity(tda_anchor, tda_positive, dim=1)
        neg_sim = F.cosine_similarity(tda_anchor, tda_negative, dim=1)
        
        # We want pos_sim > neg_sim
        # Loss = max(0, margin - (pos_sim - neg_sim))
        loss = F.relu(self.margin - (pos_sim - neg_sim))
        
        return loss.mean()


class TripletDatasetWithTDA(Dataset):
    """
    Triplet dataset that also loads precomputed TDA features.
    TDA features are used for regularization, not as input.
    
    Note: label_map maps labels to INDICES in image_paths, not paths directly.
    """
    
    def __init__(self, image_paths, labels, label_map, transform=None,
                 tda_features=None, path_to_tda_idx=None):
        self.image_paths = [str(p) for p in image_paths]  # Ensure strings
        self.labels = np.array(labels)
        self.label_map = label_map  # Maps label -> list of indices
        self.transform = transform
        self.tda_features = tda_features
        self.path_to_tda_idx = path_to_tda_idx
        
        # Create label index for negative sampling
        self.unique_labels = list(label_map.keys())
        
        from PIL import Image
        self.Image = Image
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        import random
        
        # Anchor
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]
        
        # Positive (same person) - label_map contains indices
        positive_indices = self.label_map[anchor_label]
        positive_idx = random.choice(positive_indices)
        while positive_idx == idx and len(positive_indices) > 1:
            positive_idx = random.choice(positive_indices)
        positive_path = self.image_paths[positive_idx]
        
        # Negative (different person)
        negative_label = random.choice(self.unique_labels)
        while negative_label == anchor_label:
            negative_label = random.choice(self.unique_labels)
        negative_idx = random.choice(self.label_map[negative_label])
        negative_path = self.image_paths[negative_idx]
        
        # Load images
        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)
        negative_img = self._load_image(negative_path)
        
        # Get TDA features if available
        if self.tda_features is not None and self.path_to_tda_idx is not None:
            anchor_tda = self._get_tda(anchor_path)
            positive_tda = self._get_tda(positive_path)
            negative_tda = self._get_tda(negative_path)
            return anchor_img, positive_img, negative_img, anchor_tda, positive_tda, negative_tda
        else:
            return anchor_img, positive_img, negative_img
    
    def _load_image(self, path):
        img = self.Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    
    def _get_tda(self, path):
        """Get TDA features for a given image path"""
        # Normalize path for lookup
        path_str = str(path)
        
        # Try direct lookup first
        if path_str in self.path_to_tda_idx:
            idx = self.path_to_tda_idx[path_str]
            return torch.from_numpy(self.tda_features[idx].copy()).float()
        
        # Try relative path lookup (identity/filename)
        parts = path_str.replace('\\', '/').split('/')
        if len(parts) >= 2:
            key = parts[-2] + '/' + parts[-1]
            if key in self.path_to_tda_idx:
                idx = self.path_to_tda_idx[key]
                return torch.from_numpy(self.tda_features[idx].copy()).float()
        
        # Return zeros if not found
        return torch.zeros(self.tda_features.shape[1], dtype=torch.float32)


def load_tda_cache(cache_path, image_paths):
    """Load TDA cache and create path-to-index mapping"""
    if not cache_path.exists():
        print(f"  WARNING: TDA cache not found at {cache_path}")
        return None, None
    
    print(f"  Loading TDA cache from {cache_path}...")
    tda_data = np.load(cache_path, allow_pickle=True)
    tda_features = tda_data['features']
    tda_paths = tda_data['paths']
    
    print(f"  TDA features shape: {tda_features.shape}")
    print(f"  TDA paths: {len(tda_paths)}")
    
    # Create path-to-index mapping
    path_to_idx = {}
    for i, p in enumerate(tda_paths):
        # Store with normalized path key
        path_str = str(p)
        path_to_idx[path_str] = i
        # Also store the tail (identity/filename)
        parts = path_str.replace('\\', '/').split('/')
        if len(parts) >= 2:
            key = parts[-2] + '/' + parts[-1]
            path_to_idx[key] = i
    
    # Check how many training images have TDA features
    found = 0
    for path in image_paths:
        path_str = str(path)
        parts = path_str.replace('\\', '/').split('/')
        key = parts[-2] + '/' + parts[-1] if len(parts) >= 2 else path_str
        if key in path_to_idx or path_str in path_to_idx:
            found += 1
    
    print(f"  Matched {found}/{len(image_paths)} training images to TDA cache")
    
    return tda_features, path_to_idx


def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility"""
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, triplet_loss, tda_loss, optimizer, device, 
                epoch, tda_lambda, tda_warmup_epochs, use_tda):
    """Train for one epoch with TDA consistency regularization"""
    model.train()
    
    total_triplet_loss = 0.0
    total_tda_loss = 0.0
    total_loss = 0.0
    correct_triplets = 0
    total_samples = 0
    
    # Compute effective TDA lambda (warmup)
    if epoch < tda_warmup_epochs:
        effective_lambda = 0.0
    else:
        # Linear warmup after warmup_epochs
        effective_lambda = tda_lambda * min(1.0, (epoch - tda_warmup_epochs + 1) / 5)
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        if use_tda and len(batch) == 6:
            anchor_img, pos_img, neg_img, tda_a, tda_p, tda_n = batch
            tda_a = tda_a.to(device)
            tda_p = tda_p.to(device)
            tda_n = tda_n.to(device)
            has_tda = True
        else:
            anchor_img, pos_img, neg_img = batch[:3]
            has_tda = False
        
        anchor_img = anchor_img.to(device)
        pos_img = pos_img.to(device)
        neg_img = neg_img.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (CNN only for embeddings)
        anchor_embed = model(anchor_img, mode='metric')
        pos_embed = model(pos_img, mode='metric')
        neg_embed = model(neg_img, mode='metric')
        
        # Triplet loss on CNN embeddings (primary loss)
        loss_triplet = triplet_loss(anchor_embed, pos_embed, neg_embed)
        
        # TDA consistency loss (regularization)
        loss_tda = torch.tensor(0.0, device=device)
        if has_tda and effective_lambda > 0:
            loss_tda = tda_loss(tda_a, tda_p, tda_n)
        
        # Combined loss
        loss = loss_triplet + effective_lambda * loss_tda
        
        loss.backward()
        optimizer.step()
        
        batch_size = anchor_img.size(0)
        total_triplet_loss += loss_triplet.item() * batch_size
        total_tda_loss += loss_tda.item() * batch_size
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Accuracy
        with torch.no_grad():
            pos_dist = F.pairwise_distance(anchor_embed, pos_embed)
            neg_dist = F.pairwise_distance(anchor_embed, neg_embed)
            correct_triplets += (pos_dist < neg_dist).sum().item()
    
    return (total_loss / total_samples, 
            total_triplet_loss / total_samples,
            total_tda_loss / total_samples,
            correct_triplets / total_samples,
            effective_lambda)


def validate(model, dataloader, triplet_loss, device, use_tda):
    """Validate model (only triplet loss - TDA is just regularization)"""
    model.eval()
    total_loss = 0.0
    correct_triplets = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            anchor_img, pos_img, neg_img = batch[:3]
            anchor_img = anchor_img.to(device)
            pos_img = pos_img.to(device)
            neg_img = neg_img.to(device)
            
            anchor_embed = model(anchor_img, mode='metric')
            pos_embed = model(pos_img, mode='metric')
            neg_embed = model(neg_img, mode='metric')
            
            loss = triplet_loss(anchor_embed, pos_embed, neg_embed)
            
            batch_size = anchor_img.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            pos_dist = F.pairwise_distance(anchor_embed, pos_embed)
            neg_dist = F.pairwise_distance(anchor_embed, neg_embed)
            correct_triplets += (pos_dist < neg_dist).sum().item()
    
    return total_loss / total_samples, correct_triplets / total_samples


def main():
    print("=" * 70)
    print("TDA CONSISTENCY REGULARIZATION TRAINING")
    print("=" * 70)
    print("""
Strategy: Use TDA as regularization, NOT as input features.
  - CNN extracts identity embeddings (primary task)
  - TDA provides consistency signal (auxiliary regularization)
  - Loss = TripletLoss(CNN) + λ * TDAConsistencyLoss
""")
    
    set_seed()
    
    # Configuration
    print("Configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  TDA lambda: {TDA_LAMBDA}")
    print(f"  TDA warmup epochs: {TDA_WARMUP_EPOCHS}")
    print(f"  Device: {DEVICE}")
    
    # Load data
    print("\n" + "-" * 70)
    print("Loading data...")
    
    train_paths, train_labels, train_label_map, train_num_classes = load_classification_data(TRAIN_DIR)
    print(f"  Training: {len(train_paths)} images from {train_num_classes} identities")
    
    val_paths, val_labels, val_label_map, val_num_classes = load_classification_data(VAL_DIR)
    print(f"  Validation: {len(val_paths)} images from {val_num_classes} identities")
    
    # Load TDA cache
    tda_features, path_to_tda_idx = load_tda_cache(TDA_CACHE_PATH, train_paths)
    use_tda = tda_features is not None
    
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    if use_tda:
        train_dataset = TripletDatasetWithTDA(
            train_paths, train_labels, train_label_map, train_transform,
            tda_features=tda_features, path_to_tda_idx=path_to_tda_idx
        )
        # Validation doesn't need TDA (only for regularization during training)
        val_dataset = TripletDatasetWithTDA(
            val_paths, val_labels, val_label_map, val_transform
        )
    else:
        from data_loader import TripletDataset
        train_dataset = TripletDataset(train_paths, train_labels, train_label_map, train_transform)
        val_dataset = TripletDataset(val_paths, val_labels, val_label_map, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "-" * 70)
    print("Initializing CNN model...")
    
    model = FaceEmbeddingCNN(
        embedding_dim=EMBEDDING_DIM,
        num_classes=train_num_classes,
        use_cbam=True,
    )
    
    # Try to load pretrained weights
    pretrained_path = OUTPUT_DIR / "best_cnn_baseline_model.pth"
    if pretrained_path.exists():
        print(f"  Loading pretrained CNN weights from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
        print("  ✓ Pretrained weights loaded successfully")
    else:
        print("  No pretrained weights found, training from scratch")
    
    model = model.to(DEVICE)
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    
    # Loss functions
    _, triplet_loss = get_loss_functions()
    tda_loss = TDAConsistencyLoss(margin=0.3)
    
    # Optimizer with smaller LR if using pretrained (fine-tuning)
    lr = LEARNING_RATE * 0.1 if pretrained_path.exists() else LEARNING_RATE
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training
    print("\n" + "-" * 70)
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print(f"  TDA regularization: {'Enabled' if use_tda else 'Disabled (no cache)'}")
    print("-" * 70)
    
    best_val_acc = 0.0
    history = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        train_loss, triplet_loss_val, tda_loss_val, train_acc, eff_lambda = train_epoch(
            model, train_loader, triplet_loss, tda_loss, optimizer, DEVICE,
            epoch, TDA_LAMBDA, TDA_WARMUP_EPOCHS, use_tda
        )
        
        val_loss, val_acc = validate(model, val_loader, triplet_loss, DEVICE, use_tda)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'triplet_loss': triplet_loss_val,
            'tda_loss': tda_loss_val,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr,
            'tda_lambda': eff_lambda,
            'time': epoch_time,
        })
        
        gap = train_acc - val_acc
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        print(f"  Losses - Total: {train_loss:.4f} | Triplet: {triplet_loss_val:.4f} | TDA: {tda_loss_val:.4f} (λ={eff_lambda:.3f})")
        print(f"  Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Gap: {gap*100:+.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ New best model saved (val_acc: {val_acc*100:.2f}%)")
        
        scheduler.step(val_loss)
    
    # Save history
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    
    # Comparison with baseline
    print("\n" + "-" * 70)
    print("Comparison with CNN Baseline (94.15%):")
    print("-" * 70)
    improvement = best_val_acc - 0.9415
    if improvement > 0:
        print(f"  ✓ IMPROVEMENT: +{improvement*100:.2f}%")
    else:
        print(f"  ✗ No improvement: {improvement*100:.2f}%")


if __name__ == "__main__":
    main()
