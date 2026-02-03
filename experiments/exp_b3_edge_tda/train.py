#!/usr/bin/env python3
"""
Experiment B3: Edge-based TDA Regularization Training

Uses TDA features computed on Canny edge maps instead of raw intensity.
Edges capture facial structure better and may provide more identity-relevant topology.

Prerequisites:
    Run precompute_edge_tda.py first to generate edge-based TDA cache.

Configuration:
- TDA Lambda: 0.05 (same as winning A1 experiment)
- No warmup: TDA regularization from epoch 1
- Edge-based TDA features from Canny edge detection
- 30 epochs with early stopping

Usage:
    conda run -n face_recog python experiments/exp_b3_edge_tda/train.py
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np

from config import (
    DEVICE, LEARNING_RATE, RANDOM_SEED,
    WEIGHT_DECAY, BATCH_SIZE, TRAIN_DIR, VAL_DIR
)
from models import FaceEmbeddingCNN, get_loss_functions, count_parameters
from data_loader import load_classification_data, get_transforms

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
EXPERIMENT_NAME = "exp_b3_edge_tda"
EXPERIMENT_DIR = Path(__file__).parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
NUM_EPOCHS = 30
EMBEDDING_DIM = 256
EARLY_STOPPING_PATIENCE = 10

# TDA Regularization - Use same λ as winning A1 experiment
TDA_LAMBDA = 0.05           # Same as A1 for fair comparison
TDA_WARMUP_EPOCHS = 0       # No warmup - TDA from epoch 1

# Edge-based TDA cache (different from original intensity-based TDA)
TDA_CACHE_PATH = OUTPUT_DIR / "tda_edge_train.npz"

# Output paths (experiment-specific)
MODEL_SAVE_PATH = OUTPUT_DIR / "best_model.pth"
HISTORY_SAVE_PATH = OUTPUT_DIR / "training_history.json"
LOG_FILE = OUTPUT_DIR / "train.log"

# Use pretrained CNN baseline
PRETRAINED_PATH = PROJECT_ROOT / "outputs" / "best_cnn_baseline_model.pth"

# ============================================================================


def log(msg, log_file=LOG_FILE):
    """Print and log to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(log_file, 'a') as f:
        f.write(full_msg + '\n')


class TDAConsistencyLoss(nn.Module):
    """TDA Consistency Regularization Loss"""
    
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, tda_anchor, tda_positive, tda_negative):
        pos_sim = F.cosine_similarity(tda_anchor, tda_positive, dim=1)
        neg_sim = F.cosine_similarity(tda_anchor, tda_negative, dim=1)
        loss = F.relu(self.margin - (pos_sim - neg_sim))
        return loss.mean()


class TripletDatasetWithTDA(Dataset):
    """Triplet dataset with TDA features for regularization"""
    
    def __init__(self, image_paths, labels, label_map, transform=None,
                 tda_features=None, path_to_tda_idx=None):
        self.image_paths = [str(p) for p in image_paths]
        self.labels = np.array(labels)
        self.label_map = label_map
        self.transform = transform
        self.tda_features = tda_features
        self.path_to_tda_idx = path_to_tda_idx
        self.unique_labels = list(label_map.keys())
        from PIL import Image
        self.Image = Image
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        import random
        
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]
        
        positive_indices = self.label_map[anchor_label]
        positive_idx = random.choice(positive_indices)
        while positive_idx == idx and len(positive_indices) > 1:
            positive_idx = random.choice(positive_indices)
        positive_path = self.image_paths[positive_idx]
        
        negative_label = random.choice(self.unique_labels)
        while negative_label == anchor_label:
            negative_label = random.choice(self.unique_labels)
        negative_idx = random.choice(self.label_map[negative_label])
        negative_path = self.image_paths[negative_idx]
        
        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)
        negative_img = self._load_image(negative_path)
        
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
        path_str = str(path)
        if path_str in self.path_to_tda_idx:
            idx = self.path_to_tda_idx[path_str]
            return torch.from_numpy(self.tda_features[idx].copy()).float()
        parts = path_str.replace('\\', '/').split('/')
        if len(parts) >= 2:
            key = parts[-2] + '/' + parts[-1]
            if key in self.path_to_tda_idx:
                idx = self.path_to_tda_idx[key]
                return torch.from_numpy(self.tda_features[idx].copy()).float()
        return torch.zeros(self.tda_features.shape[1], dtype=torch.float32)


def load_tda_cache(cache_path, image_paths):
    """Load TDA cache and create path-to-index mapping"""
    if not cache_path.exists():
        log(f"ERROR: Edge TDA cache not found at {cache_path}")
        log("Run precompute_edge_tda.py first!")
        return None, None
    
    log(f"Loading edge-based TDA cache from {cache_path}...")
    tda_data = np.load(cache_path, allow_pickle=True)
    tda_features = tda_data['features']
    tda_paths = tda_data['paths']
    
    log(f"TDA features shape: {tda_features.shape}")
    
    path_to_idx = {}
    for i, p in enumerate(tda_paths):
        path_str = str(p)
        path_to_idx[path_str] = i
        parts = path_str.replace('\\', '/').split('/')
        if len(parts) >= 2:
            key = parts[-2] + '/' + parts[-1]
            path_to_idx[key] = i
    
    found = 0
    for path in image_paths:
        path_str = str(path)
        parts = path_str.replace('\\', '/').split('/')
        key = parts[-2] + '/' + parts[-1] if len(parts) >= 2 else path_str
        if key in path_to_idx or path_str in path_to_idx:
            found += 1
    
    log(f"Matched {found}/{len(image_paths)} training images to edge TDA cache")
    return tda_features, path_to_idx


def set_seed(seed=RANDOM_SEED):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, triplet_loss, tda_loss, optimizer, device, 
                epoch, tda_lambda, use_tda):
    """Train for one epoch"""
    model.train()
    
    total_triplet_loss = 0.0
    total_tda_loss = 0.0
    total_loss = 0.0
    correct_triplets = 0
    total_samples = 0
    
    effective_lambda = tda_lambda
    
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
        
        anchor_embed = model(anchor_img, mode='metric')
        pos_embed = model(pos_img, mode='metric')
        neg_embed = model(neg_img, mode='metric')
        
        loss_triplet = triplet_loss(anchor_embed, pos_embed, neg_embed)
        
        loss_tda = torch.tensor(0.0, device=device)
        if has_tda and effective_lambda > 0:
            loss_tda = tda_loss(tda_a, tda_p, tda_n)
        
        loss = loss_triplet + effective_lambda * loss_tda
        
        loss.backward()
        optimizer.step()
        
        batch_size = anchor_img.size(0)
        total_triplet_loss += loss_triplet.item() * batch_size
        total_tda_loss += loss_tda.item() * batch_size
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        with torch.no_grad():
            pos_dist = F.pairwise_distance(anchor_embed, pos_embed)
            neg_dist = F.pairwise_distance(anchor_embed, neg_embed)
            correct_triplets += (pos_dist < neg_dist).sum().item()
    
    return (total_loss / total_samples, 
            total_triplet_loss / total_samples,
            total_tda_loss / total_samples,
            correct_triplets / total_samples,
            effective_lambda)


def validate(model, dataloader, triplet_loss, device):
    """Validate model"""
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
    log("=" * 70)
    log(f"EXPERIMENT: {EXPERIMENT_NAME}")
    log("=" * 70)
    log(f"Edge-based TDA Regularization with λ={TDA_LAMBDA}")
    log(f"Output directory: {OUTPUT_DIR}")
    
    set_seed()
    
    # Configuration
    log(f"\nConfiguration:")
    log(f"  Epochs: {NUM_EPOCHS}")
    log(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    log(f"  Learning rate: {LEARNING_RATE}")
    log(f"  TDA lambda: {TDA_LAMBDA}")
    log(f"  TDA type: Edge-based (Canny)")
    log(f"  Device: {DEVICE}")
    
    # Load data
    log("\n" + "-" * 70)
    log("Loading data...")
    
    train_paths, train_labels, train_label_map, train_num_classes = load_classification_data(TRAIN_DIR)
    log(f"Training: {len(train_paths)} images from {train_num_classes} identities")
    
    val_paths, val_labels, val_label_map, val_num_classes = load_classification_data(VAL_DIR)
    log(f"Validation: {len(val_paths)} images from {val_num_classes} identities")
    
    # Load edge-based TDA cache
    tda_features, path_to_tda_idx = load_tda_cache(TDA_CACHE_PATH, train_paths)
    use_tda = tda_features is not None
    
    if not use_tda:
        log("\nERROR: Cannot proceed without edge TDA cache.")
        log("Run: python experiments/exp_b3_edge_tda/precompute_edge_tda.py")
        return
    
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = TripletDatasetWithTDA(
        train_paths, train_labels, train_label_map, train_transform,
        tda_features=tda_features, path_to_tda_idx=path_to_tda_idx
    )
    val_dataset = TripletDatasetWithTDA(
        val_paths, val_labels, val_label_map, val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    log(f"Train batches: {len(train_loader)}")
    log(f"Val batches: {len(val_loader)}")
    
    # Create model
    log("\n" + "-" * 70)
    log("Initializing CNN model...")
    
    model = FaceEmbeddingCNN(
        embedding_dim=EMBEDDING_DIM,
        num_classes=train_num_classes,
        use_cbam=True,
    )
    
    if PRETRAINED_PATH.exists():
        log(f"Loading pretrained CNN weights from {PRETRAINED_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
        log("✓ Pretrained weights loaded successfully")
    else:
        log("No pretrained weights found, training from scratch")
    
    model = model.to(DEVICE)
    total_params, trainable_params = count_parameters(model)
    log(f"Total parameters: {total_params:,}")
    
    # Loss functions
    _, triplet_loss = get_loss_functions()
    tda_loss = TDAConsistencyLoss(margin=0.3)
    
    # Optimizer with cosine annealing
    lr = LEARNING_RATE * 0.1 if PRETRAINED_PATH.exists() else LEARNING_RATE
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training
    log("\n" + "-" * 70)
    log(f"Starting training for {NUM_EPOCHS} epochs...")
    log(f"TDA regularization: Edge-based (Enabled)")
    log("-" * 70)
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        train_loss, triplet_loss_val, tda_loss_val, train_acc, eff_lambda = train_epoch(
            model, train_loader, triplet_loss, tda_loss, optimizer, DEVICE,
            epoch, TDA_LAMBDA, use_tda
        )
        
        val_loss, val_acc = validate(model, val_loader, triplet_loss, DEVICE)
        
        scheduler.step()
        
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
            'tda_type': 'edge_canny',
            'time': epoch_time,
        })
        
        # Save history after each epoch
        with open(HISTORY_SAVE_PATH, 'w') as f:
            json.dump(history, f, indent=2)
        
        gap = train_acc - val_acc
        log(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        log(f"  Losses - Total: {train_loss:.4f} | Triplet: {triplet_loss_val:.4f} | TDA: {tda_loss_val:.4f} (λ={eff_lambda:.3f})")
        log(f"  Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Gap: {gap*100:+.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            log(f"  ✓ New best model saved (val_acc: {val_acc*100:.2f}%)")
        else:
            epochs_without_improvement += 1
            log(f"  No improvement ({epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})")
            
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                log(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    log("\n" + "=" * 70)
    log("TRAINING COMPLETE")
    log("=" * 70)
    log(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")
    log(f"Model saved to: {MODEL_SAVE_PATH}")
    
    # Comparison with baselines
    log("\n" + "-" * 70)
    log("Comparison with Baselines:")
    log("-" * 70)
    log(f"  CNN Baseline:              94.15%")
    log(f"  TDA Intensity (λ=0.10):    94.45%")
    log(f"  Exp A1 Intensity (λ=0.05): 94.54%")
    log(f"  This experiment (Edge):    {best_val_acc*100:.2f}%")
    improvement = best_val_acc - 0.9454
    if improvement > 0:
        log(f"  ✓ IMPROVEMENT over A1: +{improvement*100:.2f}%")
    else:
        log(f"  Δ vs A1: {improvement*100:.2f}%")


if __name__ == "__main__":
    main()
