#!/usr/bin/env python3
"""
Training script for CNN-only baseline (FaceEmbeddingCNN)

This trains the CNN backbone without TDA features to establish a baseline
and create pretrained weights for the dual-stream model.

Usage:
    conda run -n face_recog python scripts/train_cnn_baseline.py
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

# Try relative import first (when used as package), fall back to absolute
try:
    from .config import (
        DEVICE, LEARNING_RATE, RANDOM_SEED, OUTPUT_DIR,
        WEIGHT_DECAY, BATCH_SIZE, TRAIN_DIR, VAL_DIR
    )
except ImportError:
    from config import (
        DEVICE, LEARNING_RATE, RANDOM_SEED, OUTPUT_DIR,
        WEIGHT_DECAY, BATCH_SIZE, TRAIN_DIR, VAL_DIR
    )

from models import FaceEmbeddingCNN, get_loss_functions, count_parameters
from data_loader import (
    load_classification_data, get_transforms,
    TripletDataset
)


# Training configuration
NUM_EPOCHS = 20
EMBEDDING_DIM = 256
DROPOUT = 0.3
USE_LR_SCHEDULER = True

# Output paths
MODEL_SAVE_PATH = OUTPUT_DIR / "best_cnn_baseline_model.pth"
HISTORY_SAVE_PATH = OUTPUT_DIR / "cnn_baseline_history.json"


def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility"""
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct_triplets = 0
    total_samples = 0
    
    for anchor_img, pos_img, neg_img in tqdm(dataloader, desc="Training", leave=False):
        anchor_img = anchor_img.to(device)
        pos_img = pos_img.to(device)
        neg_img = neg_img.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (CNN only, no TDA)
        anchor_embed = model(anchor_img, mode='metric')
        pos_embed = model(pos_img, mode='metric')
        neg_embed = model(neg_img, mode='metric')
        
        loss = criterion(anchor_embed, pos_embed, neg_embed)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * anchor_img.size(0)
        total_samples += anchor_img.size(0)
        
        # Calculate accuracy
        with torch.no_grad():
            pos_dist = F.pairwise_distance(anchor_embed, pos_embed)
            neg_dist = F.pairwise_distance(anchor_embed, neg_embed)
            correct_triplets += (pos_dist < neg_dist).sum().item()
    
    avg_loss = total_loss / total_samples
    accuracy = correct_triplets / total_samples
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct_triplets = 0
    total_samples = 0
    
    with torch.no_grad():
        for anchor_img, pos_img, neg_img in tqdm(dataloader, desc="Validating", leave=False):
            anchor_img = anchor_img.to(device)
            pos_img = pos_img.to(device)
            neg_img = neg_img.to(device)
            
            anchor_embed = model(anchor_img, mode='metric')
            pos_embed = model(pos_img, mode='metric')
            neg_embed = model(neg_img, mode='metric')
            
            loss = criterion(anchor_embed, pos_embed, neg_embed)
            
            total_loss += loss.item() * anchor_img.size(0)
            total_samples += anchor_img.size(0)
            
            pos_dist = F.pairwise_distance(anchor_embed, pos_embed)
            neg_dist = F.pairwise_distance(anchor_embed, neg_embed)
            correct_triplets += (pos_dist < neg_dist).sum().item()
    
    avg_loss = total_loss / total_samples
    accuracy = correct_triplets / total_samples
    return avg_loss, accuracy


def main():
    print("=" * 70)
    print("CNN BASELINE TRAINING (No TDA)")
    print("=" * 70)
    
    set_seed()
    
    # Configuration summary
    print("\nConfiguration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Embedding dim: {EMBEDDING_DIM}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Device: {DEVICE}")
    print(f"  Use LR scheduler: {USE_LR_SCHEDULER}")
    
    # Load data
    print("\n" + "-" * 70)
    print("Loading data...")
    
    train_paths, train_labels, train_label_map, train_num_classes = load_classification_data(TRAIN_DIR)
    print(f"  Training: {len(train_paths)} images from {train_num_classes} identities")
    
    val_paths, val_labels, val_label_map, val_num_classes = load_classification_data(VAL_DIR)
    print(f"  Validation: {len(val_paths)} images from {val_num_classes} identities")
    
    train_transform, val_transform = get_transforms()
    
    # Create datasets (CNN only - no TDA)
    train_dataset = TripletDataset(
        train_paths, train_labels, train_label_map, train_transform
    )
    val_dataset = TripletDataset(
        val_paths, val_labels, val_label_map, val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
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
    
    model = model.to(DEVICE)
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    _, loss_fn = get_loss_functions()  # triplet loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    print("\n" + "-" * 70)
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print("-" * 70)
    
    best_val_acc = 0.0
    history = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, loss_fn, DEVICE)
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr,
            'time': epoch_time,
        })
        
        # Print progress
        gap = train_acc - val_acc
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% ")
        print(f"  Gap: {gap*100:+.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ New best model saved (val_acc: {val_acc*100:.2f}%)")
        
        # Update scheduler
        if scheduler:
            scheduler.step(val_loss)
    
    # Save history
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Training history saved to: {HISTORY_SAVE_PATH}")
    
    print("\n" + "-" * 70)
    print("Training Summary:")
    print("-" * 70)
    print(f"  Final Train Acc: {history[-1]['train_acc']*100:.2f}%")
    print(f"  Final Val Acc:   {history[-1]['val_acc']*100:.2f}%")
    print(f"  Best Val Acc:    {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    main()
