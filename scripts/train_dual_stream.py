#!/usr/bin/env python3
"""
Training script for Dual-Stream Face Network (CNN + TDA)

This script trains the DualStreamFaceNet model that combines:
- Stream 1: CNN backbone processing images
- Stream 2: TDA features from precomputed persistence images

Uses triplet loss for metric learning.

Usage:
    conda run -n face_recog python scripts/train_dual_stream.py
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

from config import (
    DEVICE, LEARNING_RATE, RANDOM_SEED, OUTPUT_DIR,
    WEIGHT_DECAY
)
from models import DualStreamFaceNet, FaceEmbeddingCNN, get_loss_functions, count_parameters
from data_loader import create_tda_dataloaders


# Training configuration
NUM_EPOCHS = 20
EMBEDDING_DIM = 256
TDA_DIM = 400
FUSION_STRATEGY = 'attention'  # Options: 'attention' (best), 'gated', 'residual', 'concat'
FUSION_HIDDEN_DIM = 384
DROPOUT = 0.3
USE_LR_SCHEDULER = True  # Use ReduceLROnPlateau
PRETRAINED_CNN_PATH = OUTPUT_DIR / "best_cnn_baseline_model.pth"  # CNN baseline weights
USE_PRETRAINED = True  # Whether to use pretrained CNN weights

# Dynamic output paths based on fusion strategy and pretrained status
pretrained_suffix = "_pretrained" if USE_PRETRAINED else "_scratch"
MODEL_SAVE_PATH = OUTPUT_DIR / f"best_dual_stream_{FUSION_STRATEGY}{pretrained_suffix}_model.pth"
HISTORY_SAVE_PATH = OUTPUT_DIR / f"dual_stream_{FUSION_STRATEGY}{pretrained_suffix}_history.json"


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
    """Train for one epoch with TDA features"""
    model.train()
    total_loss = 0.0
    correct_triplets = 0
    total_samples = 0
    
    for (anchor_img, anchor_tda), (pos_img, pos_tda), (neg_img, neg_tda) in tqdm(dataloader, desc="Training", leave=False):
        # Move to device
        anchor_img = anchor_img.to(device)
        pos_img = pos_img.to(device)
        neg_img = neg_img.to(device)
        anchor_tda = anchor_tda.to(device)
        pos_tda = pos_tda.to(device)
        neg_tda = neg_tda.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with TDA features
        anchor_embed = model(anchor_img, tda_features=anchor_tda, mode='metric')
        pos_embed = model(pos_img, tda_features=pos_tda, mode='metric')
        neg_embed = model(neg_img, tda_features=neg_tda, mode='metric')
        
        loss = criterion(anchor_embed, pos_embed, neg_embed)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * anchor_img.size(0)
        total_samples += anchor_img.size(0)
        
        # Calculate accuracy (positive distance < negative distance)
        pos_dist = F.pairwise_distance(anchor_embed, pos_embed)
        neg_dist = F.pairwise_distance(anchor_embed, neg_embed)
        correct_triplets += (pos_dist < neg_dist).sum().item()
    
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_triplets / total_samples
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch with TDA features"""
    model.eval()
    total_loss = 0.0
    correct_triplets = 0
    total_samples = 0
    
    with torch.no_grad():
        for (anchor_img, anchor_tda), (pos_img, pos_tda), (neg_img, neg_tda) in tqdm(dataloader, desc="Validating", leave=False):
            anchor_img = anchor_img.to(device)
            pos_img = pos_img.to(device)
            neg_img = neg_img.to(device)
            anchor_tda = anchor_tda.to(device)
            pos_tda = pos_tda.to(device)
            neg_tda = neg_tda.to(device)
            
            anchor_embed = model(anchor_img, tda_features=anchor_tda, mode='metric')
            pos_embed = model(pos_img, tda_features=pos_tda, mode='metric')
            neg_embed = model(neg_img, tda_features=neg_tda, mode='metric')
            
            loss = criterion(anchor_embed, pos_embed, neg_embed)
            
            total_loss += loss.item() * anchor_img.size(0)
            total_samples += anchor_img.size(0)
            
            pos_dist = F.pairwise_distance(anchor_embed, pos_embed)
            neg_dist = F.pairwise_distance(anchor_embed, neg_embed)
            correct_triplets += (pos_dist < neg_dist).sum().item()
    
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_triplets / total_samples
    return epoch_loss, epoch_acc


def main():
    print("=" * 70)
    print("DUAL-STREAM FACE NETWORK TRAINING (CNN + TDA)")
    print("=" * 70)
    
    set_seed()
    
    # Configuration summary
    print("\nConfiguration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Embedding dim: {EMBEDDING_DIM}")
    print(f"  TDA input dim: {TDA_DIM}")
    print(f"  Fusion strategy: {FUSION_STRATEGY}")
    print(f"  Fusion hidden dim: {FUSION_HIDDEN_DIM}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Device: {DEVICE}")
    print(f"  Use LR scheduler: {USE_LR_SCHEDULER}")
    
    # Load data with TDA features
    print("\n" + "-" * 70)
    print("Loading data with TDA features...")
    data_dict = create_tda_dataloaders()
    train_loader = data_dict['triplet_train_loader']
    val_loader = data_dict['triplet_val_loader']
    num_classes = data_dict['train_num_classes']
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "-" * 70)
    print("Initializing model...")
    
    # Option 1: Initialize from pretrained CNN
    if USE_PRETRAINED and PRETRAINED_CNN_PATH.exists():
        print(f"Loading pretrained CNN from: {PRETRAINED_CNN_PATH}")
        cnn_model = FaceEmbeddingCNN(embedding_dim=EMBEDDING_DIM, num_classes=num_classes, use_cbam=True)
        cnn_model.load_state_dict(torch.load(PRETRAINED_CNN_PATH, map_location='cpu'))
        
        model = DualStreamFaceNet.from_pretrained_cnn(
            cnn_model,
            embedding_dim=EMBEDDING_DIM,
            num_classes=num_classes,
            tda_dim=TDA_DIM,
            fusion_strategy=FUSION_STRATEGY,
            fusion_hidden_dim=FUSION_HIDDEN_DIM,
            use_cbam=True,
            dropout=DROPOUT,
        )
        print("  ✓ Loaded pretrained CNN backbone weights")
    else:
        # Option 2: Train from scratch
        if USE_PRETRAINED:
            print(f"Warning: Pretrained model not found at {PRETRAINED_CNN_PATH}")
        print("Training from scratch...")
        model = DualStreamFaceNet(
            embedding_dim=EMBEDDING_DIM,
            num_classes=num_classes,
            tda_dim=TDA_DIM,
            fusion_strategy=FUSION_STRATEGY,
            fusion_hidden_dim=FUSION_HIDDEN_DIM,
            use_cbam=True,
            dropout=DROPOUT,
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
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn, DEVICE)
        
        epoch_time = time.time() - epoch_start_time
        
        # Update scheduler
        if scheduler:
            scheduler.step(val_loss)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        })
        
        acc_gap = train_acc - val_acc
        gap_indicator = "⚠️" if acc_gap > 0.05 else ""
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% {gap_indicator}")
        print(f"  Gap: {acc_gap*100:+.2f}%")
        
        # Save best model (by validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ New best model saved (val_acc: {best_val_acc*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    
    # Save history
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {HISTORY_SAVE_PATH}")
    
    # Final summary
    print("\n" + "-" * 70)
    print("Training Summary:")
    print("-" * 70)
    final_train_acc = history[-1]['train_acc']
    final_val_acc = history[-1]['val_acc']
    final_gap = final_train_acc - final_val_acc
    
    print(f"  Final Train Acc: {final_train_acc*100:.2f}%")
    print(f"  Final Val Acc:   {final_val_acc*100:.2f}%")
    print(f"  Final Gap:       {final_gap*100:+.2f}%")
    print(f"  Best Val Acc:    {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    main()
