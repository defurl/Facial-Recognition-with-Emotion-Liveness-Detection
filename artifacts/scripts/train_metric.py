"""
Training script for Metric Learning (Triplet Loss) model
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from config import (
    DEVICE, NUM_EPOCHS_METRIC, LEARNING_RATE, MODEL_METRIC_PATH,
    RANDOM_SEED, print_config, USE_CBAM
)
from models import FaceEmbeddingCNN, get_loss_functions
from data_loader import create_dataloaders


def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility"""
    import numpy as np
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
    
    for anchor, positive, negative in tqdm(dataloader, desc="Training", leave=False):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        optimizer.zero_grad()
        
        anchor_embed = model(anchor, mode='metric')
        positive_embed = model(positive, mode='metric')
        negative_embed = model(negative, mode='metric')
        
        loss = criterion(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * anchor.size(0)
        total_samples += anchor.size(0)
        
        # Calculate accuracy (positive distance < negative distance)
        pos_dist = F.pairwise_distance(anchor_embed, positive_embed)
        neg_dist = F.pairwise_distance(anchor_embed, negative_embed)
        correct_triplets += (pos_dist < neg_dist).sum().item()
    
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_triplets / total_samples
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    correct_triplets = 0
    total_samples = 0
    
    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc="Validating", leave=False):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            anchor_embed = model(anchor, mode='metric')
            positive_embed = model(positive, mode='metric')
            negative_embed = model(negative, mode='metric')
            
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            
            total_loss += loss.item() * anchor.size(0)
            total_samples += anchor.size(0)
            
            pos_dist = F.pairwise_distance(anchor_embed, positive_embed)
            neg_dist = F.pairwise_distance(anchor_embed, negative_embed)
            correct_triplets += (pos_dist < neg_dist).sum().item()
    
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_triplets / total_samples
    return epoch_loss, epoch_acc


def main():
    print("=" * 70)
    print("METRIC LEARNING (TRIPLET LOSS) TRAINING")
    print("=" * 70)
    
    # Configuration
    print_config()
    set_seed()
    
    # Load data
    print("\nLoading data...")
    data_dict = create_dataloaders()
    train_loader = data_dict['triplet_train_loader']
    val_loader = data_dict['triplet_val_loader']
    num_classes = data_dict['train_num_classes']
    
    # Create model
    print("\nInitializing model...")
    model = FaceEmbeddingCNN(embedding_dim=256, num_classes=num_classes, use_cbam=USE_CBAM).to(DEVICE)
    print(f"Model created with {num_classes} classes")
    
    # Loss and optimizer
    _, loss_fn = get_loss_functions()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS_METRIC} epochs...")
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(NUM_EPOCHS_METRIC):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn, DEVICE)
        
        epoch_time = time.time() - epoch_start_time
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'time': epoch_time
        })
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS_METRIC} | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_METRIC_PATH)
            print(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f})")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_METRIC_PATH}")
    
    # Save history
    import json
    history_path = MODEL_METRIC_PATH.parent / "metric_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
