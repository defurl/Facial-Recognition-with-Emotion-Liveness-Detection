#!/usr/bin/env python3
"""
TDA-Enhanced Metric Learning Training Script

This script trains the Metric Learning (Triplet Loss) model with 
Topological Data Analysis (TDA) regularization.

Key Enhancements:
- TDA Regularization: Applies persistent homology to CBAM spatial attention maps
- Topology-Aware Attention: Encourages focused, well-structured attention patterns
- Explainable Features: TDA provides interpretable topological features

Baseline: 94.27% validation accuracy (31 epochs, ~3 min/epoch)
"""

import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Setup paths
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

# Import project modules
from models import FaceEmbeddingCNN, get_loss_functions, count_parameters
from config import (
    TRAIN_DIR, VAL_DIR, OUTPUT_DIR,
    IMG_SIZE, EMBEDDING_DIM, IMAGENET_MEAN, IMAGENET_STD,
    USE_TDA, TDA_LOSS_WEIGHT, TDA_APPLY_EVERY_N_BATCHES,
    TDA_TARGET_ENTROPY, TDA_ENTROPY_WEIGHT, TDA_COMPLEXITY_WEIGHT
)

# Import TDA modules
from tda_modules import (
    is_tda_available, 
    AttentionTopologyLoss, 
    TDAAnalyzer,
    compute_attention_topology_loss
)


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TripletDataset(Dataset):
    """Triplet dataset for metric learning"""
    
    def __init__(self, root_dir, transform=None, max_images_per_identity=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_images_per_identity = max_images_per_identity
        
        # Get all identity folders
        self.identities = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.num_classes = len(self.identities)
        
        # Build image list per identity
        self.identity_to_images = {}
        self.all_images = []
        
        for idx, identity_dir in enumerate(self.identities):
            images = list(identity_dir.glob('*.jpg')) + list(identity_dir.glob('*.png'))
            
            if max_images_per_identity is not None:
                images = images[:max_images_per_identity]
            
            if len(images) >= 2:
                self.identity_to_images[idx] = images
                for img in images:
                    self.all_images.append((img, idx))
        
        self.valid_identities = list(self.identity_to_images.keys())
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        anchor_path, anchor_identity = self.all_images[idx]
        
        # Get positive (same identity)
        positive_images = [img for img in self.identity_to_images[anchor_identity] if img != anchor_path]
        if len(positive_images) == 0:
            positive_images = self.identity_to_images[anchor_identity]
        positive_path = random.choice(positive_images)
        
        # Get negative (different identity)
        negative_identity = anchor_identity
        while negative_identity == anchor_identity:
            negative_identity = random.choice(self.valid_identities)
        negative_path = random.choice(self.identity_to_images[negative_identity])
        
        # Load and transform
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative


def train_epoch_with_tda(model, dataloader, triplet_criterion, tda_loss_fn, optimizer, device, config):
    """Train for one epoch with TDA regularization"""
    model.train()
    total_triplet_loss = 0.0
    total_tda_loss = 0.0
    correct_triplets = 0
    total_samples = 0
    
    use_tda = config['use_tda'] and tda_loss_fn is not None
    tda_weight = config['tda_loss_weight']
    apply_every_n = config['tda_apply_every_n']
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (anchor, positive, negative) in enumerate(pbar):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with attention extraction for TDA
        if use_tda and (batch_idx % apply_every_n == 0):
            anchor_embed, anchor_att = model(anchor, mode='metric', return_attention=True)
            positive_embed = model(positive, mode='metric')
            negative_embed = model(negative, mode='metric')
        else:
            anchor_embed = model(anchor, mode='metric')
            positive_embed = model(positive, mode='metric')
            negative_embed = model(negative, mode='metric')
            anchor_att = None
        
        # Triplet loss
        loss_triplet = triplet_criterion(anchor_embed, positive_embed, negative_embed)
        
        # TDA loss (if enabled and this batch)
        if use_tda and anchor_att is not None:
            loss_tda = tda_loss_fn(anchor_att)
            total_loss = loss_triplet + tda_weight * loss_tda
            total_tda_loss += loss_tda.item() * anchor.size(0)
        else:
            total_loss = loss_triplet
        
        total_loss.backward()
        optimizer.step()
        
        # Statistics
        total_triplet_loss += loss_triplet.item() * anchor.size(0)
        total_samples += anchor.size(0)
        
        # Triplet accuracy
        pos_dist = F.pairwise_distance(anchor_embed, positive_embed)
        neg_dist = F.pairwise_distance(anchor_embed, negative_embed)
        correct_triplets += (pos_dist < neg_dist).sum().item()
        
        pbar.set_postfix({'loss': loss_triplet.item()})
    
    epoch_triplet_loss = total_triplet_loss / total_samples
    epoch_tda_loss = total_tda_loss / total_samples if use_tda else 0.0
    epoch_acc = correct_triplets / total_samples
    
    return epoch_triplet_loss, epoch_tda_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    correct_triplets = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for anchor, positive, negative in pbar:
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


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, epoch)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, epoch)
            self.counter = 0
            self.best_epoch = epoch
    
    def save_checkpoint(self, model, epoch):
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)


def plot_training_curves(history_df, early_stopping, save_dir, tda_enabled):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history_df['epoch'], history_df['train_triplet_loss'], label='Train Triplet')
    axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Val')
    if tda_enabled:
        axes[0].plot(history_df['epoch'], history_df['train_tda_loss'], label='Train TDA', linestyle='--')
    axes[0].axvline(x=early_stopping.best_epoch, color='g', linestyle=':', label=f'Best (E{early_stopping.best_epoch})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history_df['epoch'], history_df['train_acc'] * 100, label='Train')
    axes[1].plot(history_df['epoch'], history_df['val_acc'] * 100, label='Val')
    axes[1].axvline(x=early_stopping.best_epoch, color='g', linestyle=':', label=f'Best (E{early_stopping.best_epoch})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Epoch time
    axes[2].bar(history_df['epoch'], history_df['time'])
    axes[2].axhline(y=history_df['time'].mean(), color='r', linestyle='--', label=f"Avg: {history_df['time'].mean():.1f}s")
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Time (s)')
    axes[2].set_title('Epoch Time')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'tda_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # Device setup
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ WARNING: Running on CPU - training will be slow!")
    
    # Check TDA availability
    TDA_ENABLED = USE_TDA and is_tda_available()
    print(f"\n✓ TDA Enabled: {TDA_ENABLED}")
    if not TDA_ENABLED and USE_TDA:
        print("  ⚠ TDA requested but giotto-tda not available")
    
    # Configuration
    CONFIG = {
        # Base training config
        'max_images_per_identity': None,  # Full dataset
        'num_epochs': 50,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'margin': 0.5,
        
        # Early stopping
        'early_stopping_patience': 10,
        'min_improvement': 0.001,
        
        # Model settings
        'embedding_dim': EMBEDDING_DIM,
        'use_cbam': True,
        
        # TDA settings
        'use_tda': TDA_ENABLED,
        'tda_loss_weight': TDA_LOSS_WEIGHT,
        'tda_apply_every_n': TDA_APPLY_EVERY_N_BATCHES,
        'tda_target_entropy': TDA_TARGET_ENTROPY,
        'tda_entropy_weight': TDA_ENTROPY_WEIGHT,
        'tda_complexity_weight': TDA_COMPLEXITY_WEIGHT,
        
        'seed': 42,
    }
    
    # Set seed
    set_seed(CONFIG['seed'])
    print("✓ Random seeds set")
    
    # Output paths
    SAVE_DIR = OUTPUT_DIR
    
    print("\n" + "=" * 70)
    print("TDA-ENHANCED METRIC LEARNING CONFIGURATION")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Save Directory: {SAVE_DIR}")
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 70)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Create datasets
    train_dataset = TripletDataset(
        TRAIN_DIR, 
        transform=train_transform,
        max_images_per_identity=CONFIG['max_images_per_identity']
    )
    
    val_dataset = TripletDataset(
        VAL_DIR,
        transform=val_transform,
        max_images_per_identity=None
    )
    
    # Dataloaders (num_workers=4 for proper multiprocessing)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Training:   {len(train_dataset):,} images from {train_dataset.num_classes} identities")
    print(f"Validation: {len(val_dataset):,} images from {val_dataset.num_classes} identities")
    print(f"Batches:    {len(train_loader)} train / {len(val_loader)} val")
    print("=" * 60)
    
    # Initialize model
    model = FaceEmbeddingCNN(
        embedding_dim=CONFIG['embedding_dim'],
        num_classes=train_dataset.num_classes,
        use_cbam=CONFIG['use_cbam']
    ).to(DEVICE)
    
    # Loss functions
    _, triplet_loss = get_loss_functions()
    
    # TDA Loss
    if TDA_ENABLED:
        tda_loss_fn = AttentionTopologyLoss(
            target_entropy=CONFIG['tda_target_entropy'],
            entropy_weight=CONFIG['tda_entropy_weight'],
            complexity_weight=CONFIG['tda_complexity_weight']
        )
        print("✓ TDA Loss function initialized")
    else:
        tda_loss_fn = None
        print("⚠ TDA Loss disabled")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Parameters
    total_params, trainable_params = count_parameters(model)
    
    print("\n" + "=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)
    print(f"Model: FaceEmbeddingCNN with CBAM")
    print(f"  Total Params:   {total_params:,}")
    print(f"  Trainable:      {trainable_params:,}")
    print(f"\nLoss: Triplet + TDA Regularization")
    print(f"  Triplet margin: {CONFIG['margin']}")
    print(f"  TDA weight:     {CONFIG['tda_loss_weight']}")
    print("=" * 60)
    
    # Initialize training
    best_model_path = SAVE_DIR / "best_tda_metric_model.pth"
    early_stopping = EarlyStopping(
        patience=CONFIG['early_stopping_patience'],
        min_delta=CONFIG['min_improvement'],
        save_path=best_model_path
    )
    
    history = []
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("STARTING TDA-ENHANCED TRAINING")
    print("=" * 70)
    print(f"Max Epochs: {CONFIG['num_epochs']}")
    print(f"Early Stopping: patience={CONFIG['early_stopping_patience']}")
    print(f"TDA Enabled: {TDA_ENABLED}, Weight: {CONFIG['tda_loss_weight']}")
    print(f"Saving best model to: {best_model_path}")
    print("=" * 70 + "\n")
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_triplet_loss, train_tda_loss, train_acc = train_epoch_with_tda(
            model, train_loader, triplet_loss, tda_loss_fn, optimizer, DEVICE, CONFIG
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, triplet_loss, DEVICE)
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        history.append({
            'epoch': epoch,
            'train_triplet_loss': train_triplet_loss,
            'train_tda_loss': train_tda_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'time': epoch_time
        })
        
        # Early stopping check
        early_stopping(val_loss, model, epoch)
        
        # Print progress
        tda_str = f" | TDA: {train_tda_loss:.4f}" if TDA_ENABLED else ""
        stop_str = f" [ES: {early_stopping.counter}/{early_stopping.patience}]" if early_stopping.counter > 0 else ""
        print(f"Epoch {epoch:2d}/{CONFIG['num_epochs']} | "
              f"Train: {train_triplet_loss:.4f} ({train_acc*100:.2f}%){tda_str} | "
              f"Val: {val_loss:.4f} ({val_acc*100:.2f}%) | "
              f"{epoch_time:.1f}s{stop_str}")
        
        if early_stopping.early_stop:
            print(f"\n⚡ Early stopping triggered at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best epoch: {early_stopping.best_epoch} with val_loss: {early_stopping.best_loss:.4f}")
    print("=" * 70)
    
    # Save training history
    history_path = SAVE_DIR / "tda_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training history saved to: {history_path}")
    
    # Save config
    config_path = SAVE_DIR / "tda_training_config.json"
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"✓ Config saved to: {config_path}")
    
    # Convert to DataFrame
    history_df = pd.DataFrame(history)
    print(f"\nTraining Summary:")
    print(history_df.tail())
    
    # Plot training curves
    plot_training_curves(history_df, early_stopping, SAVE_DIR, TDA_ENABLED)
    print(f"\n✓ Training curves saved to: {SAVE_DIR / 'tda_training_curves.png'}")
    
    # Compare with baseline
    baseline_history_path = OUTPUT_DIR / "metric_training_history.json"
    
    if baseline_history_path.exists():
        with open(baseline_history_path, 'r') as f:
            baseline_history = json.load(f)
        baseline_df = pd.DataFrame(baseline_history)
        
        print("\n" + "=" * 70)
        print("COMPARISON: BASELINE vs TDA-ENHANCED")
        print("=" * 70)
        
        # Baseline stats
        baseline_best_acc = baseline_df['val_acc'].max() * 100
        baseline_best_epoch = baseline_df.loc[baseline_df['val_acc'].idxmax(), 'epoch']
        baseline_final_loss = baseline_df.loc[baseline_df['val_acc'].idxmax(), 'val_loss']
        
        # TDA stats
        tda_best_acc = history_df['val_acc'].max() * 100
        tda_best_epoch = early_stopping.best_epoch
        tda_final_loss = early_stopping.best_loss
        
        print(f"\nMetric              | Baseline (No TDA) | TDA-Enhanced")
        print(f"-" * 60)
        print(f"Best Val Accuracy   | {baseline_best_acc:.2f}%           | {tda_best_acc:.2f}%")
        print(f"Best Val Loss       | {baseline_final_loss:.4f}           | {tda_final_loss:.4f}")
        print(f"Best Epoch          | {int(baseline_best_epoch)}               | {tda_best_epoch}")
        print(f"Total Epochs        | {len(baseline_df)}              | {len(history_df)}")
        print(f"Avg Epoch Time (s)  | {baseline_df['time'].mean():.1f}            | {history_df['time'].mean():.1f}")
        
        improvement = tda_best_acc - baseline_best_acc
        print(f"\n{'✓' if improvement > 0 else '✗'} Accuracy change: {improvement:+.2f}%")
        print("=" * 70)
        
        # Save comparison
        comparison = {
            'baseline_acc': baseline_best_acc,
            'tda_acc': tda_best_acc,
            'improvement': improvement,
            'baseline_epochs': len(baseline_df),
            'tda_epochs': len(history_df)
        }
        with open(SAVE_DIR / 'tda_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
    else:
        print("\nBaseline history not found - skipping comparison")
    
    # Final summary
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     TDA-ENHANCED TRAINING COMPLETE                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║
║  📁 Output Directory: {SAVE_DIR}
║
║  📊 Training Data:
║     • Images Used:    {len(train_dataset):,}
║     • Identities:     {train_dataset.num_classes}
║
║  ⚙️  Model: FaceEmbeddingCNN with CBAM
║     • Parameters:     {total_params:,}
║     • TDA Enabled:    {TDA_ENABLED}
║     • TDA Weight:     {CONFIG['tda_loss_weight']}
║
║  🏃 Training:
║     • Epochs:         {len(history)} / {CONFIG['num_epochs']}
║     • Best Epoch:     {early_stopping.best_epoch}
║     • Best Val Loss:  {early_stopping.best_loss:.4f}
║     • Best Val Acc:   {history_df['val_acc'].max() * 100:.2f}%
║     • Total Time:     {total_time/60:.1f} minutes
║
║  💾 Saved Files:
║     • Best Model:     best_tda_metric_model.pth
║     • History:        tda_training_history.json
║     • Config:         tda_training_config.json
║     • Curves:         tda_training_curves.png
║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
