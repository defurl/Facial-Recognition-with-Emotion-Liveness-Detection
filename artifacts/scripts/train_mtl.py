#!/usr/bin/env python3
"""
Multi-Task Learning Training Script for Unified Face Model.

This script implements a multi-phase training strategy:
1. Phase A: Train emotion head only (backbone frozen) on RAF-DB
2. Phase B: Train liveness head only (backbone frozen) on CelebA-Spoof
3. Phase C: Joint fine-tuning of all components

Features:
- Uncertainty-weighted multi-task loss (learned task weights)
- Gradient accumulation for effective large batch sizes
- Mixed precision training (FP16)
- Learning rate scheduling with warmup
- Checkpoint saving and resumption
- TensorBoard logging
- Early stopping

Usage:
    python train_mtl.py --phase emotion --epochs 50
    python train_mtl.py --phase liveness --epochs 30
    python train_mtl.py --phase joint --epochs 100 --resume

Author: MTL Implementation
Date: January 2026
"""

import argparse
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast

# Disable TensorBoard to avoid "Illegal instruction" on some CPUs
HAS_TENSORBOARD = False
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     HAS_TENSORBOARD = True
# except (ImportError, Exception):
#     HAS_TENSORBOARD = False
#     print("Warning: TensorBoard not available (import failed)")

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from models_mtl import UnifiedFaceModel
from losses_mtl import MTLLoss, FocalLoss, LabelSmoothingCrossEntropy
from data_loader_mtl import (
    EmotionDataset, LivenessDataset, 
    create_mtl_dataloaders, get_mtl_transforms,
    create_emotion_sampler,
    EMOTION_DIR, LIVENESS_DIR
)
from config import DEVICE, BASE_DIR


# ============= Training Configuration =============

class TrainingConfig:
    """Central configuration for MTL training."""
    
    # Paths
    OUTPUT_DIR = BASE_DIR / 'outputs' / 'mtl'
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    LOG_DIR = OUTPUT_DIR / 'logs'
    
    # Model
    NUM_IDENTITIES = 4000
    EMBEDDING_DIM = 256
    NUM_EMOTIONS = 7
    NUM_LIVENESS = 2
    
    # Training phases
    PHASES = {
        'emotion': {
            'epochs': 50,
            'lr': 1e-3,
            'freeze_backbone': True,
            'tasks': ['emotion'],
            'batch_size': 64,
        },
        'liveness': {
            'epochs': 30,
            'lr': 1e-3,
            'freeze_backbone': True,
            'tasks': ['liveness'],
            'batch_size': 128,
        },
        'joint': {
            'epochs': 100,
            'lr': 1e-4,
            'freeze_backbone': False,
            'tasks': ['face', 'emotion', 'liveness'],
            'batch_size': 32,
        },
    }
    
    # Optimizer
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    GRADIENT_CLIP = 1.0
    ACCUMULATION_STEPS = 4
    
    # Loss weights (initial, will be learned)
    LOSS_WEIGHTS = {
        'face': 0.5,
        'emotion': 0.3,
        'liveness': 0.2,
    }
    
    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 1e-4
    
    # Mixed precision
    USE_AMP = True if torch.cuda.is_available() else False
    
    @classmethod
    def setup_directories(cls):
        """Create output directories."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============= Training Utilities =============

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class MetricTracker:
    """Track training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, name: str, value: float, count: int = 1):
        if name not in self.metrics:
            self.metrics[name] = 0.0
            self.counts[name] = 0
        self.metrics[name] += value * count
        self.counts[name] += count
    
    def get_average(self, name: str) -> float:
        if name not in self.metrics or self.counts[name] == 0:
            return 0.0
        return self.metrics[name] / self.counts[name]
    
    def get_all_averages(self) -> Dict[str, float]:
        return {name: self.get_average(name) for name in self.metrics}


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


# ============= Training Functions =============

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    config: TrainingConfig,
    phase: str,
    device: torch.device,
    accumulation_steps: int = 4,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for AMP
        config: Training configuration
        phase: Training phase ('emotion', 'liveness', 'joint')
        device: Device to train on
        accumulation_steps: Steps for gradient accumulation
    
    Returns:
        Dictionary of average metrics
    """
    model.train()
    tracker = MetricTracker()
    optimizer.zero_grad()
    
    phase_config = config.PHASES[phase]
    tasks = phase_config['tasks']
    
    for batch_idx, batch in enumerate(dataloader):
        # Handle different batch formats
        if isinstance(batch, dict):
            images = batch['image'].to(device)
            labels = {k: v.to(device) for k, v in batch.items() if k != 'image'}
        else:
            images, labels = batch
            images = images.to(device)
            if isinstance(labels, torch.Tensor):
                # Single task
                if 'emotion' in tasks:
                    labels = {'emotion': labels.to(device)}
                elif 'liveness' in tasks:
                    labels = {'liveness': labels.to(device)}
        
        # Forward pass with mixed precision
        if scaler is not None:
            with autocast():
                outputs = model(images, tasks=tasks)
                losses = {}
                
                # Compute task-specific losses
                if 'emotion' in tasks and 'emotion' in labels:
                    losses['emotion'] = criterion.emotion_loss(
                        outputs['emotion'], labels['emotion']
                    )
                    tracker.update('emotion_acc', 
                                   compute_accuracy(outputs['emotion'], labels['emotion']),
                                   images.size(0))
                
                if 'liveness' in tasks and 'liveness' in labels:
                    losses['liveness'] = criterion.liveness_loss(
                        outputs['liveness'], labels['liveness']
                    )
                    tracker.update('liveness_acc',
                                   compute_accuracy(outputs['liveness'], labels['liveness']),
                                   images.size(0))
                
                # Total loss
                if losses:
                    total_loss = sum(losses.values()) / len(losses)
                else:
                    continue
                
                # Scale for gradient accumulation
                total_loss = total_loss / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
        else:
            # No AMP
            outputs = model(images, tasks=tasks)
            losses = {}
            
            if 'emotion' in tasks and 'emotion' in labels:
                losses['emotion'] = criterion.emotion_loss(
                    outputs['emotion'], labels['emotion']
                )
                tracker.update('emotion_acc',
                               compute_accuracy(outputs['emotion'], labels['emotion']),
                               images.size(0))
            
            if 'liveness' in tasks and 'liveness' in labels:
                losses['liveness'] = criterion.liveness_loss(
                    outputs['liveness'], labels['liveness']
                )
                tracker.update('liveness_acc',
                               compute_accuracy(outputs['liveness'], labels['liveness']),
                               images.size(0))
            
            if losses:
                total_loss = sum(losses.values()) / len(losses)
            else:
                continue
            
            total_loss = total_loss / accumulation_steps
            total_loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
        
        # Track losses
        tracker.update('total_loss', total_loss.item() * accumulation_steps, images.size(0))
        for name, loss in losses.items():
            tracker.update(f'{name}_loss', loss.item(), images.size(0))
    
    return tracker.get_all_averages()


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    phase: str,
    device: torch.device,
    config: TrainingConfig,
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function
        phase: Training phase
        device: Device
        config: Training configuration
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    tracker = MetricTracker()
    
    phase_config = config.PHASES[phase]
    tasks = phase_config['tasks']
    
    for batch in dataloader:
        if isinstance(batch, dict):
            images = batch['image'].to(device)
            labels = {k: v.to(device) for k, v in batch.items() if k != 'image'}
        else:
            images, labels = batch
            images = images.to(device)
            if isinstance(labels, torch.Tensor):
                if 'emotion' in tasks:
                    labels = {'emotion': labels.to(device)}
                elif 'liveness' in tasks:
                    labels = {'liveness': labels.to(device)}
        
        outputs = model(images, tasks=tasks)
        losses = {}
        
        if 'emotion' in tasks and 'emotion' in labels:
            losses['emotion'] = criterion.emotion_loss(
                outputs['emotion'], labels['emotion']
            )
            tracker.update('emotion_acc',
                           compute_accuracy(outputs['emotion'], labels['emotion']),
                           images.size(0))
        
        if 'liveness' in tasks and 'liveness' in labels:
            losses['liveness'] = criterion.liveness_loss(
                outputs['liveness'], labels['liveness']
            )
            tracker.update('liveness_acc',
                           compute_accuracy(outputs['liveness'], labels['liveness']),
                           images.size(0))
        
        if losses:
            total_loss = sum(losses.values()) / len(losses)
            tracker.update('total_loss', total_loss.item(), images.size(0))
            for name, loss in losses.items():
                tracker.update(f'{name}_loss', loss.item(), images.size(0))
    
    return tracker.get_all_averages()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    config: TrainingConfig,
    phase: str,
    is_best: bool = False,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'phase': phase,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save latest
    latest_path = config.CHECKPOINT_DIR / f'{phase}_latest.pth'
    torch.save(checkpoint, latest_path)
    
    # Save best if applicable
    if is_best:
        best_path = config.CHECKPOINT_DIR / f'{phase}_best.pth'
        torch.save(checkpoint, best_path)
        print(f"  → Saved new best model: {best_path}")
    
    # Save epoch checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        epoch_path = config.CHECKPOINT_DIR / f'{phase}_epoch{epoch+1}.pth'
        torch.save(checkpoint, epoch_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: TrainingConfig,
    phase: str,
    load_best: bool = False,
) -> Tuple[int, Dict[str, float]]:
    """Load training checkpoint."""
    if load_best:
        checkpoint_path = config.CHECKPOINT_DIR / f'{phase}_best.pth'
    else:
        checkpoint_path = config.CHECKPOINT_DIR / f'{phase}_latest.pth'
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, {}
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    return checkpoint['epoch'] + 1, checkpoint.get('metrics', {})


# ============= Main Training Function =============

def train_phase(
    phase: str,
    epochs: Optional[int] = None,
    resume: bool = False,
    pretrained_path: Optional[str] = None,
):
    """
    Train a specific phase of the MTL model.
    
    Args:
        phase: 'emotion', 'liveness', or 'joint'
        epochs: Override default epochs
        resume: Resume from checkpoint
        pretrained_path: Path to pretrained model (for joint phase)
    """
    print(f"\n{'='*60}")
    print(f"MTL Training - Phase: {phase.upper()}")
    print(f"{'='*60}\n")
    
    # Setup
    config = TrainingConfig()
    config.setup_directories()
    phase_config = config.PHASES[phase]
    device = DEVICE
    
    if epochs is not None:
        phase_config['epochs'] = epochs
    
    print(f"Device: {device}")
    print(f"Tasks: {phase_config['tasks']}")
    print(f"Epochs: {phase_config['epochs']}")
    print(f"Learning Rate: {phase_config['lr']}")
    print(f"Batch Size: {phase_config['batch_size']}")
    print(f"Freeze Backbone: {phase_config['freeze_backbone']}")
    print()
    
    # Create model
    model = UnifiedFaceModel(
        num_identities=config.NUM_IDENTITIES,
        embedding_dim=config.EMBEDDING_DIM,
        num_emotions=config.NUM_EMOTIONS,
        num_liveness=config.NUM_LIVENESS,
    ).to(device)
    
    # Load pretrained if available
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading pretrained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    # Freeze/unfreeze backbone
    if phase_config['freeze_backbone']:
        model.freeze_backbone()
        print("Backbone frozen")
    else:
        model.unfreeze_backbone()
        print("Backbone unfrozen")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data loaders
    train_tf, val_tf = get_mtl_transforms(augment=True)
    
    # Initialize sampler (for balanced training)
    train_sampler = None
    
    if phase == 'emotion':
        train_ds = EmotionDataset(EMOTION_DIR, split='train', transform=train_tf)
        val_ds = EmotionDataset(EMOTION_DIR, split='test', transform=val_tf)
        
        # Use WeightedRandomSampler for balanced training
        # (This is preferred over class weights in loss - using both is too aggressive)
        train_sampler = create_emotion_sampler(train_ds)
        print("Using WeightedRandomSampler for balanced emotion training")
        
    elif phase == 'liveness':
        train_ds = LivenessDataset(LIVENESS_DIR, split='train', transform=train_tf)
        val_ds = LivenessDataset(LIVENESS_DIR, split='test', transform=val_tf)
    else:  # joint
        # For joint training, we need to combine datasets
        train_ds = EmotionDataset(EMOTION_DIR, split='train', transform=train_tf)
        val_ds = EmotionDataset(EMOTION_DIR, split='test', transform=val_tf)
        train_sampler = create_emotion_sampler(train_ds)
        print("Note: Joint training currently uses emotion dataset only")
        print("      Full MTL training requires LivenessDataset (CelebA-Spoof)")
    
    # Use sampler for emotion, shuffle for others
    train_loader = DataLoader(
        train_ds,
        batch_size=phase_config['batch_size'],
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=phase_config['batch_size'] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create loss function (class weights disabled - using sampler instead)
    # Note: Using BOTH sampler AND class weights is too aggressive and collapses validation
    criterion = MTLLoss(
        face_weight=config.LOSS_WEIGHTS['face'],
        emotion_weight=config.LOSS_WEIGHTS['emotion'],
        liveness_weight=config.LOSS_WEIGHTS['liveness'],
        use_uncertainty_weighting=True,
    ).to(device)
    
    # Create optimizer
    optimizer = AdamW(
        [
            {'params': model.parameters()},
            {'params': criterion.parameters(), 'lr': phase_config['lr'] * 10},
        ],
        lr=phase_config['lr'],
        weight_decay=config.WEIGHT_DECAY,
    )
    
    # Create scheduler
    total_steps = len(train_loader) * phase_config['epochs'] // config.ACCUMULATION_STEPS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=phase_config['lr'],
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
    )
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler() if config.USE_AMP else None
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_loss = float('inf')
    if resume:
        start_epoch, metrics = load_checkpoint(
            model, optimizer, scheduler, config, phase
        )
        if metrics:
            best_val_loss = metrics.get('total_loss', float('inf'))
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
    )
    
    # TensorBoard
    writer = None
    if HAS_TENSORBOARD:
        log_dir = config.LOG_DIR / f'{phase}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(log_dir)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }
    
    print(f"\nStarting training from epoch {start_epoch + 1}...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(start_epoch, phase_config['epochs']):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            phase=phase,
            device=device,
            accumulation_steps=config.ACCUMULATION_STEPS,
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            phase=phase,
            device=device,
            config=config,
        )
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        train_loss = train_metrics.get('total_loss', 0)
        val_loss = val_metrics.get('total_loss', 0)
        
        # Get task-specific accuracy
        if 'emotion' in phase_config['tasks']:
            train_acc = train_metrics.get('emotion_acc', 0)
            val_acc = val_metrics.get('emotion_acc', 0)
        elif 'liveness' in phase_config['tasks']:
            train_acc = train_metrics.get('liveness_acc', 0)
            val_acc = val_metrics.get('liveness_acc', 0)
        else:
            train_acc = val_acc = 0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{phase_config['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Check for best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=val_metrics,
            config=config,
            phase=phase,
            is_best=is_best,
        )
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
        
        print()
    
    # Save training history
    history_path = config.OUTPUT_DIR / f'{phase}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    if writer:
        writer.close()
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"History saved to: {history_path}")
    print(f"Best model saved to: {config.CHECKPOINT_DIR / f'{phase}_best.pth'}")
    print(f"{'='*60}")
    
    return model, history


# ============= Main Entry Point =============

def main():
    parser = argparse.ArgumentParser(description='MTL Training Script')
    parser.add_argument('--phase', type=str, required=True,
                        choices=['emotion', 'liveness', 'joint'],
                        help='Training phase')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides default)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    train_phase(
        phase=args.phase,
        epochs=args.epochs,
        resume=args.resume,
        pretrained_path=args.pretrained,
    )


if __name__ == '__main__':
    main()
