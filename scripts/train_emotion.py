"""
Training script for Emotion Detection using RAF-DB dataset
Supports transfer learning from face embedding model
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import DEVICE, OUTPUT_DIR, LEARNING_RATE
from emotion_model import EmotionCNN, count_parameters, freeze_backbone
from data_loader_emotion import create_emotion_dataloaders, EMOTION_LABELS

# Output paths
EMOTION_MODEL_PATH = OUTPUT_DIR / "best_emotion_model.pth"
EMOTION_HISTORY_PATH = OUTPUT_DIR / "emotion_training_history.json"
PRETRAINED_PATH = OUTPUT_DIR / "best_metric_model.pth"


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0] * len(EMOTION_LABELS)
    class_total = [0] * len(EMOTION_LABELS)
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class stats
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    
    # Compute per-class accuracy
    per_class_acc = {}
    for i in range(len(EMOTION_LABELS)):
        if class_total[i] > 0:
            per_class_acc[EMOTION_LABELS[i]] = 100.0 * class_correct[i] / class_total[i]
        else:
            per_class_acc[EMOTION_LABELS[i]] = 0.0
    
    return val_loss, val_acc, per_class_acc


def train_emotion_model(
    epochs=50,
    batch_size=128,
    learning_rate=None,
    use_pretrained=True,
    freeze_epochs=5,
    patience=10
):
    """
    Train the emotion detection model.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate (default: from config)
        use_pretrained: Use pretrained face embedding model
        freeze_epochs: Number of epochs to freeze backbone (for transfer learning)
        patience: Early stopping patience
    """
    if learning_rate is None:
        learning_rate = LEARNING_RATE
    
    print("=" * 70)
    print("EMOTION DETECTION TRAINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Use pretrained: {use_pretrained}")
    print(f"Freeze epochs: {freeze_epochs}")
    print("=" * 70)
    
    # Create dataloaders
    print("\n[1/4] Loading data...")
    data_dict = create_emotion_dataloaders(batch_size=batch_size)
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    class_weights = data_dict['class_weights'].to(DEVICE)
    
    # Create model
    print("\n[2/4] Creating model...")
    pretrained_path = PRETRAINED_PATH if use_pretrained and PRETRAINED_PATH.exists() else None
    
    model = EmotionCNN(
        num_classes=len(EMOTION_LABELS),
        use_cbam=True,
        pretrained_backbone=pretrained_path
    )
    model = model.to(DEVICE)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Freeze backbone initially for transfer learning
    if use_pretrained and pretrained_path is not None and freeze_epochs > 0:
        print(f"\nFreezing backbone for first {freeze_epochs} epochs...")
        freeze_backbone(model, freeze=True)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'per_class_acc': [],
        'lr': [],
        'best_epoch': 0,
        'best_val_acc': 0.0
    }
    
    # Training loop
    print("\n[3/4] Training...")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs and use_pretrained:
            print(f"\n>>> Unfreezing backbone at epoch {epoch + 1}")
            freeze_backbone(model, freeze=False)
            # Reset optimizer to include all parameters
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate * 0.1,  # Lower LR for fine-tuning
                weight_decay=1e-4
            )
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{epochs} (lr: {current_lr:.6f})")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc, per_class_acc = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print("Per-class accuracy:")
        for emotion, acc in per_class_acc.items():
            print(f"  {emotion}: {acc:.1f}%")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['per_class_acc'].append(per_class_acc)
        history['lr'].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            torch.save(model.state_dict(), EMOTION_MODEL_PATH)
            print(f">>> New best model saved! (Val Acc: {val_acc:.2f}%)")
            
            history['best_epoch'] = best_epoch
            history['best_val_acc'] = best_val_acc
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break
    
    # Save training history
    print("\n[4/4] Saving results...")
    with open(EMOTION_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Model saved to: {EMOTION_MODEL_PATH}")
    print(f"History saved to: {EMOTION_HISTORY_PATH}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Emotion Detection Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--no-pretrained', action='store_true', help='Train from scratch')
    parser.add_argument('--freeze-epochs', type=int, default=5, help='Epochs to freeze backbone')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    train_emotion_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_pretrained=not args.no_pretrained,
        freeze_epochs=args.freeze_epochs,
        patience=args.patience
    )


if __name__ == "__main__":
    main()
