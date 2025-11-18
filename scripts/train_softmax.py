"""
Training script for Softmax (Classification) model
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.optim as optim
from tqdm import tqdm

from config import (
    DEVICE, NUM_EPOCHS_SOFTMAX, LEARNING_RATE, MODEL_SOFTMAX_PATH,
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
    model.train()  # training mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # tqdm progress bar
    for images, labels in tqdm(dataloader, desc="Training (Softmax)", leave=False):
        # use GPU
        images, labels = images.to(device), labels.to(device)
        
        # zero the gradients before backpropagation
        optimizer.zero_grad()

        # forward pass for classification task
        outputs = model(images, mode='classification')

        # calculate loss
        loss = criterion(outputs, labels)
        
        # backpropagate to compute gradients
        loss.backward()

        # update the weights
        optimizer.step()
        
        # stats
        total_loss += loss.item() * images.size(0)
        
        # prediction
        _, predicted = torch.max(outputs.data, 1)

        # update stats for accuracy
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
    # avg loss and accuracy for epoch
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()  # evaluation mode disables dropout
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # no need to track gradients during validation
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating (Softmax)", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images, mode='classification')
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def main():
    print("=" * 70)
    print("SOFTMAX (CLASSIFICATION) TRAINING")
    print("=" * 70)
    
    # Configuration
    print_config()
    set_seed()
    
    # Load data
    print("\nLoading data...")
    data_dict = create_dataloaders()
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    num_classes = data_dict['train_num_classes']
    
    # Create model
    print("\nInitializing model...")
    model = FaceEmbeddingCNN(embedding_dim=256, num_classes=num_classes, use_cbam=USE_CBAM).to(DEVICE)
    print(f"Model created with {num_classes} classes")
    
    # Loss and optimizer
    loss_fn, _ = get_loss_functions()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS_SOFTMAX} epochs...")
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(NUM_EPOCHS_SOFTMAX):
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
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS_SOFTMAX} | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SOFTMAX_PATH)
            print(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f})")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_SOFTMAX_PATH}")
    
    # Save history
    import json
    history_path = MODEL_SOFTMAX_PATH.parent / "softmax_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
