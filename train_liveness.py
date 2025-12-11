"""
Training script for CNN-based liveness detector
Uses data augmentation to simulate spoof attacks from real face images
"""
import os
# fix duplicate OpenMP runtime on Windows (libiomp5md.dll)
# set before importing libraries that load OpenMP (e.g., torch, cv2)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob
from pathlib import Path
import cv2
import random

import sys
sys.path.insert(0, 'src')
from liveness_cnn import LivenessCNN
from config import TRAIN_DIR, VAL_DIR, OUTPUT_DIR


class LivenessDataset(Dataset):
    """
    Liveness dataset with augmentation-based spoof simulation
    
    Strategy:
    - Real samples: Original face images with standard augmentation
    - Spoof samples: Real images with spoof-simulating transforms:
        * Moiré patterns (simulate screen recapture)
        * Color quantization (simulate printing)
        * Motion blur reduction (simulate static photo)
        * Contrast reduction (simulate print quality loss)
        * JPEG compression artifacts
    """
    
    def __init__(self, real_image_paths, img_size=224, spoof_augment_prob=0.5):
        """
        Args:
            real_image_paths: List of paths to real face images
            img_size: Target image size
            spoof_augment_prob: Probability of applying spoof augmentation
        """
        self.image_paths = real_image_paths
        self.img_size = img_size
        self.spoof_augment_prob = spoof_augment_prob
        
        # Base transforms (for both real and spoof)
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        # Normalization (applied last)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.image_paths) * 2  # Each image generates 1 real + 1 spoof
    
    def __getitem__(self, idx):
        # Determine if this is real or spoof sample
        image_idx = idx // 2
        is_real = (idx % 2 == 0)
        
        # Load image
        img_path = self.image_paths[image_idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply base transform
        image = self.base_transform(image)
        
        if not is_real:
            # Apply spoof simulation
            image = self._apply_spoof_augmentation(image)
        else:
            # Apply real augmentation (slight variations)
            image = self._apply_real_augmentation(image)
        
        # Normalize
        image = self.normalize(image)
        
        # Label: 0 = spoof, 1 = real
        label = 1 if is_real else 0
        
        return image, label
    
    def _apply_real_augmentation(self, image_tensor):
        """Apply subtle augmentations to real faces"""
        # Convert to numpy for OpenCV operations
        img = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Random brightness/contrast
        if random.random() < 0.5:
            alpha = random.uniform(0.9, 1.1)  # Contrast
            beta = random.uniform(-10, 10)    # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Random slight blur (natural camera motion)
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Convert back to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img
    
    def _apply_spoof_augmentation(self, image_tensor):
        """Apply spoof-simulating transformations"""
        # Convert to numpy
        img = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Choose spoof type randomly
        spoof_type = random.choice(['print', 'screen', 'photo'])
        
        if spoof_type == 'print':
            img = self._simulate_print_spoof(img)
        elif spoof_type == 'screen':
            img = self._simulate_screen_spoof(img)
        else:  # photo
            img = self._simulate_photo_spoof(img)
        
        # Convert back to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img
    
    def _simulate_print_spoof(self, img):
        """Simulate printed photo characteristics"""
        # Color quantization (printer has limited color gamut)
        levels = random.randint(16, 32)
        img = (img // (256 // levels)) * (256 // levels)
        
        # Contrast reduction
        img = cv2.convertScaleAbs(img, alpha=0.85, beta=10)
        
        # Add paper texture noise
        noise = np.random.normal(0, 3, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Slight blur (print resolution)
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        return img
    
    def _simulate_screen_spoof(self, img):
        """Simulate screen display characteristics"""
        # Add Moiré pattern (screen refresh + camera interaction)
        h, w = img.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Create interference pattern
        freq = random.uniform(0.05, 0.15)
        moire = np.sin(x * freq) * np.sin(y * freq) * 20
        moire = moire[:, :, np.newaxis].astype(np.int16)
        
        img = np.clip(img.astype(np.int16) + moire, 0, 255).astype(np.uint8)
        
        # Screen refresh lines
        if random.random() < 0.5:
            for i in range(0, h, random.randint(4, 8)):
                img[i, :] = img[i, :] * 0.95
        
        # Reduce sharpness (screen pixel grid)
        img = cv2.GaussianBlur(img, (3, 3), 0.3)
        
        return img
    
    def _simulate_photo_spoof(self, img):
        """Simulate photo-of-photo characteristics"""
        # JPEG compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 85)]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        
        # Flatten texture (loss of detail)
        img = cv2.bilateralFilter(img, 5, 50, 50)
        
        # Reduce dynamic range
        img = cv2.convertScaleAbs(img, alpha=0.9, beta=5)
        
        return img


def train_liveness_detector(
    train_image_dir=str(TRAIN_DIR),
    val_image_dir=str(VAL_DIR),
    backbone_path='best_face_embedding_model.pth',
    output_path=str(OUTPUT_DIR / 'liveness_detector.pth'),
    epochs=20,
    batch_size=32,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train liveness detector
    
    Args:
        train_image_dir: Directory with training face images
        val_image_dir: Directory with validation face images
        backbone_path: Path to pre-trained face recognition model
        output_path: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    """
    print("="*60)
    print("LIVENESS DETECTOR TRAINING")
    print("="*60)
    
    # Collect image paths
    print("\n[1/6] Collecting training images...")
    train_paths = []
    for person_dir in glob.glob(os.path.join(train_image_dir, '*')):
        if os.path.isdir(person_dir):
            images = glob.glob(os.path.join(person_dir, '*.jpg')) + \
                    glob.glob(os.path.join(person_dir, '*.png'))
            train_paths.extend(images)
    
    val_paths = []
    for person_dir in glob.glob(os.path.join(val_image_dir, '*')):
        if os.path.isdir(person_dir):
            images = glob.glob(os.path.join(person_dir, '*.jpg')) + \
                    glob.glob(os.path.join(person_dir, '*.png'))
            val_paths.extend(images)
    
    print(f"   Found {len(train_paths)} training images")
    print(f"   Found {len(val_paths)} validation images")
    
    # Create datasets
    print("\n[2/6] Creating datasets with spoof augmentation...")
    train_dataset = LivenessDataset(train_paths, img_size=224)
    val_dataset = LivenessDataset(val_paths, img_size=224)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"   Training samples: {len(train_dataset)} ({len(train_paths)} real + {len(train_paths)} spoof)")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Create model
    print(f"\n[3/6] Creating model with backbone from {backbone_path}...")
    model = LivenessCNN(backbone_weights=backbone_path, freeze_backbone=False)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Training loop
    print(f"\n[4/6] Training for {epochs} epochs on {device}...")
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = 0  # True positives (real detected as real)
        val_tn = 0  # True negatives (spoof detected as spoof)
        val_fp = 0  # False positives (spoof detected as real)
        val_fn = 0  # False negatives (real detected as spoof)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Calculate confusion matrix components
                val_tp += ((predicted == 1) & (labels == 1)).sum().item()
                val_tn += ((predicted == 0) & (labels == 0)).sum().item()
                val_fp += ((predicted == 1) & (labels == 0)).sum().item()
                val_fn += ((predicted == 0) & (labels == 1)).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate precision and recall
        precision = val_tp / (val_tp + val_fp + 1e-8)
        recall = val_tp / (val_tp + val_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss
            }, output_path)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        scheduler.step(val_acc)
    
    print(f"\n[5/6] Training complete!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved to: {output_path}")
    
    print(f"\n[6/6] Testing final model...")
    # Load best model and test
    checkpoint = torch.load(output_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\nTraining complete! ✓")
    print(f"Use this model by setting LIVENESS_MODEL_PATH = '{output_path}'")


if __name__ == '__main__':
    train_liveness_detector()
