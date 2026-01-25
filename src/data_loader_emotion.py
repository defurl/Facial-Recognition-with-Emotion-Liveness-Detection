"""
Data loading for RAF-DB Emotion Recognition Dataset
Supports folder-based and CSV-based loading
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from config import (
    EMOTION_IMG_SIZE as IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE,
    DATASET_DIR, RANDOM_SEED
)

# RAF-DB specific paths
RAFDB_DIR = DATASET_DIR / "DATASET"
RAFDB_TRAIN_DIR = RAFDB_DIR / "train"
RAFDB_TEST_DIR = RAFDB_DIR / "test"
RAFDB_TRAIN_LABELS = DATASET_DIR / "train_labels.csv"
RAFDB_TEST_LABELS = DATASET_DIR / "test_labels.csv"

# RAF-DB emotion labels (1-indexed in CSV, 0-indexed in model)
EMOTION_LABELS = {
    0: 'Surprise',
    1: 'Fear',
    2: 'Disgust',
    3: 'Happy',
    4: 'Sad',
    5: 'Angry',
    6: 'Neutral'
}

# Mapping from RAF-DB labels (1-7) to model labels (0-6)
RAFDB_TO_MODEL_LABEL = {i: i-1 for i in range(1, 8)}


def get_emotion_transforms():
    """
    Get training and validation transforms for emotion recognition.
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return train_transform, val_transform


class RAFDBDataset(Dataset):
    """
    Dataset for RAF-DB emotion recognition.
    
    Supports two loading modes:
    - folder-based: Load images from folder structure (train/1, train/2, ...)
    - csv-based: Load images using CSV labels
    
    Args:
        data_dir: Path to data directory (e.g., dataset/DATASET/train)
        csv_path: Optional path to labels CSV
        transform: Image transforms
        mode: 'folder' or 'csv'
    """
    
    def __init__(self, data_dir, csv_path=None, transform=None, mode='folder'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mode = mode
        
        self.image_paths = []
        self.labels = []
        
        if mode == 'csv' and csv_path is not None:
            self._load_from_csv(csv_path)
        else:
            self._load_from_folders()
        
        print(f"Loaded {len(self.image_paths)} images from {self.data_dir}")
        self._print_class_distribution()
    
    def _load_from_folders(self):
        """Load images from folder structure (1/, 2/, ..., 7/)"""
        for label_folder in range(1, 8):
            folder_path = self.data_dir / str(label_folder)
            if folder_path.exists():
                for img_path in folder_path.glob("*.jpg"):
                    self.image_paths.append(img_path)
                    # Convert 1-7 to 0-6
                    self.labels.append(label_folder - 1)
                # Also check for png
                for img_path in folder_path.glob("*.png"):
                    self.image_paths.append(img_path)
                    self.labels.append(label_folder - 1)
    
    def _load_from_csv(self, csv_path):
        """Load images using CSV labels"""
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            img_name = row['image']
            label = row['label']
            
            # Find image in the corresponding label folder
            img_path = self.data_dir / str(label) / img_name
            
            if img_path.exists():
                self.image_paths.append(img_path)
                # Convert 1-7 to 0-6
                self.labels.append(label - 1)
    
    def _print_class_distribution(self):
        """Print class distribution"""
        counter = Counter(self.labels)
        print("  Class distribution:")
        for label_idx in sorted(counter.keys()):
            emotion = EMOTION_LABELS[label_idx]
            count = counter[label_idx]
            pct = 100.0 * count / len(self.labels)
            print(f"    {label_idx} ({emotion}): {count} ({pct:.1f}%)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image on error
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self):
        """
        Compute class weights for handling imbalanced data.
        
        Returns:
            Tensor of class weights (inverse frequency)
        """
        counter = Counter(self.labels)
        total = len(self.labels)
        
        weights = []
        for i in range(len(EMOTION_LABELS)):
            count = counter.get(i, 1)  # Avoid division by zero
            weight = total / (len(EMOTION_LABELS) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self):
        """
        Get per-sample weights for WeightedRandomSampler.
        
        Returns:
            List of sample weights
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label].item() for label in self.labels]
        return sample_weights


def create_emotion_dataloaders(batch_size=None, use_weighted_sampler=True, val_split=0.1):
    """
    Create dataloaders for emotion recognition training.
    
    Args:
        batch_size: Batch size (default: from config)
        use_weighted_sampler: Use weighted sampling for imbalanced classes
        val_split: Fraction of training data for validation
        
    Returns:
        Dict with train_loader, val_loader, test_loader, class_weights
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    # Set random seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Get transforms
    train_transform, val_transform = get_emotion_transforms()
    
    # Load full training dataset
    full_train_dataset = RAFDBDataset(
        RAFDB_TRAIN_DIR,
        transform=None,  # Apply transform later
        mode='folder'
    )
    
    # Split into train/val
    n_total = len(full_train_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    indices = list(range(n_total))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create train dataset with augmentation
    train_dataset = RAFDBSubset(
        full_train_dataset.image_paths,
        full_train_dataset.labels,
        train_indices,
        transform=train_transform
    )
    
    # Create val dataset without augmentation
    val_dataset = RAFDBSubset(
        full_train_dataset.image_paths,
        full_train_dataset.labels,
        val_indices,
        transform=val_transform
    )
    
    # Create test dataset
    test_dataset = RAFDBDataset(
        RAFDB_TEST_DIR,
        transform=val_transform,
        mode='folder'
    )
    
    # Compute class weights from full training data
    class_weights = full_train_dataset.get_class_weights()
    print(f"\nClass weights: {class_weights.tolist()}")
    
    # Create samplers
    if use_weighted_sampler:
        # Get sample weights for train subset
        train_labels = [full_train_dataset.labels[i] for i in train_indices]
        sample_weights = [class_weights[label].item() for label in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'class_weights': class_weights,
        'emotion_labels': EMOTION_LABELS
    }


class RAFDBSubset(Dataset):
    """Subset of RAFDB dataset with specific indices and transforms"""
    
    def __init__(self, image_paths, labels, indices, transform=None):
        self.image_paths = [image_paths[i] for i in indices]
        self.labels = [labels[i] for i in indices]
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


if __name__ == "__main__":
    print("Testing RAF-DB data loading...")
    print("=" * 50)
    
    # Test dataloader creation
    data_dict = create_emotion_dataloaders(batch_size=32)
    
    # Test batch loading
    print("\nTesting batch loading...")
    train_loader = data_dict['train_loader']
    images, labels = next(iter(train_loader))
    print(f"  Batch images shape: {images.shape}")
    print(f"  Batch labels shape: {labels.shape}")
    print(f"  Labels in batch: {labels.tolist()}")
    
    print("\n✓ RAF-DB data loading test passed!")
