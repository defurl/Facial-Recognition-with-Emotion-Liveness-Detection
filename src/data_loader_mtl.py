"""
Multi-Task Data Loading for Unified Face Model.

This module provides:
1. EmotionDataset - RAF-DB emotion dataset loader
2. LivenessDataset - CelebA-Spoof liveness dataset loader  
3. MultiTaskDataset - Combined dataset for MTL training
4. create_mtl_dataloaders - Factory function for all dataloaders

Dataset Structure Expected:
    dataset/
    ├── emotion/                   # RAF-DB
    │   ├── DATASET/
    │   │   ├── train/
    │   │   │   ├── 1/ ... 7/     # Emotion class folders
    │   │   └── test/
    │   ├── train_labels.csv
    │   └── test_labels.csv
    └── liveness/                  # CelebA-Spoof
        └── CelebA_Spoof/
            ├── train/
            └── test/

Author: MTL Implementation
Date: January 2026
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms

# Import from existing config
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE, RANDOM_SEED,
    BASE_DIR
)

# Dataset paths
DATASET_DIR = BASE_DIR / "dataset"
EMOTION_DIR = DATASET_DIR / "emotion"
LIVENESS_DIR = DATASET_DIR / "liveness"

# Emotion label mapping (RAF-DB uses 1-7, we use 0-6)
EMOTION_MAP = {
    1: 0,  # Surprise -> 0
    2: 1,  # Fear -> 1
    3: 2,  # Disgust -> 2
    4: 3,  # Happiness -> 3
    5: 4,  # Sadness -> 4
    6: 5,  # Anger -> 5
    7: 6,  # Neutral -> 6
}

EMOTION_NAMES = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']


# ============= Data Transforms =============

def get_mtl_transforms(augment: bool = True):
    """
    Get transforms for MTL training.
    
    Args:
        augment: Whether to apply data augmentation
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return train_transform, val_transform


# ============= Emotion Dataset (RAF-DB) =============

class EmotionDataset(Dataset):
    """
    RAF-DB Emotion Dataset.
    
    Structure:
        emotion/
        ├── DATASET/
        │   ├── train/
        │   │   ├── 1/ ... 7/    # Emotion folders (surprise to neutral)
        │   └── test/
        ├── train_labels.csv     # image,label
        └── test_labels.csv
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load labels from CSV
        labels_file = self.root_dir / f'{split}_labels.csv'
        if labels_file.exists():
            self.df = pd.read_csv(labels_file)
            # Construct paths
            self.image_dir = self.root_dir / 'DATASET' / split
        else:
            # Fallback: scan directories
            self.df = self._scan_directories()
            self.image_dir = self.root_dir / 'DATASET' / split
        
        print(f"EmotionDataset ({split}): {len(self.df)} samples")
    
    def _scan_directories(self) -> pd.DataFrame:
        """Scan emotion directories to build dataset."""
        data = []
        base_dir = self.root_dir / 'DATASET' / self.split
        
        for emotion_id in range(1, 8):
            emotion_dir = base_dir / str(emotion_id)
            if emotion_dir.exists():
                for img_path in emotion_dir.glob('*.jpg'):
                    data.append({
                        'image': img_path.name,
                        'label': emotion_id,
                        'folder': str(emotion_id)
                    })
        
        return pd.DataFrame(data)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        
        # Handle different CSV formats
        if 'folder' in row:
            img_path = self.image_dir / row['folder'] / row['image']
        else:
            # Original RAF-DB format: images in numbered folders based on label
            label = int(row['label'])
            img_path = self.image_dir / str(label) / row['image']
            if not img_path.exists():
                # Try flat structure
                img_path = self.image_dir / row['image']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Map label (1-7) to (0-6)
        label = EMOTION_MAP.get(int(row['label']), int(row['label']) - 1)
        
        return image, label


# ============= Liveness Dataset (CelebA-Spoof) =============

class LivenessDataset(Dataset):
    """
    CelebA-Spoof Liveness Dataset.
    
    Structure (expected after extraction):
        liveness/
        └── CelebA_Spoof/
            ├── Data/
            │   └── train/ or test/
            │       └── <subject_id>/
            │           └── <image_files>
            └── metas/
                └── intra_test/
                    └── train_label.json or test_label.json
    
    Labels:
        0 = live (real face)
        1 = spoof (attack)
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        balance_classes: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        
        # Try to find the data
        self.data_dir = self._find_data_dir()
        self.samples = self._load_samples()
        
        if balance_classes and len(self.samples) > 0:
            self.samples = self._balance_samples()
        
        if max_samples and len(self.samples) > max_samples:
            random.seed(RANDOM_SEED)
            self.samples = random.sample(self.samples, max_samples)
        
        print(f"LivenessDataset ({split}): {len(self.samples)} samples")
        if len(self.samples) > 0:
            live_count = sum(1 for _, l in self.samples if l == 0)
            spoof_count = len(self.samples) - live_count
            print(f"  Live: {live_count}, Spoof: {spoof_count}")
    
    def _find_data_dir(self) -> Optional[Path]:
        """Find the data directory in various possible structures."""
        # Try common structures
        candidates = [
            self.root_dir / 'synthetic' / self.split,  # Our synthetic dataset
            self.root_dir / 'CelebA_Spoof' / 'Data' / self.split,
            self.root_dir / 'CelebA_Spoof_' / 'Data' / self.split,
            self.root_dir / 'Data' / self.split,
            self.root_dir / self.split,
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        return None
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load sample paths and labels."""
        if self.data_dir is None:
            print(f"Warning: Data directory not found for {self.split}")
            return []
        
        samples = []
        
        # Check for synthetic dataset structure (live/ and spoof/ folders)
        live_dir = self.data_dir / 'live'
        spoof_dir = self.data_dir / 'spoof'
        
        if live_dir.exists() and spoof_dir.exists():
            # Synthetic dataset structure
            for img_file in live_dir.glob('*.jpg'):
                samples.append((img_file, 0))  # 0 = live
            for img_file in spoof_dir.glob('*.jpg'):
                samples.append((img_file, 1))  # 1 = spoof
            return samples
        
        # Try to load from label file first (CelebA-Spoof)
        label_file = self.root_dir / 'CelebA_Spoof' / 'metas' / 'intra_test' / f'{self.split}_label.json'
        
        if label_file.exists():
            import json
            with open(label_file, 'r') as f:
                labels = json.load(f)
            
            for img_path, label_info in labels.items():
                full_path = self.root_dir / 'CelebA_Spoof' / img_path
                if full_path.exists():
                    # Label is the first element (0=live, 1=spoof)
                    label = int(label_info[0]) if isinstance(label_info, list) else int(label_info)
                    samples.append((full_path, label))
        else:
            # Scan directories - use folder structure or filename patterns
            for subject_dir in self.data_dir.iterdir():
                if not subject_dir.is_dir():
                    continue
                
                for img_file in subject_dir.glob('*.jpg'):
                    # Infer label from filename or parent folder
                    # CelebA-Spoof: 'live' in path or specific naming convention
                    filename = img_file.stem.lower()
                    
                    # Check for spoof indicators
                    if any(x in filename for x in ['spoof', 'print', 'replay', 'mask']):
                        label = 1  # Spoof
                    else:
                        label = 0  # Live
                    
                    samples.append((img_file, label))
        
        return samples
    
    def _balance_samples(self) -> List[Tuple[Path, int]]:
        """Balance live/spoof samples."""
        live = [(p, l) for p, l in self.samples if l == 0]
        spoof = [(p, l) for p, l in self.samples if l == 1]
        
        min_count = min(len(live), len(spoof))
        if min_count == 0:
            return self.samples
        
        random.seed(RANDOM_SEED)
        live = random.sample(live, min_count)
        spoof = random.sample(spoof, min_count)
        
        balanced = live + spoof
        random.shuffle(balanced)
        
        return balanced
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============= Multi-Task Dataset =============

class MultiTaskSample:
    """Container for multi-task sample data."""
    def __init__(
        self,
        image: torch.Tensor,
        identity: Optional[int] = None,
        emotion: Optional[int] = None,
        liveness: Optional[int] = None,
        task: str = 'all'
    ):
        self.image = image
        self.identity = identity
        self.emotion = emotion
        self.liveness = liveness
        self.task = task


class MultiTaskDataset(Dataset):
    """
    Combined dataset for multi-task learning.
    
    Samples from multiple task-specific datasets and returns
    with available labels (missing labels are set to -1).
    """
    
    def __init__(
        self,
        face_dataset: Optional[Dataset] = None,
        emotion_dataset: Optional[Dataset] = None,
        liveness_dataset: Optional[Dataset] = None,
        sampling_strategy: str = 'balanced'  # 'balanced', 'proportional', 'round_robin'
    ):
        self.face_dataset = face_dataset
        self.emotion_dataset = emotion_dataset
        self.liveness_dataset = liveness_dataset
        self.sampling_strategy = sampling_strategy
        
        # Calculate dataset sizes
        self.face_len = len(face_dataset) if face_dataset else 0
        self.emotion_len = len(emotion_dataset) if emotion_dataset else 0
        self.liveness_len = len(liveness_dataset) if liveness_dataset else 0
        
        # For balanced sampling, use the max dataset size
        if sampling_strategy == 'balanced':
            self.total_len = max(self.face_len, self.emotion_len, self.liveness_len) * 3
        else:
            self.total_len = self.face_len + self.emotion_len + self.liveness_len
        
        print(f"MultiTaskDataset: face={self.face_len}, emotion={self.emotion_len}, liveness={self.liveness_len}")
    
    def __len__(self) -> int:
        return self.total_len
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with task-specific labels.
        
        Returns:
            Dictionary with:
                - 'image': tensor (3, H, W)
                - 'identity': int or -1 if not available
                - 'emotion': int or -1 if not available
                - 'liveness': int or -1 if not available
                - 'task': str indicating source task
        """
        if self.sampling_strategy == 'round_robin':
            task_idx = idx % 3
            sample_idx = idx // 3
        else:
            # Random task selection
            task_idx = random.randint(0, 2)
            sample_idx = idx
        
        result = {
            'identity': -1,
            'emotion': -1,
            'liveness': -1,
            'task': 'none'
        }
        
        # Select based on task
        if task_idx == 0 and self.face_dataset:
            actual_idx = sample_idx % self.face_len
            image, identity = self.face_dataset[actual_idx]
            result['image'] = image
            result['identity'] = identity
            result['task'] = 'face'
        
        elif task_idx == 1 and self.emotion_dataset:
            actual_idx = sample_idx % self.emotion_len
            image, emotion = self.emotion_dataset[actual_idx]
            result['image'] = image
            result['emotion'] = emotion
            result['task'] = 'emotion'
        
        elif task_idx == 2 and self.liveness_dataset:
            actual_idx = sample_idx % self.liveness_len
            image, liveness = self.liveness_dataset[actual_idx]
            result['image'] = image
            result['liveness'] = liveness
            result['task'] = 'liveness'
        
        else:
            # Fallback to any available dataset
            if self.face_dataset:
                image, identity = self.face_dataset[idx % self.face_len]
                result['image'] = image
                result['identity'] = identity
                result['task'] = 'face'
            elif self.emotion_dataset:
                image, emotion = self.emotion_dataset[idx % self.emotion_len]
                result['image'] = image
                result['emotion'] = emotion
                result['task'] = 'emotion'
            elif self.liveness_dataset:
                image, liveness = self.liveness_dataset[idx % self.liveness_len]
                result['image'] = image
                result['liveness'] = liveness
                result['task'] = 'liveness'
        
        return result


def mtl_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for MTL batches.
    
    Groups samples by task and returns batched tensors with masks.
    """
    images = torch.stack([sample['image'] for sample in batch])
    
    identities = torch.tensor([sample['identity'] for sample in batch], dtype=torch.long)
    emotions = torch.tensor([sample['emotion'] for sample in batch], dtype=torch.long)
    liveness = torch.tensor([sample['liveness'] for sample in batch], dtype=torch.long)
    
    # Create masks for valid labels
    identity_mask = identities >= 0
    emotion_mask = emotions >= 0
    liveness_mask = liveness >= 0
    
    return {
        'images': images,
        'identities': identities,
        'emotions': emotions,
        'liveness': liveness,
        'identity_mask': identity_mask,
        'emotion_mask': emotion_mask,
        'liveness_mask': liveness_mask,
        'tasks': [sample['task'] for sample in batch]
    }


# ============= Factory Functions =============

def create_emotion_dataloaders(
    root_dir: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """Create dataloaders for emotion task only."""
    if root_dir is None:
        root_dir = EMOTION_DIR
    
    train_transform, val_transform = get_mtl_transforms(augment=True)
    
    train_dataset = EmotionDataset(root_dir, split='train', transform=train_transform)
    test_dataset = EmotionDataset(root_dir, split='test', transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'test': test_loader,
        'num_classes': 7
    }


def create_liveness_dataloaders(
    root_dir: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    max_samples: Optional[int] = None,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """Create dataloaders for liveness task only."""
    if root_dir is None:
        root_dir = LIVENESS_DIR
    
    train_transform, val_transform = get_mtl_transforms(augment=True)
    
    train_dataset = LivenessDataset(
        root_dir, 
        split='train', 
        transform=train_transform,
        max_samples=max_samples,
        balance_classes=True
    )
    
    test_dataset = LivenessDataset(
        root_dir,
        split='test',
        transform=val_transform,
        max_samples=max_samples // 5 if max_samples else None,
        balance_classes=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'test': test_loader,
        'num_classes': 2
    }


def create_mtl_dataloaders(
    face_train_dataset: Optional[Dataset] = None,
    emotion_root: Optional[Path] = None,
    liveness_root: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    liveness_max_samples: Optional[int] = 50000
) -> Dict[str, DataLoader]:
    """
    Create combined multi-task dataloaders.
    
    Args:
        face_train_dataset: Existing face recognition dataset
        emotion_root: Path to emotion dataset (RAF-DB)
        liveness_root: Path to liveness dataset (CelebA-Spoof)
        batch_size: Batch size
        num_workers: DataLoader workers
        liveness_max_samples: Limit liveness samples (CelebA-Spoof is huge)
    
    Returns:
        Dictionary with 'train' and 'val' DataLoaders
    """
    train_transform, val_transform = get_mtl_transforms(augment=True)
    
    # Emotion datasets
    emotion_train = None
    emotion_val = None
    if emotion_root and emotion_root.exists():
        emotion_train = EmotionDataset(emotion_root, 'train', train_transform)
        emotion_val = EmotionDataset(emotion_root, 'test', val_transform)
    
    # Liveness datasets
    liveness_train = None
    liveness_val = None
    if liveness_root and liveness_root.exists():
        liveness_train = LivenessDataset(
            liveness_root, 'train', train_transform,
            max_samples=liveness_max_samples,
            balance_classes=True
        )
        liveness_val = LivenessDataset(
            liveness_root, 'test', val_transform,
            max_samples=liveness_max_samples // 5 if liveness_max_samples else None,
            balance_classes=True
        )
    
    # Multi-task datasets
    mtl_train = MultiTaskDataset(
        face_dataset=face_train_dataset,
        emotion_dataset=emotion_train,
        liveness_dataset=liveness_train,
        sampling_strategy='balanced'
    )
    
    mtl_val = MultiTaskDataset(
        face_dataset=None,  # Face val uses separate verification
        emotion_dataset=emotion_val,
        liveness_dataset=liveness_val,
        sampling_strategy='balanced'
    )
    
    train_loader = DataLoader(
        mtl_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=mtl_collate_fn
    )
    
    val_loader = DataLoader(
        mtl_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=mtl_collate_fn
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'emotion_train': emotion_train,
        'emotion_val': emotion_val,
        'liveness_train': liveness_train,
        'liveness_val': liveness_val
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MTL Data Loaders")
    print("=" * 60)
    
    # Test emotion dataset
    print("\n--- Testing EmotionDataset ---")
    if EMOTION_DIR.exists():
        try:
            emotion_loaders = create_emotion_dataloaders(EMOTION_DIR, batch_size=8)
            batch = next(iter(emotion_loaders['train']))
            print(f"  Batch images: {batch[0].shape}")
            print(f"  Batch labels: {batch[1]}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"  Emotion dir not found: {EMOTION_DIR}")
    
    # Test liveness dataset
    print("\n--- Testing LivenessDataset ---")
    if LIVENESS_DIR.exists():
        try:
            liveness_loaders = create_liveness_dataloaders(
                LIVENESS_DIR, 
                batch_size=8,
                max_samples=1000
            )
            if len(liveness_loaders['train'].dataset) > 0:
                batch = next(iter(liveness_loaders['train']))
                print(f"  Batch images: {batch[0].shape}")
                print(f"  Batch labels: {batch[1]}")
            else:
                print("  No samples found in liveness dataset")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"  Liveness dir not found: {LIVENESS_DIR}")
    
    print("\n✓ Data loader tests complete!")
