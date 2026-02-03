"""
Data loading and preprocessing for Face Recognition
"""

import random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Try relative import first (when used as package), fall back to absolute
try:
    from .config import (
        IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE,
        TRAIN_DIR, VAL_DIR, VERIFICATION_VAL_PAIRS
    )
except ImportError:
    from config import (
        IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE,
        TRAIN_DIR, VAL_DIR, VERIFICATION_VAL_PAIRS
    )

# Maximum images per identity for training (None = no limit)
MAX_IMAGES_PER_IDENTITY_TRAIN = None


# ============= Data Transforms =============

def get_transforms():
    """
    Get training and validation transforms.
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
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
    
    return train_transform, val_transform


# ============= Dataset Classes =============

class StandardDataset(Dataset):
    """Standard dataset for classification (softmax) training"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class TripletDataset(Dataset):
    """Triplet dataset for metric learning training"""
    
    def __init__(self, image_paths, labels, label_map, transform=None):
        self.image_paths = image_paths
        self.labels = np.array(labels)
        self.label_map = label_map
        self.transform = transform
        self.unique_labels = list(self.label_map.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Anchor
        anchor_path = self.image_paths[index]
        anchor_label = self.labels[index]

        # Positive (same label as anchor)
        positive_indices = self.label_map[anchor_label]
        positive_index = random.choice(positive_indices)
        while positive_index == index and len(positive_indices) > 1:
            positive_index = random.choice(positive_indices)
        positive_path = self.image_paths[positive_index]

        # Negative (different label)
        negative_label = random.choice(self.unique_labels)
        while negative_label == anchor_label:
            negative_label = random.choice(self.unique_labels)
        negative_index = random.choice(self.label_map[negative_label])
        negative_path = self.image_paths[negative_index]

        # Load images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        # Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img


class TripletDatasetWithTDA(Dataset):
    """
    Triplet dataset that also returns precomputed TDA features.
    
    For dual-stream training where CNN processes images and TDA features
    are looked up from a precomputed cache.
    """
    
    def __init__(self, image_paths, labels, label_map, tda_cache, transform=None):
        """
        Args:
            image_paths: List of image paths
            labels: List of labels
            label_map: Dict mapping label -> list of indices
            tda_cache: TDAFeatureCache object for looking up TDA features
            transform: Image transform
        """
        self.image_paths = [str(p) for p in image_paths]  # Convert to strings for cache lookup
        self.labels = np.array(labels)
        self.label_map = label_map
        self.tda_cache = tda_cache
        self.transform = transform
        self.unique_labels = list(self.label_map.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Anchor
        anchor_path = self.image_paths[index]
        anchor_label = self.labels[index]

        # Positive (same label as anchor)
        positive_indices = self.label_map[anchor_label]
        positive_index = random.choice(positive_indices)
        while positive_index == index and len(positive_indices) > 1:
            positive_index = random.choice(positive_indices)
        positive_path = self.image_paths[positive_index]

        # Negative (different label)
        negative_label = random.choice(self.unique_labels)
        while negative_label == anchor_label:
            negative_label = random.choice(self.unique_labels)
        negative_index = random.choice(self.label_map[negative_label])
        negative_path = self.image_paths[negative_index]

        # Load images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        # Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        # Get TDA features from cache
        anchor_tda = self.tda_cache.get(anchor_path)
        positive_tda = self.tda_cache.get(positive_path)
        negative_tda = self.tda_cache.get(negative_path)
        
        # Handle cache miss (shouldn't happen if precomputed correctly)
        if anchor_tda is None or positive_tda is None or negative_tda is None:
            raise KeyError(f"TDA features not found in cache for one of: {anchor_path}, {positive_path}, {negative_path}")
        
        # Convert to tensors
        anchor_tda = torch.from_numpy(anchor_tda).float()
        positive_tda = torch.from_numpy(positive_tda).float()
        negative_tda = torch.from_numpy(negative_tda).float()
        
        return (anchor_img, anchor_tda), (positive_img, positive_tda), (negative_img, negative_tda)


# ============= Data Loading Functions =============

def load_classification_data(data_dir, max_images_per_identity=None):
    """
    Load image paths and labels from directory structure.
    
    Args:
        data_dir: Path to data directory
        max_images_per_identity: Maximum images to load per identity
    
    Returns:
        Tuple of (image_paths, labels, label_map, num_classes)
    """
    image_paths = []
    labels = []
    label_map = {}
    
    for index, identity_folder in enumerate(sorted(data_dir.iterdir())):
        if identity_folder.is_dir():
            identity_images = list(identity_folder.glob('*.jpg'))
            
            # Limit images per identity
            if max_images_per_identity and len(identity_images) > max_images_per_identity:
                identity_images = identity_images[:max_images_per_identity]
            
            for img_path in identity_images:
                image_paths.append(img_path)
                labels.append(index)
                
                if index not in label_map:
                    label_map[index] = []
                label_map[index].append(len(image_paths) - 1)
    
    num_classes = len(label_map)
    return image_paths, labels, label_map, num_classes


def create_dataloaders():
    """
    Create all dataloaders (standard and triplet, train and val).
    
    Returns:
        Dict with keys: train_loader, val_loader, triplet_train_loader, 
        triplet_val_loader, train_num_classes, val_num_classes
    """
    print("Loading training data...")
    train_paths, train_labels, train_label_map, train_num_classes = load_classification_data(
        TRAIN_DIR, max_images_per_identity=MAX_IMAGES_PER_IDENTITY_TRAIN
    )
    print(f"  Training: {len(train_paths)} images from {train_num_classes} identities")
    
    print("Loading validation data...")
    val_paths, val_labels, val_label_map, val_num_classes = load_classification_data(VAL_DIR)
    print(f"  Validation: {len(val_paths)} images from {val_num_classes} identities")
    
    train_transform, val_transform = get_transforms()
    
    # Standard datasets
    train_dataset = StandardDataset(train_paths, train_labels, train_transform)
    val_dataset = StandardDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # Triplet datasets
    triplet_train_dataset = TripletDataset(train_paths, train_labels, train_label_map, train_transform)
    triplet_val_dataset = TripletDataset(val_paths, val_labels, val_label_map, val_transform)
    
    triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                     num_workers=4, pin_memory=True)
    triplet_val_loader = DataLoader(triplet_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=4, pin_memory=True)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'triplet_train_loader': triplet_train_loader,
        'triplet_val_loader': triplet_val_loader,
        'train_num_classes': train_num_classes,
        'val_num_classes': val_num_classes
    }


def create_tda_dataloaders(tda_cache_dir=None):
    """
    Create triplet dataloaders with TDA features.
    
    Args:
        tda_cache_dir: Directory containing TDA cache files (*.npz)
                       If None, uses default outputs/tda_cache
    
    Returns:
        Dict with keys: triplet_train_loader, triplet_val_loader,
        train_num_classes, val_num_classes, tda_cache_train, tda_cache_val
    """
    from config import OUTPUT_DIR
    from tda_features import TDAFeatureCache
    
    if tda_cache_dir is None:
        tda_cache_dir = OUTPUT_DIR / "tda_cache"
    
    # Load TDA caches
    print("Loading TDA feature caches...")
    tda_cache_train = TDAFeatureCache(tda_cache_dir / "tda_train.npz")
    tda_cache_val = TDAFeatureCache(tda_cache_dir / "tda_val.npz")
    print(f"  Train TDA cache: {len(tda_cache_train.paths)} images")
    print(f"  Val TDA cache: {len(tda_cache_val.paths)} images")
    
    # Load image data
    print("Loading training data...")
    train_paths, train_labels, train_label_map, train_num_classes = load_classification_data(
        TRAIN_DIR, max_images_per_identity=MAX_IMAGES_PER_IDENTITY_TRAIN
    )
    print(f"  Training: {len(train_paths)} images from {train_num_classes} identities")
    
    print("Loading validation data...")
    val_paths, val_labels, val_label_map, val_num_classes = load_classification_data(VAL_DIR)
    print(f"  Validation: {len(val_paths)} images from {val_num_classes} identities")
    
    train_transform, val_transform = get_transforms()
    
    # Create TDA-aware triplet datasets
    triplet_train_dataset = TripletDatasetWithTDA(
        train_paths, train_labels, train_label_map, tda_cache_train, train_transform
    )
    triplet_val_dataset = TripletDatasetWithTDA(
        val_paths, val_labels, val_label_map, tda_cache_val, val_transform
    )
    
    triplet_train_loader = DataLoader(
        triplet_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    triplet_val_loader = DataLoader(
        triplet_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return {
        'triplet_train_loader': triplet_train_loader,
        'triplet_val_loader': triplet_val_loader,
        'train_num_classes': train_num_classes,
        'val_num_classes': val_num_classes,
        'tda_cache_train': tda_cache_train,
        'tda_cache_val': tda_cache_val,
    }


def load_verification_pairs(pairs_file_path):
    """
    Load verification pairs from text file.
    
    Args:
        pairs_file_path: Path to verification pairs text file
    
    Returns:
        List of (img1_path, img2_path, label) tuples
    """
    verification_pairs = []
    
    with open(pairs_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                img1_path = Path(pairs_file_path).parent / parts[0]
                img2_path = Path(pairs_file_path).parent / parts[1]
                label = int(parts[2])
                
                if img1_path.exists() and img2_path.exists():
                    verification_pairs.append((img1_path, img2_path, label))
    
    print(f"Loaded {len(verification_pairs)} verification pairs")
    same_person = sum(1 for _, _, l in verification_pairs if l == 1)
    diff_person = sum(1 for _, _, l in verification_pairs if l == 0)
    print(f"  Same person: {same_person}, Different person: {diff_person}")
    
    return verification_pairs


if __name__ == "__main__":
    print("Testing data loading...")
    data_dict = create_dataloaders()
    print(f"\nDataLoader Summary:")
    print(f"  Train batches: {len(data_dict['train_loader'])}")
    print(f"  Val batches: {len(data_dict['val_loader'])}")
    print(f"  Triplet train batches: {len(data_dict['triplet_train_loader'])}")
    print(f"  Triplet val batches: {len(data_dict['triplet_val_loader'])}")
    print(f"  Train classes: {data_dict['train_num_classes']}")
    print(f"  Val classes: {data_dict['val_num_classes']}")
    print("\n✓ Data loading test passed!")
