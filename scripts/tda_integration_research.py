"""
TDA Integration Research: Optimal Fusion Strategies

This script systematically investigates different ways to integrate TDA features
with CNN features for face verification. The goal is to move beyond naive 
concatenation and find architectures that better leverage topological information.

Research Questions:
1. How should TDA features be processed before fusion?
2. What fusion mechanism works best (concat, attention, gating)?
3. Should TDA inform CNN features (cross-attention) or vice versa?
4. At what level should fusion occur (early, mid, late)?

Integration Strategies to Test:
1. Baseline: Naive concatenation (current)
2. Attention Fusion: Learn to weight modalities
3. Gated Fusion: Learn when to use TDA
4. Cross-Modal Attention: TDA guides CNN attention
5. Bilinear Fusion: Multiplicative interactions
6. Multi-level Fusion: Fuse at multiple stages
7. TDA as Auxiliary: Separate TDA head with auxiliary loss

Usage:
    conda run -n face_recog python scripts/tda_integration_research.py --strategy all --epochs 10
"""

import os
import sys
import time
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms
from PIL import Image


# ============================================================================
# FUSION MODULES - Different strategies for combining CNN + TDA
# ============================================================================

class BaseFusion(nn.Module, ABC):
    """Base class for fusion modules."""
    
    @abstractmethod
    def forward(self, cnn_features: torch.Tensor, tda_features: torch.Tensor) -> torch.Tensor:
        """Fuse CNN and TDA features."""
        pass


class ConcatFusion(BaseFusion):
    """
    Strategy 1: Naive Concatenation (Baseline)
    
    Simply concatenates CNN and TDA features, then applies MLP.
    This is what we currently have.
    """
    def __init__(self, cnn_dim: int, tda_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.tda_proj = nn.Sequential(
            nn.Linear(tda_dim, tda_dim // 3),
            nn.BatchNorm1d(tda_dim // 3),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        fusion_dim = cnn_dim + tda_dim // 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, cnn_features, tda_features):
        tda_proj = self.tda_proj(tda_features)
        fused = torch.cat([cnn_features, tda_proj], dim=1)
        return self.fusion(fused)


class AttentionFusion(BaseFusion):
    """
    Strategy 2: Attention-based Fusion
    
    Learn to weight the importance of CNN vs TDA features dynamically.
    Uses attention mechanism to compute modality importance scores.
    
    Key insight: Not all images benefit equally from TDA. Learn when 
    TDA is informative and weight accordingly.
    """
    def __init__(self, cnn_dim: int, tda_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        # Project both to same dimension
        self.hidden_dim = output_dim
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )
        self.tda_proj = nn.Sequential(
            nn.Linear(tda_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 2),  # 2 modalities
            nn.Softmax(dim=1),
        )
        
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, cnn_features, tda_features):
        cnn_proj = self.cnn_proj(cnn_features)  # (B, hidden)
        tda_proj = self.tda_proj(tda_features)  # (B, hidden)
        
        # Compute attention weights
        combined = torch.cat([cnn_proj, tda_proj], dim=1)  # (B, hidden*2)
        weights = self.attention(combined)  # (B, 2)
        
        # Weighted combination
        cnn_weight = weights[:, 0:1]  # (B, 1)
        tda_weight = weights[:, 1:2]  # (B, 1)
        fused = cnn_weight * cnn_proj + tda_weight * tda_proj  # (B, hidden)
        
        return self.output(fused)


class GatedFusion(BaseFusion):
    """
    Strategy 3: Gated Fusion
    
    Use gating mechanism to control information flow from each modality.
    Similar to LSTM gates - learn to selectively pass information.
    
    Key insight: TDA might be noisy for some images. Learn to gate it.
    """
    def __init__(self, cnn_dim: int, tda_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = output_dim
        
        # Projections
        self.cnn_proj = nn.Linear(cnn_dim, self.hidden_dim)
        self.tda_proj = nn.Linear(tda_dim, self.hidden_dim)
        
        # Gates - sigmoid activation for 0-1 range
        self.cnn_gate = nn.Sequential(
            nn.Linear(cnn_dim + tda_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.tda_gate = nn.Sequential(
            nn.Linear(cnn_dim + tda_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        
        # Output
        self.output = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
    
    def forward(self, cnn_features, tda_features):
        # Compute gates based on both modalities
        combined = torch.cat([cnn_features, tda_features], dim=1)
        cnn_g = self.cnn_gate(combined)  # (B, hidden)
        tda_g = self.tda_gate(combined)  # (B, hidden)
        
        # Project and gate
        cnn_proj = self.cnn_proj(cnn_features)  # (B, hidden)
        tda_proj = self.tda_proj(tda_features)  # (B, hidden)
        
        # Gated fusion
        fused = cnn_g * cnn_proj + tda_g * tda_proj
        
        return self.output(fused)


class BilinearFusion(BaseFusion):
    """
    Strategy 4: Bilinear Fusion
    
    Capture multiplicative interactions between CNN and TDA features.
    Instead of just concatenation, compute outer product style interactions.
    
    Key insight: TDA encodes structure, CNN encodes appearance.
    Multiplicative interaction can capture "appearance at structure locations".
    """
    def __init__(self, cnn_dim: int, tda_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        # Low-rank bilinear (for efficiency)
        self.rank = 64  # Low rank approximation
        
        self.cnn_proj = nn.Linear(cnn_dim, self.rank)
        self.tda_proj = nn.Linear(tda_dim, self.rank)
        
        # Bilinear output
        self.bilinear_out = nn.Linear(self.rank, output_dim)
        
        # Also keep additive path
        self.additive = nn.Linear(cnn_dim + tda_dim, output_dim)
        
        self.output = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, cnn_features, tda_features):
        # Bilinear interaction (element-wise product in projected space)
        cnn_proj = self.cnn_proj(cnn_features)  # (B, rank)
        tda_proj = self.tda_proj(tda_features)  # (B, rank)
        bilinear = cnn_proj * tda_proj  # Element-wise product (B, rank)
        bilinear_out = self.bilinear_out(bilinear)  # (B, output_dim)
        
        # Additive path
        additive_out = self.additive(torch.cat([cnn_features, tda_features], dim=1))
        
        # Combine both paths
        fused = bilinear_out + additive_out
        
        return self.output(fused)


class CrossModalAttention(BaseFusion):
    """
    Strategy 5: Cross-Modal Attention
    
    Use TDA features to generate attention weights for CNN features.
    TDA provides structural information about "where" to focus.
    
    Key insight: Persistence images encode topological significance.
    Use this to guide which CNN features are most important.
    """
    def __init__(self, cnn_dim: int, tda_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = output_dim
        
        # TDA generates query for attention
        self.tda_to_query = nn.Sequential(
            nn.Linear(tda_dim, self.hidden_dim),
            nn.ReLU(),
        )
        
        # CNN features as keys and values
        self.cnn_to_key = nn.Linear(cnn_dim, self.hidden_dim)
        self.cnn_to_value = nn.Linear(cnn_dim, self.hidden_dim)
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, output_dim),  # attention + residual TDA
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.tda_residual = nn.Linear(tda_dim, self.hidden_dim)
        self.scale = self.hidden_dim ** -0.5
    
    def forward(self, cnn_features, tda_features):
        # TDA as query
        query = self.tda_to_query(tda_features)  # (B, hidden)
        
        # CNN as key/value
        key = self.cnn_to_key(cnn_features)  # (B, hidden)
        value = self.cnn_to_value(cnn_features)  # (B, hidden)
        
        # Attention (simplified - treating as single "token")
        # In practice, could treat CNN feature channels as sequence
        attention = torch.sigmoid(torch.sum(query * key, dim=1, keepdim=True) * self.scale)
        
        # Apply attention
        attended = attention * value  # (B, hidden)
        
        # Combine with TDA residual
        tda_res = self.tda_residual(tda_features)
        combined = torch.cat([attended, tda_res], dim=1)
        
        return self.output(combined)


class ResidualFusion(BaseFusion):
    """
    Strategy 6: Residual Fusion
    
    Treat TDA as a residual correction to CNN features.
    CNN provides main signal, TDA refines it.
    
    Key insight: TDA adds topological context to appearance features.
    Residual learning makes optimization easier.
    """
    def __init__(self, cnn_dim: int, tda_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        # Main path: CNN
        self.cnn_main = nn.Sequential(
            nn.Linear(cnn_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
        # Residual path: TDA -> correction
        self.tda_residual = nn.Sequential(
            nn.Linear(tda_dim, output_dim // 2),
            nn.BatchNorm1d(output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, output_dim),
            nn.Tanh(),  # Bounded residual
        )
        
        # Learnable residual scale
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
        self.output = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, cnn_features, tda_features):
        main = self.cnn_main(cnn_features)
        residual = self.tda_residual(tda_features) * self.residual_scale
        
        fused = main + residual
        return self.output(fused)


class FilmFusion(BaseFusion):
    """
    Strategy 7: FiLM (Feature-wise Linear Modulation)
    
    TDA modulates CNN features through learned affine transformations.
    Inspired by FiLM layers used in visual reasoning.
    
    Key insight: TDA provides context that modulates how CNN features
    should be interpreted.
    
    Formula: output = gamma(TDA) * CNN + beta(TDA)
    """
    def __init__(self, cnn_dim: int, tda_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        # CNN projection
        self.cnn_proj = nn.Linear(cnn_dim, output_dim)
        
        # TDA generates modulation parameters
        self.tda_to_gamma = nn.Sequential(
            nn.Linear(tda_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),
        )
        self.tda_to_beta = nn.Sequential(
            nn.Linear(tda_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),
        )
        
        # Initialize gamma close to 1, beta close to 0
        nn.init.ones_(self.tda_to_gamma[-1].weight.data * 0.01)
        nn.init.zeros_(self.tda_to_gamma[-1].bias.data)
        self.tda_to_gamma[-1].bias.data.fill_(1.0)  # gamma = 1 initially
        nn.init.zeros_(self.tda_to_beta[-1].weight.data)
        nn.init.zeros_(self.tda_to_beta[-1].bias.data)
        
        self.output = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, cnn_features, tda_features):
        cnn_proj = self.cnn_proj(cnn_features)  # (B, output_dim)
        
        # TDA generates modulation
        gamma = self.tda_to_gamma(tda_features)  # (B, output_dim)
        beta = self.tda_to_beta(tda_features)    # (B, output_dim)
        
        # FiLM modulation
        modulated = gamma * cnn_proj + beta
        
        return self.output(modulated)


# ============================================================================
# UNIFIED MODEL with Swappable Fusion
# ============================================================================

class IntegrationResearchModel(nn.Module):
    """
    Face verification model with configurable TDA fusion strategy.
    """
    
    FUSION_STRATEGIES = {
        'concat': ConcatFusion,
        'attention': AttentionFusion,
        'gated': GatedFusion,
        'bilinear': BilinearFusion,
        'cross_modal': CrossModalAttention,
        'residual': ResidualFusion,
        'film': FilmFusion,
    }
    
    def __init__(
        self, 
        fusion_strategy: str = 'concat',
        embedding_dim: int = 256,
        tda_dim: int = 400,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.fusion_strategy_name = fusion_strategy
        
        # CNN Backbone (same as before)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        cnn_dim = 256
        fusion_output_dim = 384  # Consistent output for fair comparison
        
        # Fusion module (swappable)
        fusion_cls = self.FUSION_STRATEGIES[fusion_strategy]
        self.fusion = fusion_cls(cnn_dim, tda_dim, fusion_output_dim, dropout)
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(fusion_output_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
    
    def forward(self, x, tda_features):
        # CNN stream
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x).flatten(1)  # (B, 256)
        
        # Fusion
        fused = self.fusion(x, tda_features)  # (B, 384)
        
        # Embedding
        emb = self.embedding(fused)
        return F.normalize(emb, p=2, dim=1)


# ============================================================================
# DATASET
# ============================================================================

class TripletDataset(Dataset):
    """Triplet dataset with precomputed TDA features."""
    
    def __init__(self, data_dir: Path, tda_cache_path: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load TDA cache
        print(f"Loading TDA cache from {tda_cache_path}...")
        if tda_cache_path.exists():
            data = np.load(tda_cache_path)
            self.tda_features = {k: data[k] for k in data.files}
        else:
            self.tda_features = {}
        print(f"Loaded {len(self.tda_features)} TDA features")
        
        # Build identity mapping
        self.identity_to_images = {}
        self.all_images = []
        
        for identity_dir in sorted(self.data_dir.iterdir()):
            if identity_dir.is_dir():
                identity = identity_dir.name
                images = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    images.extend(identity_dir.glob(ext))
                
                # Only use images with TDA features
                if self.tda_features:
                    valid_images = [p for p in images if str(p) in self.tda_features]
                else:
                    valid_images = images
                
                if len(valid_images) >= 2:
                    self.identity_to_images[identity] = valid_images
                    for img_path in valid_images:
                        self.all_images.append((img_path, identity))
        
        self.identities = list(self.identity_to_images.keys())
        print(f"Loaded {len(self.all_images)} images from {len(self.identities)} identities")
    
    def __len__(self):
        return len(self.all_images)
    
    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    
    def _get_tda(self, path: Path) -> torch.Tensor:
        if self.tda_features:
            return torch.tensor(self.tda_features[str(path)], dtype=torch.float32)
        # Generate random TDA features for testing
        return torch.randn(400)
    
    def __getitem__(self, idx):
        anchor_path, anchor_id = self.all_images[idx]
        
        pos_candidates = [p for p in self.identity_to_images[anchor_id] if p != anchor_path]
        pos_path = np.random.choice(pos_candidates)
        
        neg_id = np.random.choice([i for i in self.identities if i != anchor_id])
        neg_path = np.random.choice(self.identity_to_images[neg_id])
        
        return (
            (self._load_image(anchor_path), self._get_tda(anchor_path)),
            (self._load_image(pos_path), self._get_tda(pos_path)),
            (self._load_image(neg_path), self._get_tda(neg_path)),
        )


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Train", leave=False):
        (anchor_img, anchor_tda), (pos_img, pos_tda), (neg_img, neg_tda) = batch
        
        anchor_img, anchor_tda = anchor_img.to(device), anchor_tda.to(device)
        pos_img, pos_tda = pos_img.to(device), pos_tda.to(device)
        neg_img, neg_tda = neg_img.to(device), neg_tda.to(device)
        
        optimizer.zero_grad()
        
        anchor_emb = model(anchor_img, anchor_tda)
        pos_emb = model(pos_img, pos_tda)
        neg_emb = model(neg_img, neg_tda)
        
        loss = criterion(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            pos_dist = F.pairwise_distance(anchor_emb, pos_emb)
            neg_dist = F.pairwise_distance(anchor_emb, neg_emb)
            correct += (pos_dist < neg_dist).sum().item()
            total += anchor_img.size(0)
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Val", leave=False):
        (anchor_img, anchor_tda), (pos_img, pos_tda), (neg_img, neg_tda) = batch
        
        anchor_img, anchor_tda = anchor_img.to(device), anchor_tda.to(device)
        pos_img, pos_tda = pos_img.to(device), pos_tda.to(device)
        neg_img, neg_tda = neg_img.to(device), neg_tda.to(device)
        
        anchor_emb = model(anchor_img, anchor_tda)
        pos_emb = model(pos_img, pos_tda)
        neg_emb = model(neg_img, neg_tda)
        
        loss = criterion(anchor_emb, pos_emb, neg_emb)
        total_loss += loss.item()
        
        pos_dist = F.pairwise_distance(anchor_emb, pos_emb)
        neg_dist = F.pairwise_distance(anchor_emb, neg_emb)
        correct += (pos_dist < neg_dist).sum().item()
        total += anchor_img.size(0)
    
    return total_loss / len(loader), correct / total


def run_strategy_experiment(
    strategy: str,
    epochs: int = 10,
    batch_size: int = 64,
    subset_size: int = 10000,
    lr: float = 1e-3,
):
    """Run experiment with a specific fusion strategy."""
    
    print(f"\n{'='*70}")
    print(f"STRATEGY: {strategy.upper()}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dir = PROJECT_ROOT / "dataset" / "classification_data" / "train_data"
    val_dir = PROJECT_ROOT / "dataset" / "classification_data" / "val_data"
    tda_cache = PROJECT_ROOT / "outputs" / "tda_cache" / "tda_features.npz"
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Datasets
    train_dataset = TripletDataset(train_dir, tda_cache, train_transform)
    val_dataset = TripletDataset(val_dir, tda_cache, val_transform)
    
    # Subset for faster experiments
    train_indices = np.random.choice(len(train_dataset), min(subset_size, len(train_dataset)), replace=False)
    val_indices = np.random.choice(len(val_dataset), min(subset_size // 5, len(val_dataset)), replace=False)
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = IntegrationResearchModel(
        fusion_strategy=strategy,
        embedding_dim=256,
        tda_dim=400,
        dropout=0.3,
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    print(f"Total params: {params:,}, Fusion params: {fusion_params:,}")
    
    # Training
    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=len(train_loader) * epochs, pct_start=0.1)
    
    history = []
    best_val_acc = 0
    
    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - start
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            marker = " ★"
        else:
            marker = ""
        
        gap = train_acc - val_acc
        print(f"  Epoch {epoch+1:2d}: Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}%, "
              f"Gap={gap*100:+.1f}%, Time={elapsed:.0f}s{marker}")
    
    return {
        'strategy': strategy,
        'best_val_acc': best_val_acc,
        'final_train_acc': history[-1]['train_acc'],
        'final_val_acc': history[-1]['val_acc'],
        'final_gap': history[-1]['train_acc'] - history[-1]['val_acc'],
        'fusion_params': fusion_params,
        'total_params': params,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description="TDA Integration Research")
    parser.add_argument('--strategy', type=str, default='all',
                        help="Fusion strategy: concat, attention, gated, bilinear, cross_modal, residual, film, or 'all'")
    parser.add_argument('--epochs', type=int, default=10, help="Training epochs")
    parser.add_argument('--subset', type=int, default=10000, help="Subset size")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    strategies = list(IntegrationResearchModel.FUSION_STRATEGIES.keys())
    
    if args.strategy == 'all':
        test_strategies = strategies
    elif args.strategy in strategies:
        test_strategies = [args.strategy]
    else:
        print(f"Unknown strategy: {args.strategy}")
        print(f"Available: {strategies}")
        sys.exit(1)
    
    print("=" * 70)
    print("TDA INTEGRATION RESEARCH")
    print("=" * 70)
    print(f"Strategies to test: {test_strategies}")
    print(f"Epochs: {args.epochs}, Subset: {args.subset}")
    
    results = []
    for strategy in test_strategies:
        result = run_strategy_experiment(
            strategy=strategy,
            epochs=args.epochs,
            subset_size=args.subset,
            batch_size=args.batch_size,
        )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESEARCH SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<15} {'Best Val':>10} {'Final Gap':>10} {'Fusion Params':>15}")
    print("-" * 70)
    
    results.sort(key=lambda x: -x['best_val_acc'])
    for r in results:
        print(f"{r['strategy']:<15} {r['best_val_acc']*100:>9.2f}% {r['final_gap']*100:>+9.2f}% {r['fusion_params']:>15,}")
    
    # Save
    output_dir = PROJECT_ROOT / "outputs" / "tda_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "integration_research_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    if results:
        winner = results[0]
        print(f"\n🏆 BEST STRATEGY: {winner['strategy']} with {winner['best_val_acc']*100:.2f}% validation accuracy")


if __name__ == "__main__":
    main()
