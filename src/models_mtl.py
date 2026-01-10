"""
Unified Multi-Task Learning Model for Face Recognition, Emotion, and Liveness Detection.

This module extends the FaceEmbeddingCNN with task-specific heads for:
1. Face Verification (256-dim embeddings with Triplet/ArcFace loss)
2. Emotion Recognition (7 classes: angry, disgust, fear, happy, neutral, sad, surprise)
3. Liveness Detection (binary: live vs spoof)

Architecture:
    Input (B, 3, 64, 64)
         ↓
    Shared Backbone (CBAM-enhanced CNN) → 256-dim features
         ↓
    ┌────────────────┬────────────────┬────────────────┐
    │ Face Embed Head│ Emotion Head   │ Liveness Head  │
    │ Linear→BN→L2   │ FC→ReLU→Drop→FC│ FC→ReLU→Drop→FC│
    └───────┬────────┴───────┬────────┴───────┬────────┘
            ↓                ↓                ↓
       256-dim emb      7-class logits   2-class logits

Author: MTL Implementation
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# Import CBAM from existing models
from models import CBAM, ChannelAttention, SpatialAttention, count_parameters


# ============= Emotion Labels =============
EMOTION_LABELS = {
    0: 'angry',
    1: 'disgust', 
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}
NUM_EMOTIONS = 7

# ============= Liveness Labels =============
LIVENESS_LABELS = {
    0: 'live',
    1: 'spoof'
}
NUM_LIVENESS_CLASSES = 2


class TaskHead(nn.Module):
    """
    Generic task-specific head with configurable architecture.
    
    Used for emotion and liveness classification tasks.
    """
    def __init__(
        self, 
        in_features: int, 
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        use_bn: bool = True
    ):
        super(TaskHead, self).__init__()
        
        layers = [nn.Linear(in_features, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        ])
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin) head for improved face embeddings.
    
    Implements the formula: cos(θ + m) where m is the angular margin.
    This creates a tighter clustering of same-identity embeddings.
    
    Reference: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,  # num_classes
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Weight matrix (normalized during forward)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cos/sin of margin
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.th = torch.cos(torch.tensor(torch.pi - margin))
        self.mm = torch.sin(torch.tensor(torch.pi - margin)) * margin
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: L2-normalized face embeddings (B, in_features)
            labels: Ground truth labels for training (B,). If None, returns cos similarity.
        
        Returns:
            Scaled logits for training, or cosine similarity for inference
        """
        # Normalize weight
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity: cos(θ)
        cosine = F.linear(embeddings, weight_norm)
        
        if labels is None:
            # Inference mode: return scaled cosine
            return cosine * self.scale
        
        # Training mode: apply angular margin
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        cos_m = self.cos_m.to(cosine.device)
        sin_m = self.sin_m.to(cosine.device)
        phi = cosine * cos_m - sine * sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            th = self.th.to(cosine.device)
            mm = self.mm.to(cosine.device)
            phi = torch.where(cosine > th, phi, cosine - mm)
        
        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output


class UnifiedFaceModel(nn.Module):
    """
    Unified Multi-Task Learning model for Face Recognition, Emotion, and Liveness.
    
    Shared backbone with task-specific heads:
    - Face embedding head (for verification via triplet/ArcFace loss)
    - Emotion classification head (7 emotions)
    - Liveness classification head (live/spoof)
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        num_identities: int = 4000,
        num_emotions: int = NUM_EMOTIONS,
        num_liveness: int = NUM_LIVENESS_CLASSES,
        use_cbam: bool = True,
        use_arcface: bool = False,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.5,
        emotion_dropout: float = 0.3,
        liveness_dropout: float = 0.5,
    ):
        super(UnifiedFaceModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.use_arcface = use_arcface
        
        # ============= Shared Backbone =============
        # Same architecture as FaceEmbeddingCNN
        blocks = []
        
        # Block 1: 3 → 32 channels
        blocks += [
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        if use_cbam:
            blocks.append(CBAM(32))
        blocks.append(nn.MaxPool2d(2, 2))
        
        # Block 2: 32 → 64 channels
        blocks += [
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        if use_cbam:
            blocks.append(CBAM(64))
        blocks.append(nn.MaxPool2d(2, 2))
        
        # Block 3: 64 → 128 channels
        blocks += [
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        if use_cbam:
            blocks.append(CBAM(128))
        blocks.append(nn.MaxPool2d(2, 2))
        
        # Block 4: 128 → 256 channels
        blocks += [
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        if use_cbam:
            blocks.append(CBAM(256))
        blocks.append(nn.MaxPool2d(2, 2))
        
        self.backbone = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Backbone output: 256 features
        backbone_out_dim = 256
        
        # ============= Task-Specific Heads =============
        
        # 1. Face Embedding Head (for verification)
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        # Optional: ArcFace classification head (for training with identity labels)
        if use_arcface:
            self.arcface_head = ArcFaceHead(
                in_features=embedding_dim,
                out_features=num_identities,
                scale=arcface_scale,
                margin=arcface_margin
            )
        else:
            # Standard classification head
            self.identity_head = nn.Linear(backbone_out_dim, num_identities)
        
        # 2. Emotion Classification Head
        self.emotion_head = TaskHead(
            in_features=backbone_out_dim,
            num_classes=num_emotions,
            hidden_dim=128,
            dropout=emotion_dropout,
            use_bn=True
        )
        
        # 3. Liveness Detection Head
        self.liveness_head = TaskHead(
            in_features=backbone_out_dim,
            num_classes=num_liveness,
            hidden_dim=128,
            dropout=liveness_dropout,
            use_bn=True
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract backbone features from input images.
        
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            Feature vectors (B, 256)
        """
        x = self.backbone(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get L2-normalized face embeddings.
        
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            L2-normalized embeddings (B, embedding_dim)
        """
        features = self.extract_features(x)
        embedding = self.embedding_head(features)
        return F.normalize(embedding, p=2, dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        tasks: Tuple[str, ...] = ('embedding', 'emotion', 'liveness'),
        identity_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-task forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            tasks: Tuple of task names to compute ('embedding', 'emotion', 'liveness', 'identity')
            identity_labels: Labels for ArcFace training (only used if use_arcface=True)
        
        Returns:
            Dictionary with outputs for each requested task:
                - 'embedding': L2-normalized face embeddings (B, embedding_dim)
                - 'emotion': Emotion logits (B, 7)
                - 'liveness': Liveness logits (B, 2)
                - 'identity': Identity logits (B, num_identities) [if requested]
                - 'features': Raw backbone features (B, 256) [always included]
        """
        # Shared feature extraction
        features = self.extract_features(x)
        
        outputs = {'features': features}
        
        if 'embedding' in tasks:
            embedding = self.embedding_head(features)
            outputs['embedding'] = F.normalize(embedding, p=2, dim=1)
        
        if 'identity' in tasks:
            if self.use_arcface:
                # ArcFace needs normalized embeddings
                if 'embedding' not in outputs:
                    embedding = self.embedding_head(features)
                    embedding = F.normalize(embedding, p=2, dim=1)
                else:
                    embedding = outputs['embedding']
                outputs['identity'] = self.arcface_head(embedding, identity_labels)
            else:
                outputs['identity'] = self.identity_head(features)
        
        if 'emotion' in tasks:
            outputs['emotion'] = self.emotion_head(features)
        
        if 'liveness' in tasks:
            outputs['liveness'] = self.liveness_head(features)
        
        return outputs
    
    def freeze_backbone(self):
        """Freeze backbone parameters for head-only training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.gap.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.gap.parameters():
            param.requires_grad = True
    
    def get_task_parameters(self, task: str):
        """Get parameters for a specific task head."""
        if task == 'embedding':
            return self.embedding_head.parameters()
        elif task == 'identity':
            if self.use_arcface:
                return self.arcface_head.parameters()
            else:
                return self.identity_head.parameters()
        elif task == 'emotion':
            return self.emotion_head.parameters()
        elif task == 'liveness':
            return self.liveness_head.parameters()
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def load_pretrained_backbone(self, state_dict: dict, strict: bool = False):
        """
        Load pretrained weights from FaceEmbeddingCNN.
        
        Args:
            state_dict: State dict from FaceEmbeddingCNN
            strict: Whether to require exact match
        """
        # Map keys from FaceEmbeddingCNN to UnifiedFaceModel
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_state_dict[key] = value
            elif key.startswith('embedding_head.'):
                new_state_dict[key] = value
        
        # Load with partial matching
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        
        if not strict:
            print(f"Loaded pretrained backbone. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        return missing, unexpected


def create_unified_model(
    pretrained_path: Optional[str] = None,
    use_arcface: bool = False,
    **kwargs
) -> UnifiedFaceModel:
    """
    Factory function to create UnifiedFaceModel with optional pretrained weights.
    
    Args:
        pretrained_path: Path to pretrained FaceEmbeddingCNN weights
        use_arcface: Whether to use ArcFace head
        **kwargs: Additional arguments for UnifiedFaceModel
    
    Returns:
        Initialized UnifiedFaceModel
    """
    model = UnifiedFaceModel(use_arcface=use_arcface, **kwargs)
    
    if pretrained_path:
        import os
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_pretrained_backbone(state_dict)
        else:
            print(f"Warning: Pretrained path not found: {pretrained_path}")
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Testing UnifiedFaceModel")
    print("=" * 60)
    
    # Create model
    model = UnifiedFaceModel(
        embedding_dim=256,
        num_identities=4000,
        use_cbam=True,
        use_arcface=False
    )
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 64, 64)
    
    print("\n--- Full Forward Pass ---")
    outputs = model(dummy_input, tasks=('embedding', 'emotion', 'liveness', 'identity'))
    
    for task, tensor in outputs.items():
        print(f"  {task}: {tensor.shape}")
    
    # Test individual task forwards
    print("\n--- Individual Task Tests ---")
    
    # Embedding only
    emb = model.get_embedding(dummy_input)
    print(f"  Embedding only: {emb.shape}, norm: {emb.norm(dim=1).mean():.4f}")
    
    # Emotion only
    outputs_emotion = model(dummy_input, tasks=('emotion',))
    print(f"  Emotion only: {outputs_emotion['emotion'].shape}")
    
    # Liveness only
    outputs_live = model(dummy_input, tasks=('liveness',))
    print(f"  Liveness only: {outputs_live['liveness'].shape}")
    
    # Test freeze/unfreeze
    print("\n--- Freeze/Unfreeze Test ---")
    model.freeze_backbone()
    trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable after freeze: {trainable_frozen:,}")
    
    model.unfreeze_backbone()
    trainable_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable after unfreeze: {trainable_unfrozen:,}")
    
    print("\n✓ All tests passed!")
