"""
Emotion Detection Model Architecture
Uses CBAM backbone with transfer learning from face embedding model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import CBAM modules from models.py
from models import ChannelAttention, SpatialAttention, CBAM


class EmotionCNN(nn.Module):
    """
    Custom CNN for Emotion Detection.
    Reuses CBAM backbone architecture, supports transfer learning from face embedding model.
    
    Args:
        num_classes: Number of emotion classes (default: 7 for RAF-DB)
        use_cbam: Whether to use CBAM attention modules
        pretrained_backbone: Path to pretrained face embedding model for transfer learning
    """
    
    def __init__(self, num_classes=7, use_cbam=True, pretrained_backbone=None):
        super(EmotionCNN, self).__init__()
        self.use_cbam = use_cbam
        self.num_classes = num_classes
        
        # Build backbone (same as FaceEmbeddingCNN)
        blocks = []
        
        # Block 1: 3 -> 32 channels
        blocks += [
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        if use_cbam:
            blocks.append(CBAM(32))
        blocks.append(nn.MaxPool2d(2, 2))
        
        # Block 2: 32 -> 64 channels
        blocks += [
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        if use_cbam:
            blocks.append(CBAM(64))
        blocks.append(nn.MaxPool2d(2, 2))
        
        # Block 3: 64 -> 128 channels
        blocks += [
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        if use_cbam:
            blocks.append(CBAM(128))
        blocks.append(nn.MaxPool2d(2, 2))
        
        # Block 4: 128 -> 256 channels
        blocks += [
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        if use_cbam:
            blocks.append(CBAM(256))
        blocks.append(nn.MaxPool2d(2, 2))
        
        self.backbone = nn.Sequential(*blocks)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Emotion classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Load pretrained weights if provided
        if pretrained_backbone is not None:
            self._load_pretrained_backbone(pretrained_backbone)
    
    def _load_pretrained_backbone(self, pretrained_path):
        """
        Load backbone weights from pretrained face embedding model.
        Only loads the backbone layers, not the embedding/classifier heads.
        """
        print(f"Loading pretrained backbone from: {pretrained_path}")
        
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Filter only backbone weights
            backbone_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('backbone.'):
                    backbone_state_dict[key] = value
            
            # Load with strict=False to handle any mismatches
            missing, unexpected = self.load_state_dict(backbone_state_dict, strict=False)
            
            print(f"  Loaded {len(backbone_state_dict)} backbone layers")
            if missing:
                print(f"  Missing keys (expected): {len(missing)} (classifier head)")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
                
            print("  ✓ Transfer learning applied successfully!")
            
        except Exception as e:
            print(f"  ⚠ Could not load pretrained weights: {e}")
            print("  Training from scratch...")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, 3, 64, 64)
            
        Returns:
            Logits for emotion classes (batch_size, num_classes)
        """
        # Backbone feature extraction
        x = self.backbone(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 256)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def predict_emotion(self, x):
        """
        Predict emotion with probabilities.
        
        Args:
            x: Input tensor (batch_size, 3, 64, 64)
            
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)
        return pred_class, probs


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_backbone(model, freeze=True):
    """
    Freeze/unfreeze backbone layers for fine-tuning.
    
    Args:
        model: EmotionCNN model
        freeze: If True, freeze backbone; if False, unfreeze
    """
    for param in model.backbone.parameters():
        param.requires_grad = not freeze
    
    status = "frozen" if freeze else "unfrozen"
    print(f"Backbone layers {status}")


if __name__ == "__main__":
    print("Testing EmotionCNN...")
    
    # Test without pretrained
    model = EmotionCNN(num_classes=7, use_cbam=True)
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 64, 64)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test predict
    pred_class, probs = model.predict_emotion(dummy_input)
    print(f"Predicted classes: {pred_class}")
    print(f"Probabilities shape: {probs.shape}")
    
    print("\n✓ EmotionCNN test passed!")
