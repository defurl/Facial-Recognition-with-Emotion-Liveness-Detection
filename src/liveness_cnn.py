"""
CNN-Based Liveness Detection using FaceEmbeddingCNN Backbone
Binary classification: Real (live face) vs Spoof (photo/screen)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import FaceEmbeddingCNN, CBAM
import numpy as np
from collections import deque
import time


class LivenessCNN(nn.Module):
    """
    Liveness detector using pre-trained FaceEmbeddingCNN backbone.
    
    Architecture:
    - Shared backbone from FaceEmbeddingCNN (frozen or fine-tuned)
    - Specialized liveness head for binary classification
    - Optional temporal consistency module for video
    """
    
    def __init__(self, backbone_weights=None, freeze_backbone=False):
        super(LivenessCNN, self).__init__()
        
        # Load pre-trained face recognition backbone
        base_model = FaceEmbeddingCNN(embedding_dim=256, num_classes=4000, use_cbam=True)
        
        if backbone_weights is not None:
            print(f"[LivenessCNN] Loading backbone weights from {backbone_weights}")
            checkpoint = torch.load(backbone_weights, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load weights (ignore classifier head)
            base_model.load_state_dict(state_dict, strict=False)
            print("[LivenessCNN] Backbone loaded successfully")
        
        # Extract backbone (convolutional layers + CBAM)
        self.backbone = base_model.backbone
        self.gap = base_model.gap
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[LivenessCNN] Backbone frozen")
        
        # Liveness-specific head
        # Takes 256-dim features and predicts Real vs Spoof
        self.liveness_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(64, 2)  # Binary: [spoof_score, real_score]
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor (B, 3, H, W)
        
        Returns:
            logits: (B, 2) logits for [spoof, real]
        """
        # Extract features using backbone
        features = self.backbone(x)
        features = self.gap(features)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Liveness classification
        logits = self.liveness_head(features)
        
        return logits
    
    def predict(self, x):
        """
        Predict liveness with confidence
        
        Args:
            x: Input image tensor (B, 3, H, W)
        
        Returns:
            is_live: Boolean tensor (B,)
            confidence: Confidence scores (B,)
            probabilities: (B, 2) probabilities for [spoof, real]
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        
        # Real class is index 1
        real_prob = probabilities[:, 1]
        is_live = real_prob > 0.5
        confidence = torch.max(probabilities, dim=1)[0]
        
        return is_live, confidence, probabilities


class TemporalLivenessDetector:
    """
    Wrapper for LivenessCNN with temporal consistency for video streams.
    Smooths predictions over time to reduce flickering.
    """
    
    def __init__(self, model_path, device='cpu', history_size=10, confidence_threshold=0.65):
        """
        Initialize temporal liveness detector
        
        Args:
            model_path: Path to trained LivenessCNN weights (the full trained model)
            device: Device to run model on
            history_size: Number of frames to smooth over
            confidence_threshold: Minimum confidence to change state
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load trained model
        # First create model architecture (without loading backbone weights separately)
        self.model = LivenessCNN(backbone_weights=None, freeze_backbone=False)
        
        # Then load the fully trained weights (including liveness head)
        print(f"[TemporalLiveness] Loading trained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model state dict (handle checkpoint format from training)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"[TemporalLiveness] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, "
                  f"val_acc: {checkpoint.get('val_acc', 0):.2f}%")
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        print(f"[TemporalLiveness] Model loaded successfully")
        
        # Temporal smoothing
        self.history_size = history_size
        self.prediction_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        
        # State tracking
        self.current_state = None  # 'Real' or 'Spoof'
        self.state_confidence = 0.0
        self.frames_in_state = 0
        
        print(f"[TemporalLiveness] Initialized with history={history_size}, threshold={confidence_threshold}")
    
    def analyze(self, face_image):
        """
        Analyze single frame with temporal smoothing
        
        Args:
            face_image: RGB face image (numpy array, H x W x 3)
        
        Returns:
            is_live: bool
            confidence: float (0-1)
            details: dict with diagnosis info
        """
        # Preprocess image
        from torchvision import transforms
        
        # Same transforms as training
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(face_image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            is_live_tensor, conf_tensor, probs = self.model.predict(img_tensor)
        
        is_live = is_live_tensor.item()
        confidence = conf_tensor.item()
        spoof_prob = probs[0, 0].item()
        real_prob = probs[0, 1].item()
        
        # Add to history
        self.prediction_history.append(is_live)
        self.confidence_history.append(confidence)
        
        # Temporal smoothing: majority vote + confidence check
        if len(self.prediction_history) >= self.history_size // 2:
            # Count votes
            real_votes = sum(self.prediction_history)
            total_votes = len(self.prediction_history)
            real_ratio = real_votes / total_votes
            
            # Average confidence
            avg_confidence = np.mean(list(self.confidence_history))
            
            # Determine stable state
            # Require strong majority (70%) and good confidence
            if real_ratio >= 0.7 and avg_confidence >= self.confidence_threshold:
                stable_state = True
                stable_is_live = True
            elif real_ratio <= 0.3 and avg_confidence >= self.confidence_threshold:
                stable_state = True
                stable_is_live = False
            else:
                # Uncertain - keep current state or default to spoof (safer)
                stable_state = False
                stable_is_live = self.current_state == 'Real' if self.current_state else False
            
            # Update state
            new_state = 'Real' if stable_is_live else 'Spoof'
            if new_state == self.current_state:
                self.frames_in_state += 1
            else:
                self.current_state = new_state
                self.frames_in_state = 1
            
            self.state_confidence = avg_confidence
            
            # Return stable state
            final_is_live = (self.current_state == 'Real')
            final_confidence = self.state_confidence
        else:
            # Not enough history - use instantaneous with lower confidence
            final_is_live = is_live
            final_confidence = confidence * 0.8  # Penalize for insufficient history
        
        # Build details
        details = {
            'spoof_prob': spoof_prob,
            'real_prob': real_prob,
            'instantaneous_decision': 'Real' if is_live else 'Spoof',
            'instantaneous_confidence': confidence,
            'smoothed_decision': self.current_state or 'Unknown',
            'smoothed_confidence': self.state_confidence,
            'frames_in_state': self.frames_in_state,
            'history_size': len(self.prediction_history),
            'real_vote_ratio': real_ratio if len(self.prediction_history) >= self.history_size // 2 else 0.5
        }
        
        return final_is_live, final_confidence, details
    
    def reset(self):
        """Reset temporal history"""
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.current_state = None
        self.state_confidence = 0.0
        self.frames_in_state = 0


# Global detector instance
_cnn_liveness_detector = None

def get_cnn_liveness_detector(model_path='best_face_embedding_model.pth', device='cpu'):
    """Get or create global CNN liveness detector"""
    global _cnn_liveness_detector
    if _cnn_liveness_detector is None:
        _cnn_liveness_detector = TemporalLivenessDetector(
            model_path=model_path,
            device=device,
            history_size=10,
            confidence_threshold=0.65
        )
    return _cnn_liveness_detector


def detect_liveness_cnn(face_image, model_path='best_face_embedding_model.pth', device='cpu'):
    """
    Convenience function for CNN-based liveness detection
    
    Args:
        face_image: RGB face image (numpy array)
        model_path: Path to backbone weights
        device: Device to run on
    
    Returns:
        is_live: bool
        confidence: float (0-1)
        details: dict
    """
    detector = get_cnn_liveness_detector(model_path, device)
    return detector.analyze(face_image)


__all__ = ['LivenessCNN', 'TemporalLivenessDetector', 'get_cnn_liveness_detector', 'detect_liveness_cnn']
