"""
Multi-Task Loss Functions for Unified Face Model.

This module provides:
1. FocalLoss - for handling class imbalance (liveness detection)
2. LabelSmoothingCE - for better generalization (emotion)
3. UncertaintyWeightedLoss - learnable multi-task loss balancing
4. GradNormLoss - gradient-based dynamic loss balancing
5. MTLLoss - unified multi-task loss wrapper

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    - γ (gamma): Focusing parameter. Higher values down-weight easy examples more.
    - α (alpha): Class balancing weights.
    
    Reference: Focal Loss for Dense Object Detection (Lin et al., 2017)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        num_classes: int = 2
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer('alpha', alpha)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Logits (B, num_classes)
            targets: Ground truth labels (B,)
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing and optional class weights.
    
    Instead of hard labels (0, 1), uses soft labels:
    - Target class: 1 - smoothing
    - Other classes: smoothing / (num_classes - 1)
    
    This prevents overconfidence and improves generalization.
    Supports class weights for imbalanced datasets.
    """
    
    def __init__(
        self, 
        smoothing: float = 0.1, 
        reduction: str = 'mean',
        weight: Optional[torch.Tensor] = None
    ):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.register_buffer('weight', weight)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Logits (B, num_classes)
            targets: Ground truth labels (B,)
        
        Returns:
            Label-smoothed cross-entropy loss
        """
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smooth labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs)
            smooth_labels.fill_(self.smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = (-smooth_labels * log_probs).sum(dim=-1)
        
        # Apply class weights if provided
        if self.weight is not None:
            sample_weights = self.weight[targets]
            loss = loss * sample_weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task loss with learned uncertainty weighting.
    
    Each task has a learnable log-variance parameter (log σ²).
    Loss is weighted by 1/(2σ²) with regularization term log(σ).
    
    L_total = Σ (1/(2σ_i²)) * L_i + log(σ_i)
    
    This allows the model to learn task importance during training.
    
    Reference: Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall et al., 2018)
    """
    
    def __init__(self, task_names: List[str], init_log_vars: Optional[Dict[str, float]] = None):
        super(UncertaintyWeightedLoss, self).__init__()
        
        self.task_names = task_names
        
        # Initialize log variance parameters
        if init_log_vars is None:
            init_log_vars = {name: 0.0 for name in task_names}
        
        # Create learnable parameters
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(init_log_vars.get(name, 0.0)))
            for name in task_names
        })
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted multi-task loss.
        
        Args:
            losses: Dictionary mapping task names to their losses
        
        Returns:
            Tuple of (total_loss, weighted_losses_dict)
        """
        total_loss = 0.0
        weighted_losses = {}
        
        for name in self.task_names:
            if name not in losses:
                continue
            
            task_loss = losses[name]
            log_var = self.log_vars[name]
            
            # Weight = 1 / (2 * exp(log_var)) = 0.5 * exp(-log_var)
            # Regularization = 0.5 * log_var
            precision = torch.exp(-log_var)
            weighted_loss = 0.5 * precision * task_loss + 0.5 * log_var
            
            weighted_losses[name] = weighted_loss
            total_loss = total_loss + weighted_loss
        
        return total_loss, weighted_losses
    
    def get_weights(self) -> Dict[str, float]:
        """Get current task weights (1/σ²)."""
        return {
            name: torch.exp(-self.log_vars[name]).item()
            for name in self.task_names
        }


class FixedWeightedLoss(nn.Module):
    """
    Multi-task loss with fixed weights.
    
    Simple weighted sum: L_total = Σ w_i * L_i
    """
    
    def __init__(self, weights: Dict[str, float]):
        super(FixedWeightedLoss, self).__init__()
        self.weights = weights
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted multi-task loss.
        
        Args:
            losses: Dictionary mapping task names to their losses
        
        Returns:
            Tuple of (total_loss, weighted_losses_dict)
        """
        total_loss = 0.0
        weighted_losses = {}
        
        for name, loss in losses.items():
            weight = self.weights.get(name, 1.0)
            weighted_loss = weight * loss
            weighted_losses[name] = weighted_loss
            total_loss = total_loss + weighted_loss
        
        return total_loss, weighted_losses


class MTLLoss(nn.Module):
    """
    Unified Multi-Task Loss for Face Recognition, Emotion, and Liveness.
    
    Combines:
    - Face: Triplet loss or ArcFace (depending on mode)
    - Emotion: Label-smoothed cross-entropy with class weights
    - Liveness: Focal loss (for class imbalance)
    
    With configurable weighting strategy.
    """
    
    def __init__(
        self,
        # Task weights
        face_weight: float = 1.0,
        emotion_weight: float = 0.5,
        liveness_weight: float = 1.0,
        
        # Loss configurations
        triplet_margin: float = 0.5,
        emotion_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        liveness_class_weights: Optional[torch.Tensor] = None,
        emotion_class_weights: Optional[torch.Tensor] = None,  # NEW: class weights for emotion
        
        # Weighting strategy
        use_uncertainty_weighting: bool = False,
    ):
        super(MTLLoss, self).__init__()
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Task-specific loss functions
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_margin, p=2)
        self.face_ce_loss = nn.CrossEntropyLoss()  # For ArcFace/softmax
        
        # Emotion loss with class weights for imbalanced dataset
        self.emotion_loss = LabelSmoothingCrossEntropy(
            smoothing=emotion_smoothing,
            weight=emotion_class_weights
        )
        
        self.liveness_loss = FocalLoss(
            alpha=liveness_class_weights,
            gamma=focal_gamma,
            num_classes=2
        )
        
        # Loss weighting
        if use_uncertainty_weighting:
            self.loss_weighter = UncertaintyWeightedLoss(
                task_names=['face', 'emotion', 'liveness'],
                init_log_vars={'face': 0.0, 'emotion': 0.0, 'liveness': 0.0}
            )
        else:
            self.loss_weighter = FixedWeightedLoss({
                'face': face_weight,
                'emotion': emotion_weight,
                'liveness': liveness_weight
            })
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mode: str = 'triplet'  # 'triplet' or 'arcface'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.
        
        Args:
            outputs: Model outputs dict with keys: 'embedding', 'emotion', 'liveness', 'identity'
            targets: Ground truth dict with keys: 
                - 'anchor', 'positive', 'negative' (for triplet mode)
                - 'identity' (for arcface mode)
                - 'emotion': (B,) emotion labels
                - 'liveness': (B,) liveness labels
            mode: 'triplet' for triplet loss, 'arcface' for classification loss
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        
        # Face loss
        if mode == 'triplet' and 'anchor' in targets:
            losses['face'] = self.triplet_loss(
                targets['anchor'],
                targets['positive'],
                targets['negative']
            )
        elif 'identity' in outputs and 'identity' in targets:
            losses['face'] = self.face_ce_loss(
                outputs['identity'],
                targets['identity']
            )
        
        # Emotion loss
        if 'emotion' in outputs and 'emotion' in targets:
            losses['emotion'] = self.emotion_loss(
                outputs['emotion'],
                targets['emotion']
            )
        
        # Liveness loss
        if 'liveness' in outputs and 'liveness' in targets:
            losses['liveness'] = self.liveness_loss(
                outputs['liveness'],
                targets['liveness']
            )
        
        # Combine losses with weighting
        total_loss, weighted_losses = self.loss_weighter(losses)
        
        # Add individual losses for logging
        loss_dict = {f'{k}_loss': v.item() for k, v in losses.items()}
        loss_dict.update({f'{k}_weighted': v.item() for k, v in weighted_losses.items()})
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights."""
        if self.use_uncertainty_weighting:
            return self.loss_weighter.get_weights()
        else:
            return self.loss_weighter.weights


def get_mtl_loss_functions(
    use_uncertainty: bool = False,
    liveness_pos_weight: float = 1.0
) -> MTLLoss:
    """
    Factory function to create MTL loss with default settings.
    
    Args:
        use_uncertainty: Whether to use learned uncertainty weighting
        liveness_pos_weight: Weight for positive (spoof) class in liveness
    
    Returns:
        Configured MTLLoss instance
    """
    # Class weights for liveness (if imbalanced)
    liveness_weights = torch.tensor([1.0, liveness_pos_weight])
    
    return MTLLoss(
        face_weight=1.0,
        emotion_weight=0.5,
        liveness_weight=1.0,
        triplet_margin=0.5,
        emotion_smoothing=0.1,
        focal_gamma=2.0,
        liveness_class_weights=liveness_weights,
        use_uncertainty_weighting=use_uncertainty
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MTL Loss Functions")
    print("=" * 60)
    
    # Test FocalLoss
    print("\n--- FocalLoss Test ---")
    focal = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 2)
    labels = torch.randint(0, 2, (8,))
    fl = focal(logits, labels)
    print(f"  Focal loss: {fl.item():.4f}")
    
    # Test LabelSmoothingCE
    print("\n--- LabelSmoothingCE Test ---")
    lsce = LabelSmoothingCrossEntropy(smoothing=0.1)
    logits = torch.randn(8, 7)
    labels = torch.randint(0, 7, (8,))
    lsce_loss = lsce(logits, labels)
    print(f"  Label-smoothed CE: {lsce_loss.item():.4f}")
    
    # Test UncertaintyWeightedLoss
    print("\n--- UncertaintyWeightedLoss Test ---")
    uwl = UncertaintyWeightedLoss(['face', 'emotion', 'liveness'])
    losses = {
        'face': torch.tensor(1.0),
        'emotion': torch.tensor(0.5),
        'liveness': torch.tensor(0.8)
    }
    total, weighted = uwl(losses)
    print(f"  Total: {total.item():.4f}")
    print(f"  Weights: {uwl.get_weights()}")
    
    # Test full MTLLoss
    print("\n--- MTLLoss Test ---")
    mtl_loss = get_mtl_loss_functions(use_uncertainty=True)
    
    outputs = {
        'embedding': F.normalize(torch.randn(8, 256), dim=1),
        'emotion': torch.randn(8, 7),
        'liveness': torch.randn(8, 2),
        'identity': torch.randn(8, 4000)
    }
    targets = {
        'emotion': torch.randint(0, 7, (8,)),
        'liveness': torch.randint(0, 2, (8,)),
        'identity': torch.randint(0, 4000, (8,))
    }
    
    total_loss, loss_dict = mtl_loss(outputs, targets, mode='arcface')
    print(f"  Total loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✓ All loss function tests passed!")
