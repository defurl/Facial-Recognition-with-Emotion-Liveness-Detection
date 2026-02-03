import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module from CBAM.

    Uses both average and max pooling descriptors followed by a shared MLP.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module from CBAM.

    Computes attention map using average and max pooling along channel axis,
    followed by a convolution.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_attention=False):
        # along channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        attention_map = self.sigmoid(out)
        
        if return_attention:
            return attention_map, attention_map
        return attention_map


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Combines channel and spatial attention. Lightweight and easy to insert
    after convolutional blocks.
    """
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x, return_attention=False):
        out = x * self.channel_att(x)
        spatial_att_map = self.spatial_att(out)
        out = out * spatial_att_map
        
        if return_attention:
            return out, spatial_att_map
        return out


class FaceEmbeddingCNN(nn.Module):
    """
    Custom CNN for face embedding extraction.
    Supports both metric learning (embeddings) and classification (softmax).
    """
    
    def __init__(self, embedding_dim=256, num_classes=4000, use_cbam=False):
        super(FaceEmbeddingCNN, self).__init__()
        # insert CBAM modules after each convolutional block
        self.use_cbam = use_cbam

        # Block 1
        self.block1_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1_cbam = CBAM(32)
        self.block1_pool = nn.MaxPool2d(2, 2)

        # Block 2
        self.block2_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2_cbam = CBAM(64)
        self.block2_pool = nn.MaxPool2d(2, 2)

        # Block 3
        self.block3_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block3_cbam = CBAM(128)
        self.block3_pool = nn.MaxPool2d(2, 2)

        # Block 4
        self.block4_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.block4_cbam = CBAM(256)
        self.block4_pool = nn.MaxPool2d(2, 2)

        # gap (global average pooling) to reduce feature map to vector
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # head for metric learning
        # projects to embedding space
        self.embedding_head = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        # head for classification
        # classifies into the 4000 identities
        self.classifier_head = nn.Linear(256, num_classes)
    
    def _forward_backbone(self, x, return_attention=False, multi_scale=False):
        """
        Forward pass through backbone blocks.
        
        Args:
            x: Input tensor
            return_attention: If True, return spatial attention map(s)
            multi_scale: If True, return attention from Block 2 and Block 4
            
        Returns:
            Feature tensor, and optionally the attention map(s)
            - If multi_scale=False: single attention map from Block 4 (8x8)
            - If multi_scale=True: dict with 'block2' (16x16) and 'block4' (8x8)
        """
        attention_maps = {}
        
        # Block 1 (64x64 -> 32x32)
        x = self.block1_conv(x)
        x = self.block1_cbam(x)
        x = self.block1_pool(x)
        
        # Block 2 (32x32 -> 16x16) - extract attention for multi-scale TDA
        x = self.block2_conv(x)
        if return_attention and multi_scale:
            x, att2 = self.block2_cbam(x, return_attention=True)
            attention_maps['block2'] = att2  # 16x16 attention map
        else:
            x = self.block2_cbam(x)
        x = self.block2_pool(x)
        
        # Block 3 (16x16 -> 8x8)
        x = self.block3_conv(x)
        x = self.block3_cbam(x)
        x = self.block3_pool(x)
        
        # Block 4 (8x8 -> 4x4) - always extract attention when requested
        x = self.block4_conv(x)
        if return_attention:
            x, att4 = self.block4_cbam(x, return_attention=True)
            attention_maps['block4'] = att4  # 8x8 attention map
        else:
            x = self.block4_cbam(x)
        x = self.block4_pool(x)
        
        if return_attention:
            if multi_scale:
                return x, attention_maps
            else:
                # Backward compatible: return single attention map
                return x, attention_maps.get('block4')
        return x

    def forward(self, x, mode='metric', return_attention=False, multi_scale=False):
        """
        Forward pass with different modes.
        
        args:
            x: Input image tensor (batch_size, 3, H, W)
            mode: 'metric' for embeddings, 'classification' for logits, 
                  'embedding' for raw features
            return_attention: If True, also return spatial attention map(s)
            multi_scale: If True, return attention from multiple blocks
        
        returns:
            Embeddings, logits, or features depending on mode
            If return_attention=True, returns tuple of (output, attention_map(s))
        """
        # extract features using backbone
        if return_attention:
            x, attention_maps = self._forward_backbone(x, return_attention=True, multi_scale=multi_scale)
        else:
            x = self._forward_backbone(x, return_attention=False)
            attention_maps = None
            
        features = self.gap(x) # shape: (batchsize, 256, 1, 1)
        features = features.view(x.size(0), -1) # shape: (batchsize, 256)
        
        if mode == 'metric':
            # runs the embedding head
            embedding = self.embedding_head(features)
            # normalize embeddings to unit length
            output = F.normalize(embedding, p=2, dim=1)
        
        elif mode == 'classification':
            # returns classification score (logits)
            output = self.classifier_head(features)
        
        elif mode == 'embedding':
            # returns raw features for verification task
            output = features
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if return_attention:
            return output, attention_maps
        return output


def get_loss_functions():
    """
    Get loss functions for training.
    
    returns:
        Tuple of (softmax_loss, triplet_loss)
    """
    # outputs raw logits 
    loss_softmax = nn.CrossEntropyLoss()
    # use margin for triplet loss
    loss_triplet = nn.TripletMarginLoss(margin=0.5, p=2)
    
    return loss_softmax, loss_triplet


# ============================================================================
# FUSION MODULES - Different strategies for combining CNN + TDA features
# ============================================================================

class ConcatFusion(nn.Module):
    """
    Naive concatenation fusion (baseline).
    
    Simply concatenates CNN and TDA features, then applies MLP.
    This was the original approach - now deprecated in favor of attention.
    """
    def __init__(self, cnn_dim: int, tda_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        fusion_dim = cnn_dim + tda_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    
    def forward(self, cnn_features: torch.Tensor, tda_features: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([cnn_features, tda_features], dim=1)
        return self.fusion(fused)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for CNN and TDA features.
    
    Learns to dynamically weight the importance of CNN vs TDA for each sample.
    This is the recommended fusion strategy based on experiments showing +3.13%
    improvement over naive concatenation.
    
    Key insight: Not all images benefit equally from TDA. Some faces have 
    distinctive topology (TDA useful), others are better distinguished by 
    appearance (CNN useful). The network learns to weight appropriately per-sample.
    
    Architecture:
        CNN features ──┬──▶ Project to hidden_dim ──┐
                       │                            │
                       ▼                            ▼
                 ┌───────────┐                ┌─────────────┐
                 │ Attention │                │   Weighted  │
                 │  Network  │ ──────────────▶│     Sum     │ ──▶ Output
                 └───────────┘                └─────────────┘
                       ▲                            ▲
                       │                            │
        TDA features ──┴──▶ Project to hidden_dim ──┘
    """
    def __init__(self, cnn_dim: int, tda_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = output_dim
        
        # Project both modalities to same dimension
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.tda_proj = nn.Sequential(
            nn.Linear(tda_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Attention mechanism - learns modality importance per sample
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 2),  # 2 modalities: CNN and TDA
            nn.Softmax(dim=1),
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    
    def forward(self, cnn_features: torch.Tensor, tda_features: torch.Tensor) -> torch.Tensor:
        # Project both to common dimension
        cnn_proj = self.cnn_proj(cnn_features)  # (B, hidden_dim)
        tda_proj = self.tda_proj(tda_features)  # (B, hidden_dim)
        
        # Compute attention weights based on both modalities
        combined = torch.cat([cnn_proj, tda_proj], dim=1)  # (B, hidden_dim * 2)
        weights = self.attention(combined)  # (B, 2)
        
        # Weighted combination of modalities
        cnn_weight = weights[:, 0:1]  # (B, 1)
        tda_weight = weights[:, 1:2]  # (B, 1)
        fused = cnn_weight * cnn_proj + tda_weight * tda_proj  # (B, hidden_dim)
        
        return self.output(fused)


class GatedFusion(nn.Module):
    """
    Gated fusion using LSTM-style gates to control information flow.
    
    Uses sigmoid gates to selectively pass information from each modality.
    Can completely shut off a modality if it's noisy for a given sample.
    """
    def __init__(self, cnn_dim: int, tda_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = output_dim
        
        # Projections
        self.cnn_proj = nn.Linear(cnn_dim, self.hidden_dim)
        self.tda_proj = nn.Linear(tda_dim, self.hidden_dim)
        
        # Gates (sigmoid for 0-1 range)
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
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
    
    def forward(self, cnn_features: torch.Tensor, tda_features: torch.Tensor) -> torch.Tensor:
        # Compute gates based on both modalities
        combined = torch.cat([cnn_features, tda_features], dim=1)
        cnn_g = self.cnn_gate(combined)
        tda_g = self.tda_gate(combined)
        
        # Project and gate
        cnn_proj = self.cnn_proj(cnn_features)
        tda_proj = self.tda_proj(tda_features)
        
        # Gated fusion
        fused = cnn_g * cnn_proj + tda_g * tda_proj
        
        return self.output(fused)


class ResidualFusion(nn.Module):
    """
    Residual fusion - CNN is primary, TDA provides learned corrections.
    
    Most parameter-efficient option. TDA acts as a refinement signal
    rather than an equal partner.
    
    Formula: output = CNN_main + α * tanh(TDA_residual)
    where α is a learnable scale initialized to 0.1
    """
    def __init__(self, cnn_dim: int, tda_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        # Main path: CNN
        self.cnn_main = nn.Sequential(
            nn.Linear(cnn_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )
        
        # Residual path: TDA -> bounded correction
        self.tda_residual = nn.Sequential(
            nn.Linear(tda_dim, output_dim // 2),
            nn.BatchNorm1d(output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, output_dim),
            nn.Tanh(),  # Bound residual to [-1, 1]
        )
        
        # Learnable residual scale (starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
        self.output = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, cnn_features: torch.Tensor, tda_features: torch.Tensor) -> torch.Tensor:
        main = self.cnn_main(cnn_features)
        residual = self.tda_residual(tda_features) * self.residual_scale
        
        fused = main + residual
        return self.output(fused)


# Registry of available fusion strategies
FUSION_STRATEGIES = {
    'concat': ConcatFusion,
    'attention': AttentionFusion,
    'gated': GatedFusion,
    'residual': ResidualFusion,
}


class DualStreamFaceNet(nn.Module):
    """
    Dual-stream face embedding network combining CNN and TDA features.
    
    Architecture:
        Stream 1: CNN backbone → 256-dim CNN features
        Stream 2: TDA features (400-dim persistence images) → projected
        Fusion: Configurable strategy (attention, gated, residual, concat)
        Output: Embedding head for metric learning
        
    The fusion strategy is configurable. Based on experiments:
        - 'attention': Best accuracy (+3.13% over concat) - RECOMMENDED
        - 'gated': Second best (+2.63% over concat)
        - 'residual': Most efficient (+2.50%, fewest params)
        - 'concat': Baseline (original naive approach)
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        num_classes: int = 4000,
        tda_dim: int = 400,
        fusion_strategy: str = 'attention',  # NEW: configurable fusion
        fusion_hidden_dim: int = 384,
        use_cbam: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize dual-stream network.
        
        Args:
            embedding_dim: Output embedding dimension
            num_classes: Number of identity classes  
            tda_dim: Input TDA feature dimension (default 400 = 20x20 persistence image)
            fusion_strategy: Fusion method - 'attention' (best), 'gated', 'residual', 'concat'
            fusion_hidden_dim: Hidden dimension for fusion layer (default 384)
            use_cbam: Whether to use CBAM attention in CNN backbone
            dropout: Dropout rate for fusion layer
        """
        super(DualStreamFaceNet, self).__init__()
        
        self.tda_dim = tda_dim
        self.fusion_strategy = fusion_strategy
        
        # Validate fusion strategy
        if fusion_strategy not in FUSION_STRATEGIES:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}. "
                           f"Available: {list(FUSION_STRATEGIES.keys())}")
        
        # Stream 1: CNN backbone (reuse existing architecture)
        self.use_cbam = use_cbam
        
        # Block 1
        self.block1_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1_cbam = CBAM(32)
        self.block1_pool = nn.MaxPool2d(2, 2)

        # Block 2
        self.block2_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2_cbam = CBAM(64)
        self.block2_pool = nn.MaxPool2d(2, 2)

        # Block 3
        self.block3_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block3_cbam = CBAM(128)
        self.block3_pool = nn.MaxPool2d(2, 2)

        # Block 4
        self.block4_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.block4_cbam = CBAM(256)
        self.block4_pool = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # CNN output: 256-dim
        cnn_out_dim = 256
        
        # Fusion module - configurable strategy
        # fusion_hidden_dim is output dim of fusion, used as input to heads
        FusionClass = FUSION_STRATEGIES[fusion_strategy]
        self.fusion = FusionClass(
            cnn_dim=cnn_out_dim,
            tda_dim=tda_dim,
            output_dim=fusion_hidden_dim,
            dropout=dropout,
        )
        
        # Embedding head (for metric learning)
        self.embedding_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        # Classification head
        self.classifier_head = nn.Linear(fusion_hidden_dim, num_classes)
        
    def _forward_cnn_backbone(self, x):
        """Forward pass through CNN backbone blocks."""
        # Block 1 (64x64 -> 32x32)
        x = self.block1_conv(x)
        if self.use_cbam:
            x = self.block1_cbam(x)
        x = self.block1_pool(x)
        
        # Block 2 (32x32 -> 16x16)
        x = self.block2_conv(x)
        if self.use_cbam:
            x = self.block2_cbam(x)
        x = self.block2_pool(x)
        
        # Block 3 (16x16 -> 8x8)
        x = self.block3_conv(x)
        if self.use_cbam:
            x = self.block3_cbam(x)
        x = self.block3_pool(x)
        
        # Block 4 (8x8 -> 4x4)
        x = self.block4_conv(x)
        if self.use_cbam:
            x = self.block4_cbam(x)
        x = self.block4_pool(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # (B, 256)
        
        return x
    
    def forward(self, x, tda_features=None, mode='metric'):
        """
        Forward pass with dual streams.
        
        Args:
            x: Input image tensor (batch_size, 3, H, W)
            tda_features: TDA features tensor (batch_size, tda_dim)
                          If None, only uses CNN stream (backward compatible)
            mode: 'metric' for embeddings, 'classification' for logits
        
        Returns:
            Output tensor based on mode
        """
        # Stream 1: CNN features
        cnn_features = self._forward_cnn_backbone(x)  # (B, 256)
        
        # Stream 2: TDA features (optional)
        if tda_features is not None:
            # Use fusion module (attention, gated, residual, or concat)
            fused = self.fusion(cnn_features, tda_features)
        else:
            # CNN-only mode: create zero TDA features for compatibility
            batch_size = cnn_features.size(0)
            device = cnn_features.device
            zero_tda = torch.zeros(batch_size, self.tda_dim, device=device)
            fused = self.fusion(cnn_features, zero_tda)
        
        if mode == 'metric':
            embedding = self.embedding_head(fused)
            return F.normalize(embedding, p=2, dim=1)
        
        elif mode == 'classification':
            return self.classifier_head(fused)
        
        elif mode == 'features':
            return fused
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    @classmethod
    def from_pretrained_cnn(cls, cnn_model: FaceEmbeddingCNN, **kwargs):
        """
        Create DualStreamFaceNet and initialize CNN weights from pretrained model.
        
        Args:
            cnn_model: Pretrained FaceEmbeddingCNN
            **kwargs: Additional arguments for DualStreamFaceNet
        
        Returns:
            DualStreamFaceNet with CNN weights initialized
        """
        dual_model = cls(**kwargs)
        
        # Copy CNN backbone weights
        dual_model.block1_conv.load_state_dict(cnn_model.block1_conv.state_dict())
        dual_model.block1_cbam.load_state_dict(cnn_model.block1_cbam.state_dict())
        dual_model.block2_conv.load_state_dict(cnn_model.block2_conv.state_dict())
        dual_model.block2_cbam.load_state_dict(cnn_model.block2_cbam.state_dict())
        dual_model.block3_conv.load_state_dict(cnn_model.block3_conv.state_dict())
        dual_model.block3_cbam.load_state_dict(cnn_model.block3_cbam.state_dict())
        dual_model.block4_conv.load_state_dict(cnn_model.block4_conv.state_dict())
        dual_model.block4_cbam.load_state_dict(cnn_model.block4_cbam.state_dict())
        
        print("Loaded CNN backbone weights from pretrained model")
        return dual_model


def count_parameters(model):
    """
    Count total and trainable parameters in model.
    
    args:
        model: PyTorch model
    
    returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test model creation
    print("Testing FaceEmbeddingCNN...")
    
    model = FaceEmbeddingCNN(embedding_dim=256, num_classes=4000)
    total, trainable = count_parameters(model)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 64, 64)
    
    print("\nTesting forward modes:")
    embedding_out = model(dummy_input, mode='metric')
    print(f"  Metric mode output shape: {embedding_out.shape}")
    
    classification_out = model(dummy_input, mode='classification')
    print(f"  Classification mode output shape: {classification_out.shape}")
    
    feature_out = model(dummy_input, mode='embedding')
    print(f"  Embedding mode output shape: {feature_out.shape}")
    
    print("\n✓ FaceEmbeddingCNN test passed!")
    
    # Test DualStreamFaceNet
    print("\n" + "=" * 50)
    print("Testing DualStreamFaceNet...")
    
    dual_model = DualStreamFaceNet(
        embedding_dim=256,
        num_classes=4000,
        tda_dim=400,
        tda_hidden_dim=128,
        use_cbam=True,
    )
    total, trainable = count_parameters(dual_model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test with TDA features
    dummy_tda = torch.randn(4, 400)
    
    print("\nTesting with TDA features:")
    embedding_out = dual_model(dummy_input, tda_features=dummy_tda, mode='metric')
    print(f"  Metric mode output shape: {embedding_out.shape}")
    
    classification_out = dual_model(dummy_input, tda_features=dummy_tda, mode='classification')
    print(f"  Classification mode output shape: {classification_out.shape}")
    
    # Test without TDA features (backward compatible)
    print("\nTesting without TDA features (CNN-only):")
    embedding_out = dual_model(dummy_input, tda_features=None, mode='metric')
    print(f"  Metric mode output shape: {embedding_out.shape}")
    
    # Test from_pretrained_cnn
    print("\nTesting from_pretrained_cnn...")
    dual_from_cnn = DualStreamFaceNet.from_pretrained_cnn(
        model,
        embedding_dim=256,
        num_classes=4000,
        tda_dim=400,
    )
    
    print("\n✓ DualStreamFaceNet test passed!")
