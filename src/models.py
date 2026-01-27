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
    
    def _forward_backbone(self, x, return_attention=False):
        """
        Forward pass through backbone blocks.
        
        Args:
            x: Input tensor
            return_attention: If True, return Block 4 spatial attention map
            
        Returns:
            Feature tensor, and optionally the attention map from Block 4
        """
        # Block 1
        x = self.block1_conv(x)
        x = self.block1_cbam(x)
        x = self.block1_pool(x)
        
        # Block 2
        x = self.block2_conv(x)
        x = self.block2_cbam(x)
        x = self.block2_pool(x)
        
        # Block 3
        x = self.block3_conv(x)
        x = self.block3_cbam(x)
        x = self.block3_pool(x)
        
        # Block 4 - optionally extract attention map
        x = self.block4_conv(x)
        if return_attention:
            x, attention_map = self.block4_cbam(x, return_attention=True)
        else:
            x = self.block4_cbam(x)
            attention_map = None
        x = self.block4_pool(x)
        
        if return_attention:
            return x, attention_map
        return x

    def forward(self, x, mode='metric', return_attention=False):
        """
        Forward pass with different modes.
        
        args:
            x: Input image tensor (batch_size, 3, H, W)
            mode: 'metric' for embeddings, 'classification' for logits, 
                  'embedding' for raw features
            return_attention: If True, also return Block 4 spatial attention map
        
        returns:
            Embeddings, logits, or features depending on mode
            If return_attention=True, returns tuple of (output, attention_map)
        """
        # extract features using backbone
        if return_attention:
            x, attention_map = self._forward_backbone(x, return_attention=True)
        else:
            x = self._forward_backbone(x, return_attention=False)
            attention_map = None
            
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
            return output, attention_map
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
    
    print("\n✓ Model test passed!")
