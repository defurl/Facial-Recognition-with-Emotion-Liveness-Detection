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

    def forward(self, x):
        # along channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Combines channel and spatial attention. Lightweight and easy to insert
    after convolutional blocks.
    """
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
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

        blocks = []

        # block 1
        blocks += [
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        blocks.append(CBAM(32))
        blocks.append(nn.MaxPool2d(2, 2))

        # block 2
        blocks += [
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        blocks.append(CBAM(64))
        blocks.append(nn.MaxPool2d(2, 2))

        # block 3
        blocks += [
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        blocks.append(CBAM(128))
        blocks.append(nn.MaxPool2d(2, 2))

        # block 4
        blocks += [
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        blocks.append(CBAM(256))
        blocks.append(nn.MaxPool2d(2, 2))

        self.backbone = nn.Sequential(*blocks)

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

    def forward(self, x, mode='metric'):
        """
        Forward pass with different modes.
        
        args:
            x: Input image tensor (batch_size, 3, H, W)
            mode: 'metric' for embeddings, 'classification' for logits, 
                  'embedding' for raw features
        
        returns:
            Embeddings, logits, or features depending on mode
        """
        # extract features using backbone
        x = self.backbone(x)
        features = self.gap(x) # shape: (batchsize, 256, 1, 1)
        features = features.view(x.size(0), -1) # shape: (batchsize, 256)
        
        if mode == 'metric':
            # runs the embedding head
            embedding = self.embedding_head(features)
            # normalize embeddings to unit length
            return F.normalize(embedding, p=2, dim=1)
        
        elif mode == 'classification':
            # returns classification score (logits)
            return self.classifier_head(features)
        
        elif mode == 'embedding':
            # returns raw features for verification task
            return features
        
        else:
            raise ValueError(f"Unknown mode: {mode}")

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
