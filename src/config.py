"""
Configuration for Face Verification with Dual-Stream CNN + TDA
Combines CNN embeddings with GUDHI-based Persistence Image features
"""

from pathlib import Path

# ============= Project Paths =============
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "classification_data" / "train_data"
VAL_DIR = DATASET_DIR / "classification_data" / "val_data"
TEST_DIR = DATASET_DIR / "classification_data" / "test_data"
VERIFICATION_DIR = DATASET_DIR / "verification_data"

VERIFICATION_VAL_PAIRS = DATASET_DIR / "verification_pairs_val.txt"
VERIFICATION_TEST_PAIRS = DATASET_DIR / "verification_pairs_test.txt"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============= Model Paths =============
MODEL_PATH = OUTPUT_DIR / "best_dual_stream_model.pth"
TDA_CACHE_DIR = OUTPUT_DIR / "tda_cache"

# ============= Model Architecture =============
IMG_SIZE = 64  # Image size (64x64)
EMBEDDING_DIM = 256  # CNN embedding dimension
TDA_DIM = 400  # TDA persistence image features (20x20)
TDA_HIDDEN_DIM = 128  # TDA branch hidden dimension
FUSION_DIM = EMBEDDING_DIM + TDA_HIDDEN_DIM  # Combined: 256 + 128 = 384
USE_CBAM = True  # CBAM attention modules in CNN

# ============= Training Hyperparameters =============
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4  # L2 regularization
MARGIN = 0.5  # Triplet loss margin
DROPOUT = 0.3  # Dropout rate

# ============= Training Settings =============
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10

# ============= Data Augmentation =============
AUGMENTATION = {
    'horizontal_flip': 0.5,
    'rotation_degrees': 10,
    'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1},
    'random_affine_translate': (0.1, 0.1),
    'random_erasing': {'p': 0.1, 'scale': (0.02, 0.1)},
}

# ============= Learning Rate Scheduler =============
LR_SCHEDULER = {
    'type': 'ReduceLROnPlateau',
    'mode': 'min',
    'factor': 0.5,
    'patience': 3,
}

# ============= TDA Feature Extraction (GUDHI) =============
TDA_CONFIG = {
    'resolution': (20, 20),  # Persistence image resolution
    'bandwidth': 50.0,  # Gaussian kernel bandwidth
    'weight_max': 1.0,  # Maximum weight for persistence
}

# ============= Verification Settings =============
OPTIMAL_THRESHOLD = 1.15  # Distance threshold for verification

# ============= Image Normalization =============
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============= Device Configuration =============
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_config():
    """Print configuration summary"""
    print("=" * 70)
    print("FACE VERIFICATION - DUAL-STREAM CNN + TDA CONFIGURATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"\nModel Architecture:")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  CNN Embedding: {EMBEDDING_DIM}-dim")
    print(f"  TDA Features: {TDA_DIM}-dim -> {TDA_HIDDEN_DIM}-dim")
    print(f"  Fusion Dim: {FUSION_DIM}-dim")
    print(f"  CBAM Attention: {USE_CBAM}")
    print(f"\nTraining:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Triplet Margin: {MARGIN}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  LR Scheduler: {LR_SCHEDULER['type']}")
    print(f"\nTDA (Persistence Images):")
    print(f"  Resolution: {TDA_CONFIG['resolution']}")
    print(f"  Bandwidth: {TDA_CONFIG['bandwidth']}")
    print(f"\nPaths:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  TDA Cache: {TDA_CACHE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
