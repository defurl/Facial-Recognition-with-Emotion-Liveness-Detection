"""
Configuration file for Face Verification with Metric Learning
Contains hyperparameters, paths, and settings for triplet loss training
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
MODEL_METRIC_PATH = OUTPUT_DIR / "best_metric_model.pth"
MODEL_BASELINE_PATH = OUTPUT_DIR / "OLD_baseline_metric_model.pth"

# ============= Model Hyperparameters =============
IMG_SIZE = 64  # default image size (64x64)
EMBEDDING_DIM = 256  # embedding dimension for metric learning
BATCH_SIZE = 128
USE_CBAM = True  # CBAM attention modules enabled
NUM_EPOCHS = 50  # optimized epoch count
LEARNING_RATE = 1e-3
MARGIN = 0.5  # triplet loss margin

# ============= Training Settings =============
MAX_IMAGES_PER_IDENTITY_TRAIN = None  # None = use ALL images (full dataset)
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10

# ============= TDA (Topological Data Analysis) Settings =============
USE_TDA = True  # Enable TDA regularization on attention maps
TDA_LOSS_WEIGHT = 0.05  # Weight for TDA loss term in total loss
TDA_APPLY_EVERY_N_BATCHES = 1  # Apply TDA loss every N batches (1 = every batch)
TDA_TARGET_ENTROPY = 0.5  # Target persistence entropy (lower = more focused attention)
TDA_ENTROPY_WEIGHT = 1.0  # Weight for entropy deviation in TDA loss
TDA_COMPLEXITY_WEIGHT = 0.1  # Weight for topological complexity penalty

# ============= Verification Settings =============
OPTIMAL_THRESHOLD = 1.15  # for ROC evaluation

# ============= Image Normalization =============
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============= Device Configuration =============
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_config():
    """Print configuration summary"""
    print("=" * 70)
    print("FACE VERIFICATION - CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"\nModel Settings:")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Embedding Dim: {EMBEDDING_DIM}")
    print(f"  CBAM Attention: {USE_CBAM}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Triplet Margin: {MARGIN}")
    print(f"\nTraining:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Early Stopping: patience={EARLY_STOPPING_PATIENCE}")
    print(f"  Max Images/Identity: {'ALL' if MAX_IMAGES_PER_IDENTITY_TRAIN is None else MAX_IMAGES_PER_IDENTITY_TRAIN}")
    print(f"\nTDA Settings:")
    print(f"  Enabled: {USE_TDA}")
    print(f"  Loss Weight: {TDA_LOSS_WEIGHT}")
    print(f"  Apply Every N Batches: {TDA_APPLY_EVERY_N_BATCHES}")
    print(f"  Target Entropy: {TDA_TARGET_ENTROPY}")
    print(f"\nVerification:")
    print(f"  Threshold: {OPTIMAL_THRESHOLD}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()
