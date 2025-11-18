"""
Configuration file for Face Recognition Attendance System
Contains all hyperparameters, paths, and settings
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
MODEL_SOFTMAX_PATH = OUTPUT_DIR / "best_softmax_model.pth"
MODEL_METRIC_PATH = OUTPUT_DIR / "best_metric_model.pth"
EMPLOYEE_DB_PATH = OUTPUT_DIR / "employee_db.pt"

# ============= Model Hyperparameters =============
IMG_SIZE = 64  # default image size (64x64)
EMBEDDING_DIM = 256  # embedding dimension for metric learning
BATCH_SIZE = 128
USE_CBAM = True  # toggle to enable/disable CBAM modules in the model
# Update default training epochs when experimenting with CBAM
NUM_EPOCHS_SOFTMAX = 80
NUM_EPOCHS_METRIC = 80
LEARNING_RATE = 1e-3
MARGIN = 0.5  # triplet loss margin

# ============= Training Settings =============
MAX_IMAGES_PER_IDENTITY_TRAIN = 30  # limit training images per identity
RANDOM_SEED = 42

# ============= Verification Settings =============
OPTIMAL_THRESHOLD = 1.15  # for notebook evaluation
OPTIMAL_THRESHOLD_GUI = 0.8  # for GUI (stricter)

# ============= GUI Settings =============
PROCESS_EVERY_N_FRAMES = 10  # process heavy tasks every N frames
TARGET_FPS = 30
GUI_WINDOW_WIDTH = 640
GUI_WINDOW_HEIGHT = 480
CAMERA_INDEX = 1  # default camera index; app will fallback to others if unavailable

# ============= Image Normalization =============
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============= Device Configuration =============
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_config():
    """Print configuration summary"""
    print("=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"\nModel Settings:")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Embedding Dim: {EMBEDDING_DIM}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"\nTraining:")
    print(f"  Softmax Epochs: {NUM_EPOCHS_SOFTMAX}")
    print(f"  Metric Epochs: {NUM_EPOCHS_METRIC}")
    print(f"  Max Images/Identity: {MAX_IMAGES_PER_IDENTITY_TRAIN}")
    print(f"\nVerification:")
    print(f"  Threshold (Evaluation): {OPTIMAL_THRESHOLD}")
    print(f"  Threshold (GUI): {OPTIMAL_THRESHOLD_GUI}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()
