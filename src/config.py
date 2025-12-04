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
OPTIMAL_THRESHOLD_GUI = 1.1  # for GUI - optimized based on verification analysis (was 0.8)

# ============= GUI Settings =============
PROCESS_EVERY_N_FRAMES = 10  # process heavy tasks every N frames
TARGET_FPS = 30
GUI_WINDOW_WIDTH = 640
GUI_WINDOW_HEIGHT = 480
CAMERA_INDEX = 0  # default camera index; app will fallback to others if unavailable

# ============= Registration Settings =============
REGISTRATION_MODE_FULL = 6  # number of captures for full registration
REGISTRATION_MODE_QUICK = 3  # number of captures for quick registration
REGISTRATION_POSES_FULL = ["center", "right", "left", "up", "down", "center"]
REGISTRATION_POSES_QUICK = ["center", "right", "left"]
REGISTRATION_INSTRUCTIONS_FULL = [
    "Face forward",
    "Turn right",
    "Turn left",
    "Look up",
    "Look down",
    "Face forward again"
]
REGISTRATION_INSTRUCTIONS_QUICK = [
    "Face forward",
    "Turn left",
    "Turn right"
]

# Quality thresholds for registration
BLUR_THRESHOLD_STRICT = 100  # Laplacian variance threshold
BLUR_THRESHOLD_RELAXED = 80
LIGHTING_MIN_BRIGHT_STRICT = 30
LIGHTING_MAX_BRIGHT_STRICT = 220
LIGHTING_MIN_BRIGHT_RELAXED = 25
LIGHTING_MAX_BRIGHT_RELAXED = 230
LIGHTING_MIN_CONTRAST = 30

# Pose tolerances
YAW_TOLERANCE_STRICT = 5.0  # degrees
YAW_TOLERANCE_RELAXED = 8.0
PITCH_TOLERANCE_STRICT = 7.0  # degrees
PITCH_TOLERANCE_RELAXED = 10.0

# Adaptive threshold settings
ADAPTIVE_THRESHOLD_ATTEMPTS = 2  # switch to relaxed after this many attempts
FAILURE_TOOLTIP_THRESHOLD = 5  # show tooltip after this many failures

# ============= Attendance Settings =============
ATTENDANCE_COOLDOWN_MINUTES = 60  # minimum minutes between attendance marks
ATTENDANCE_LOG_PATH = OUTPUT_DIR / "attendance_log.csv"

# ============= Multi-Face Settings =============
PRIMARY_FACE_AREA_WEIGHT = 0.6  # weight for face area in primary selection
PRIMARY_FACE_CENTER_WEIGHT = 0.4  # weight for centeredness in primary selection
MAX_CONCURRENT_FACES = 4  # maximum number of faces to process simultaneously
CONFIDENCE_REJECTION_THRESHOLD = 0.05  # minimum confidence to accept recognition - lowered to 5% based on analysis
UNRECOGNIZED_DISTANCE_MULTIPLIER = 1.2  # stricter threshold for small databases to prevent identity mix-ups

# ============= Feature Flags =============
USE_MULTI_EMBEDDING = True  # use multi-embedding matching for verification
DEBUG_MODE = False  # enable debug logging and frame saving

# ============= Explainability Settings (NEW for Hackathon) =============
ENABLE_EXPLAINABILITY = True  # Enable xAI features
SHOW_ATTENTION_MAPS = True  # Show CBAM attention heatmaps
SHOW_KNN_NEIGHBORS = True  # Show k-nearest neighbors
NUM_NEIGHBORS_DISPLAY = 5  # Number of neighbors to display
KNN_CONFIDENCE_THRESHOLD = 0.6  # Minimum kNN confidence for acceptance
USE_ADAPTIVE_CBAM = False  # Use adaptive CBAM (slower but more robust)
USE_REGION_CBAM = False  # Use region-aware CBAM

# ============= Dashboard Settings (NEW for Hackathon) =============
ENABLE_TSNE_DASHBOARD = True  # Enable t-SNE visualization dashboard
TSNE_UPDATE_INTERVAL = 2000  # Update dashboard every N ms
TSNE_PERPLEXITY = 30  # t-SNE perplexity parameter
TRACK_EMBEDDING_DRIFT = True  # Track embedding changes over time
DRIFT_WARNING_THRESHOLD = 0.3  # Cosine distance threshold for drift warning

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
