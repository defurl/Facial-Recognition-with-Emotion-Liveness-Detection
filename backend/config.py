"""Backend configuration for artifact paths and runtime defaults.

Adjust these paths if you change layout; expected to run from backend/ with artifacts/ as sibling.
"""
from pathlib import Path

# Base directories
BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

# Artifact paths (models, db, thresholds)
OUTPUTS_DIR = ARTIFACTS_ROOT / "outputs"
MODEL_METRIC_PATH = OUTPUTS_DIR / "best_metric_model.pth"
EMPLOYEE_DB_PATH = OUTPUTS_DIR / "employee_db.pt"
GUI_THRESHOLD_PATH = OUTPUTS_DIR / "gui_threshold.json"

# Camera/API defaults (override via env vars or server settings as needed)
DEFAULT_DEVICE = "cpu"
DEFAULT_THRESHOLD_GUI = 1.1  # matches previous OPTIMAL_THRESHOLD_GUI fallback

# Ensure outputs dir exists (non-fatal if not present yet)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
