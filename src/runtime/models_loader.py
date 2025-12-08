"""Model loading utilities for the GUI pipeline."""
from pathlib import Path
import torch

from src.config import DEVICE, EMBEDDING_DIM, MODEL_METRIC_PATH
from src.models import FaceEmbeddingCNN


def load_verification_model(model_path: Path = MODEL_METRIC_PATH, device=DEVICE, num_classes: int = 4000) -> FaceEmbeddingCNN:
    """Create the verification model and load weights if present."""
    model = FaceEmbeddingCNN(embedding_dim=EMBEDDING_DIM, num_classes=num_classes).to(device)

    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
    else:
        # Keep model in eval mode even if weights are missing to avoid dropout/bn drift.
        model.eval()
    return model
