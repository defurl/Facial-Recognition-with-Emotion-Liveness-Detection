"""
Face Recognition Package
"""

__version__ = "1.0.0"
__author__ = "Minh Hieu Tran"

from .config import *
from .models import FaceEmbeddingCNN, get_loss_functions
from .data_loader import create_dataloaders, load_verification_pairs, get_transforms
from .utils import check_liveness, detect_faces
