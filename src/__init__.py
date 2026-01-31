"""
Face Recognition with TDA (Topological Data Analysis)
Dual-Stream CNN + Persistence Image Architecture
"""

__version__ = "2.0.0"
__author__ = "Minh Hieu Tran"

# Core modules
from .config import *
from .models import FaceEmbeddingCNN, DualStreamFaceNet, get_loss_functions
from .data_loader import (
    create_dataloaders, 
    create_tda_dataloaders,
    load_verification_pairs, 
    get_transforms
)
from .tda_features import TDAFeatureExtractor, TDAFeatureCache

