"""
Emotion Detection Module
Supports both DeepFace (pretrained) and Custom trained model
Toggle via USE_CUSTOM_EMOTION_MODEL in config.py
"""

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from config import (
    USE_CUSTOM_EMOTION_MODEL, EMOTION_MODEL_PATH, EMOTION_IMG_SIZE,
    EMOTION_LABELS, NUM_EMOTION_CLASSES, DEVICE, IMAGENET_MEAN, IMAGENET_STD
)

# DeepFace emotion labels mapping
DEEPFACE_EMOTION_LABELS = {
    'angry': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happy': 'Happy',
    'sad': 'Sad',
    'surprise': 'Surprise',
    'neutral': 'Neutral'
}

# Global custom model (lazy loaded)
_custom_emotion_model = None
_emotion_transform = None


def _load_custom_model():
    """Lazy load custom emotion model"""
    global _custom_emotion_model, _emotion_transform
    
    if _custom_emotion_model is not None:
        return _custom_emotion_model
    
    if not EMOTION_MODEL_PATH.exists():
        print(f"Warning: Custom emotion model not found at {EMOTION_MODEL_PATH}")
        print("Falling back to DeepFace...")
        return None
    
    try:
        from emotion_model import EmotionCNN
        
        print(f"Loading custom emotion model from {EMOTION_MODEL_PATH}...")
        model = EmotionCNN(num_classes=NUM_EMOTION_CLASSES, use_cbam=True)
        model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        
        _custom_emotion_model = model
        
        # Create transform
        _emotion_transform = transforms.Compose([
            transforms.Resize((EMOTION_IMG_SIZE, EMOTION_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        print("Custom emotion model loaded successfully!")
        return _custom_emotion_model
        
    except Exception as e:
        print(f"Error loading custom emotion model: {e}")
        print("Falling back to DeepFace...")
        return None


def analyze_emotion_custom(img):
    """
    Analyze emotion using custom trained model.
    
    Args:
        img: PIL Image, numpy array (H, W, 3), or file path
        
    Returns:
        Tuple of (emotion_label, confidence)
    """
    global _custom_emotion_model, _emotion_transform
    
    model = _load_custom_model()
    if model is None:
        return 'Neutral', 0.5
    
    try:
        # Convert to PIL Image if needed
        if isinstance(img, str):
            pil_img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img).convert('RGB')
        else:
            pil_img = img.convert('RGB')
        
        # Transform
        input_tensor = _emotion_transform(pil_img).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
        
        emotion_idx = pred_class.item()
        emotion_label = EMOTION_LABELS.get(emotion_idx, 'Neutral')
        confidence_score = confidence.item()
        
        return emotion_label, confidence_score
        
    except Exception as e:
        print(f"Custom emotion analysis error: {e}")
        return 'Neutral', 0.5


def analyze_emotion_deepface(img_path):
    """
    Analyze emotion using DeepFace.
    
    Args:
        img_path: Path or image array accepted by DeepFace.
        
    Returns:
        Tuple of (emotion_label, is_real)
    """
    try:
        from deepface import DeepFace
        
        analysis = DeepFace.analyze(
            img_path,
            actions=['emotion'],
            detector_backend='skip',  # Face already cropped
            anti_spoofing=True
        )
        
        result = analysis[0] if isinstance(analysis, list) else analysis
        
        # Get emotion
        em = result.get('dominant_emotion') or \
             (result.get('emotion') or {}).get('dominant')
        emotion = DEEPFACE_EMOTION_LABELS.get(str(em).lower(), 'Neutral') if em else 'Neutral'
        
        # Get liveness
        is_real = result.get('is_real')
        
        return emotion, bool(is_real) if is_real is not None else True
        
    except Exception as e:
        print(f"DeepFace analysis error: {e}")
        return 'Neutral', True


def analyze_emotion_and_liveness(img_path):
    """
    Analyze emotion and liveness from face image.
    
    Uses custom model or DeepFace based on USE_CUSTOM_EMOTION_MODEL config.
    Note: Custom model does NOT provide liveness detection,
          so liveness always returns True when using custom model.
    
    Args:
        img_path: Path or image array
        
    Returns:
        Tuple of (emotion_label, is_real)
    """
    if USE_CUSTOM_EMOTION_MODEL:
        # Try custom model first
        model = _load_custom_model()
        if model is not None:
            emotion, confidence = analyze_emotion_custom(img_path)
            # Custom model doesn't do liveness, assume real
            # For liveness, still use DeepFace if needed
            return emotion, True
    
    # Fall back to DeepFace
    return analyze_emotion_deepface(img_path)


def get_emotion_probabilities(img):
    """
    Get full emotion probability distribution.
    Only available with custom model.
    
    Args:
        img: PIL Image, numpy array, or file path
        
    Returns:
        Dict of {emotion: probability} or None if using DeepFace
    """
    if not USE_CUSTOM_EMOTION_MODEL:
        return None
    
    model = _load_custom_model()
    if model is None:
        return None
    
    try:
        # Convert to PIL Image if needed
        if isinstance(img, str):
            pil_img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img).convert('RGB')
        else:
            pil_img = img.convert('RGB')
        
        # Transform
        input_tensor = _emotion_transform(pil_img).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
        
        result = {}
        for idx, emotion in EMOTION_LABELS.items():
            result[emotion] = float(probs[idx].item())
        
        return result
        
    except Exception as e:
        print(f"Error getting emotion probabilities: {e}")
        return None


# Export for backward compatibility
emotion_labels = DEEPFACE_EMOTION_LABELS

__all__ = [
    'emotion_labels',
    'analyze_emotion_and_liveness',
    'analyze_emotion_custom',
    'analyze_emotion_deepface',
    'get_emotion_probabilities',
    'EMOTION_LABELS'
]
