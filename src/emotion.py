from deepface import DeepFace
import torch
import os

emotion_labels = {
    'angry': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happy': 'Happy',
    'sad': 'Sad',
    'surprise': 'Surprise',
    'neutral': 'Neutral'
}

# Lazy load liveness detector (using traditional CV-based detector)
_liveness_detector = None

def _get_liveness_detector():
    """Lazy initialization of traditional CV-based liveness detector"""
    global _liveness_detector
    
    if _liveness_detector is None:
        print("[Liveness] Using traditional CV-based liveness detector (LBP + Color + Moiré)")
        from liveness import get_liveness_detector
        _liveness_detector = get_liveness_detector()
    
    return _liveness_detector


def analyze_emotion_and_liveness(img_path):
    """
    Analyze emotion using DeepFace and liveness using traditional CV-based detector.

    args:
        img_path: Path or image array (RGB numpy array expected)
    returns:
        emotion_label: str
        is_real: bool
        liveness_confidence: float (0-1)
        liveness_details: dict (includes 'texture', 'color', 'moire', 'motion', 'overall', 'decision')
    """
    emotion = 'Neutral'
    is_live = False
    liveness_confidence = 0.0
    liveness_details = {}
    
    # 1. Emotion Analysis (DeepFace)
    try:
        # DeepFace anti-spoofing disabled - we use our own CNN-based detector
        analysis = DeepFace.analyze(img_path, actions=['emotion'], anti_spoofing=False, 
                                   enforce_detection=False, silent=True)

        result = analysis[0] if isinstance(analysis, list) else analysis
        em = result.get('dominant_emotion') or (result.get('emotion') or {}).get('dominant')
        emotion = emotion_labels.get(str(em).lower(), 'Neutral') if em else 'Neutral'
    except Exception as exc:
        # Silently handle errors - emotion is optional
        emotion = 'Neutral'
    
    # 2. Liveness Detection (CNN-based with temporal smoothing)
    try:
        detector = _get_liveness_detector()
        is_live, liveness_confidence, liveness_details = detector.analyze(img_path)
    except Exception as exc:
        # On error, default to not live (safer)
        is_live = False
        liveness_confidence = 0.0
        liveness_details = {'error': str(exc)}
        print(f"[Liveness Error] {exc}")
    
    return emotion, is_live, liveness_confidence, liveness_details


# redefine __all__ for explicit exports
__all__ = ['emotion_labels', 'analyze_emotion_and_liveness']
