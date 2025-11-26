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

# Lazy load liveness detector
_liveness_detector = None
_liveness_model_path = None

def _get_liveness_detector():
    """Lazy initialization of liveness detector"""
    global _liveness_detector, _liveness_model_path
    
    if _liveness_detector is None:
        # Try to find liveness model, fallback to backbone
        if os.path.exists('outputs/liveness_detector.pth'):
            _liveness_model_path = 'outputs/liveness_detector.pth'
            print("[Liveness] Using trained liveness model")
        else:
            _liveness_model_path = 'best_face_embedding_model.pth'
            print("[Liveness] Using backbone model (train liveness detector for better accuracy)")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        from liveness_cnn import get_cnn_liveness_detector
        _liveness_detector = get_cnn_liveness_detector(
            model_path=_liveness_model_path,
            device=device
        )
    
    return _liveness_detector


def analyze_emotion_and_liveness(img_path):
    """
    Analyze emotion using DeepFace and liveness using CNN-based detector.

    args:
        img_path: Path or image array (RGB numpy array expected)
    returns:
        emotion_label: str
        is_real: bool
        liveness_confidence: float (0-1)
        liveness_details: dict
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
