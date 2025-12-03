"""
Emotion Detection and Liveness Analysis Module
- Emotion: DeepFace with conservative filtering
- Liveness: Multi-method detector with eye blink detection
"""

from deepface import DeepFace
import numpy as np
import cv2

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

def _get_liveness_detector():
    """Lazy initialization of liveness detector with blink detection"""
    global _liveness_detector
    
    if _liveness_detector is None:
        print("[Liveness] Initializing multi-method liveness detector with eye blink detection")
        try:
            from .liveness import LivenessDetector
        except ImportError:
            from liveness import LivenessDetector
        _liveness_detector = LivenessDetector()
    
    return _liveness_detector


def analyze_emotion_and_liveness(face_image, landmarks=None):
    """
    Analyze emotion using enhanced DeepFace configuration and liveness using multi-method detector.
    
    Args:
        face_image: RGB numpy array of face crop
        landmarks: MediaPipe Face Mesh landmarks (required for blink detection)
        
    Returns:
        emotion_label: str - Detected emotion
        is_live: bool - True if face appears to be real
        liveness_confidence: float (0-1) - Confidence score
        liveness_details: dict - Detailed scores from each method
    """
    emotion = 'Neutral'
    is_live = False
    liveness_confidence = 0.0
    liveness_details = {}
    
    # Validate input
    if face_image is None or face_image.size == 0:
        return emotion, False, 0.0, {'error': 'Invalid face image'}
    
    # Ensure RGB format and proper size for emotion detection
    if len(face_image.shape) == 2:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
    elif face_image.shape[2] == 4:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
    
    # 1. Emotion Analysis - DISABLED (not critical for attendance)
    # TODO: Implement proper emotion detection as QoL feature later
    # For now, emotion is not necessary for attendance system security
    emotion = 'Neutral'  # Always neutral - focus on liveness instead
    
    # # COMMENTED OUT: DeepFace emotion detection (unreliable)
    # try:
    #     analysis = DeepFace.analyze(
    #         face_image, 
    #         actions=['emotion'], 
    #         enforce_detection=False, 
    #         silent=True
    #     )
    #     result = analysis[0] if isinstance(analysis, list) else analysis
    #     dominant_emotion = result.get('dominant_emotion', 'neutral')
    #     emotion = emotion_labels.get(str(dominant_emotion).lower(), 'Neutral')
    # except Exception:
    #     emotion = 'Neutral'
    
    # 2. Liveness Detection with Blink Detection
    try:
        detector = _get_liveness_detector()
        is_live, liveness_confidence, liveness_details = detector.analyze(face_image, landmarks)
        
    except Exception as exc:
        # On error, default to not live (safer)
        is_live = False
        liveness_confidence = 0.0
        liveness_details = {'error': str(exc)}
        print(f"[Liveness Error] {exc}")
    
    return emotion, is_live, liveness_confidence, liveness_details


def reset_liveness_detector():
    """Reset liveness detector state for new verification"""
    global _liveness_detector
    if _liveness_detector is not None:
        _liveness_detector.reset()


# Explicit exports
__all__ = ['emotion_labels', 'analyze_emotion_and_liveness', 'reset_liveness_detector']
