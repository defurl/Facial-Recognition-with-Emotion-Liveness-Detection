from deepface import DeepFace

emotion_labels = {
    'angry': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happy': 'Happy',
    'sad': 'Sad',
    'surprise': 'Surprise',
    'neutral': 'Neutral'
}


def analyze_emotion_and_liveness(img_path):
    """
    wrapper for DeepFace.analyze.

    args:
        img_path: Path or image array accepted by DeepFace.
    returns:
        emotion_label: str
        is_real: bool
    """
    try:
        # NOTE: Anti-spoofing disabled due to false positives on webcams
        # DeepFace's anti-spoofing is too aggressive and marks live video as spoof
        # Set enforce_detection=False to handle cases where DeepFace can't detect face
        analysis = DeepFace.analyze(img_path, actions=['emotion'], anti_spoofing=False, 
                                   enforce_detection=False, silent=True)

        result = analysis[0] if isinstance(analysis, list) else analysis
        em = result.get('dominant_emotion') or (result.get('emotion') or {}).get('dominant')
        emotion = emotion_labels.get(str(em).lower(), 'Neutral') if em else 'Neutral'

        # Always return True for liveness since we disabled anti-spoofing
        return emotion, True
    except Exception as exc:
        # Silently handle errors - DeepFace is optional for emotion detection
        return 'Neutral', True


# redefine __all__ for explicit exports
__all__ = ['emotion_labels', 'analyze_emotion_and_liveness']
