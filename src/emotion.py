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
        # , detector_backend='retinaface'
        analysis = DeepFace.analyze(img_path, actions=['emotion'], anti_spoofing=True)

        result = analysis[0] if isinstance(analysis, list) else analysis
        em = result.get('dominant_emotion') or (result.get('emotion') or {}).get('dominant')
        emotion = emotion_labels.get(str(em).lower(), 'Neutral') if em else 'Neutral'

        is_real = result.get('is_real')
        return emotion, bool(is_real) if is_real is not None else True
    except Exception as exc:
        print(f"analyze_emotion_and_liveness error: {exc}")
        return 'Neutral', True


# redefine __all__ for explicit exports
__all__ = ['emotion_labels', 'analyze_emotion_and_liveness']
