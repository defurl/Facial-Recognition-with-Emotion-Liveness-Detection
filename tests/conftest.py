"""Test-wide logging/verbosity shims to keep noisy backends quiet."""
import os

# Suppress TensorFlow / MediaPipe / absl noise in tests
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

try:  # absl logging can spam even with env vars
    import absl.logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
