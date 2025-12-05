"""
Face Recognition Attendance System - GUI Application

Run this script to launch the interactive GUI for real-time face recognition,
employee management, and attendance tracking.

Usage:
    python app.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import os
# fix duplicate OpenMP runtime on Windows (libiomp5md.dll)
# set before importing libraries that load OpenMP (e.g., torch, cv2)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Suppress TensorFlow GPU warnings and disable GPU usage
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress TF warnings
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # Disable GPU for TensorFlow

# Suppress OpenCV warnings (MSMF errors, etc.)
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

# Disable MediaPipe GPU/hardware acceleration to prevent crashes
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("GLOG_minloglevel", "2")  # Suppress MediaPipe logs

import time
import datetime
import queue
import threading
import asyncio
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    DEVICE, IMG_SIZE, EMPLOYEE_DB_PATH, MODEL_METRIC_PATH,
    OPTIMAL_THRESHOLD_GUI, PROCESS_EVERY_N_FRAMES, CAMERA_INDEX, OUTPUT_DIR,
    PRIMARY_FACE_AREA_WEIGHT, PRIMARY_FACE_CENTER_WEIGHT, USE_MULTI_EMBEDDING,
    MAX_CONCURRENT_FACES, CONFIDENCE_REJECTION_THRESHOLD, UNRECOGNIZED_DISTANCE_MULTIPLIER
)
from src.models import FaceEmbeddingCNN
from src.data_loader import get_transforms
from src.utils import detect_faces, crop_face_with_padding, face_mesh_detector
from src.emotion import analyze_emotion_and_liveness, reset_liveness_detector
from src.blink_detector import BlinkDetector
from src.attendance import AttendanceLogger
from src.explainability import ExplainabilityEngine
from src.deep_knn import knn_predict_with_confidence, get_knn_explanation_text

# Import performance optimized classes with fallback
try:
    from src.performance_optimized_core import (
        AsyncFaceProcessor, VectorizedKNN, VectorizedLivenessDetector,
        MemoryPool, PerformanceMonitor, create_optimized_pipeline
    )
    OPTIMIZED_CORE_AVAILABLE = True
    print("[PERFORMANCE] Optimized core loaded - enhanced performance available")
except ImportError as e:
    print(f"[INFO] Optimized core not available: {e} - using standard processing")
    OPTIMIZED_CORE_AVAILABLE = False
    # Fallback classes
    class AsyncFaceProcessor: 
        def cleanup(self): pass
        async def process_frame_async(self, frame): return []
    class VectorizedKNN: 
        def predict_batch_with_confidence(self, embeddings): return [], []
    class VectorizedLivenessDetector: pass
    class MemoryPool: pass
    class PerformanceMonitor:
        def log_timing(self, name, time_ms): pass
        def get_stats(self): return {}
    def create_optimized_pipeline(*args): return None, None

# Global variables
verification_model = None
employee_db = {}
val_transform = None
# Threshold persistence
import json
_GUI_THRESH_FILE = OUTPUT_DIR / 'gui_threshold.json'

def _load_gui_threshold(default_val: float) -> float:
    try:
        if _GUI_THRESH_FILE.exists():
            data = json.loads(_GUI_THRESH_FILE.read_text())
            val = float(data.get('threshold', default_val))
            return val
    except Exception as e:
        print(f"Warning: could not load GUI threshold: {e}")
    return default_val

def _save_gui_threshold(value: float) -> None:
    try:
        OUTPUT_DIR.mkdir(exist_ok=True)
        _GUI_THRESH_FILE.write_text(json.dumps({'threshold': float(value)}))
    except Exception as e:
        print(f"Warning: could not save GUI threshold: {e}")

OPTIMAL_THRESHOLD_GUI = _load_gui_threshold(OPTIMAL_THRESHOLD_GUI)


def load_model_and_database():
    """Load the trained model and employee database"""
    global verification_model, employee_db, val_transform
    
    print("Loading model and database...")
    
    # Load model
    verification_model = FaceEmbeddingCNN(embedding_dim=256, num_classes=4000).to(DEVICE)
    if MODEL_METRIC_PATH.exists():
        verification_model.load_state_dict(torch.load(MODEL_METRIC_PATH, map_location=DEVICE))
        verification_model.eval()
        print(f"[OK] Loaded model from {MODEL_METRIC_PATH}")
    else:
        print(f"[WARNING] Model not found at {MODEL_METRIC_PATH}")
        print("  Please train the model first using: python scripts/train_metric.py")
    
    # Load transforms
    _, val_transform = get_transforms()
    
    # Load database
    if EMPLOYEE_DB_PATH.exists():
        employee_db = torch.load(EMPLOYEE_DB_PATH)
        print(f"[OK] Loaded {len(employee_db)} employees from database")
        for name in employee_db:
            print(f"    - {name}")
    else:
        print("[INFO] No existing employee database. Starting fresh.")
        employee_db = {}


def select_primary_face(faces, frame_width, frame_height, previous_primary_idx=-1, previous_bbox=None):
    """
    Select primary face from multiple detections using area and centeredness scoring
    with temporal stability to prevent flickering.
    
    Args:
        faces: List of (x, y, w, h) face bounding boxes
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        previous_primary_idx: Index of primary face from previous frame (-1 if none)
        previous_bbox: Bounding box (x, y, w, h) of previous primary face
    
    Returns:
        int: Index of primary face, or -1 if no faces
    """
    if not faces or len(faces) == 0:
        return -1
    
    if len(faces) == 1:
        return 0
    
    try:
        # Calculate frame center
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        frame_diagonal = (frame_width**2 + frame_height**2) ** 0.5
        
        # Find largest face area for normalization
        max_area = 0
        for x, y, w, h in faces:
            area = w * h
            if area > max_area:
                max_area = area
        
        # Helper function to calculate IoU (Intersection over Union)
        def calculate_iou(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            union = w1 * h1 + w2 * h2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Score each face
        scores = []
        for idx, (x, y, w, h) in enumerate(faces):
            area = w * h
            
            # Area score (normalized)
            area_score = area / max_area if max_area > 0 else 0.0
            
            # Centeredness score (distance from center, inverted)
            face_center_x = x + w / 2
            face_center_y = y + h / 2
            distance_from_center = ((face_center_x - frame_center_x)**2 + 
                                   (face_center_y - frame_center_y)**2) ** 0.5
            center_score = 1.0 - (distance_from_center / frame_diagonal)
            center_score = max(0.0, center_score)
            
            # Weighted combination (80% area, 20% centeredness) - increased area weight for stability
            final_score = 0.8 * area_score + 0.2 * center_score
            
            # Add hysteresis bonus: if this face was previously primary and still present, give it a boost
            if previous_bbox is not None:
                iou = calculate_iou((x, y, w, h), previous_bbox)
                if iou > 0.5:  # Same face detected (IoU > 50%)
                    # Add significant stability bonus (20% boost)
                    final_score *= 1.20
            
            scores.append(final_score)
        
        # Return index of highest scoring face
        primary_idx = scores.index(max(scores))
        
        # Debug: Print scoring details when multiple faces
        if len(faces) > 1:
            print(f"[DEBUG] Primary face selection: {len(faces)} faces detected")
            for idx, score in enumerate(scores):
                area = faces[idx][2] * faces[idx][3]
                print(f"  Face {idx}: score={score:.3f}, area={area} {'<-- PRIMARY' if idx == primary_idx else ''}")
        
        return primary_idx
    except Exception as e:
        print(f"Error selecting primary face: {e}")
        return -1 if not faces else 0


def draw_rounded_rectangle(img, pt1, pt2, color, thickness=2, radius=15):
    """Draw a rectangle with rounded corners for modern UI look"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    if thickness < 0:  # Filled
        # Draw filled rectangles and circles for rounded corners
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:  # Outline
        # Draw lines with rounded corners
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_ear_graph(frame, ear_history, ear_threshold=0.5):
    """Draw EAR (Eye Aspect Ratio) graph overlay for blink detection debugging"""
    if len(ear_history) < 1:
        return
    
    # Draw background even with minimal data
    if len(ear_history) < 2:
        # Just draw the graph background with "Collecting data..." message
        h, w = frame.shape[:2]
        graph_x = 10
        graph_y = h - 130
        graph_w = 300
        graph_h = 120
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + graph_w, graph_y + graph_h), 
                     (20, 20, 20), -1)
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + graph_w, graph_y + graph_h), 
                     (100, 100, 100), 2)
        cv2.putText(frame, "EAR Graph - Collecting data...", (graph_x + 5, graph_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return
    
    h, w = frame.shape[:2]
    
    # Graph dimensions (positioned at bottom-left corner)
    graph_x = 10
    graph_y = h - 130
    graph_w = 300
    graph_h = 120
    
    # Background
    cv2.rectangle(frame, (graph_x, graph_y), 
                 (graph_x + graph_w, graph_y + graph_h), 
                 (20, 20, 20), -1)
    cv2.rectangle(frame, (graph_x, graph_y), 
                 (graph_x + graph_w, graph_y + graph_h), 
                 (100, 100, 100), 2)
    
    # Threshold line
    threshold_y = int(graph_y + graph_h - (ear_threshold / 0.8 * graph_h))  # Scale: 0-0.8
    cv2.line(frame, (graph_x, threshold_y), 
            (graph_x + graph_w, threshold_y), 
            (0, 255, 255), 2)
    cv2.putText(frame, f"Threshold: {ear_threshold:.2f}", (graph_x + 5, threshold_y - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    
    # Plot EAR values
    for i in range(1, len(ear_history)):
        x1 = graph_x + int((i - 1) * graph_w / 150)
        x2 = graph_x + int(i * graph_w / 150)
        
        # Scale EAR to graph (0-0.8 range)
        y1 = graph_y + graph_h - int(min(ear_history[i-1], 0.8) / 0.8 * graph_h)
        y2 = graph_y + graph_h - int(min(ear_history[i], 0.8) / 0.8 * graph_h)
        
        # Color: green=open, red=closed
        line_color = (0, 255, 0) if ear_history[i] > ear_threshold else (0, 0, 255)
        cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
    
    # Labels
    current_ear = ear_history[-1] if ear_history else 0.0
    ear_color = (0, 255, 0) if current_ear > ear_threshold else (0, 0, 255)
    cv2.putText(frame, "EAR Over Time", (graph_x + 5, graph_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Current: {current_ear:.3f}", (graph_x + 5, graph_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, ear_color, 1)


def interpolate_color(color1, color2, factor):
    """Smoothly interpolate between two colors for transitions"""
    return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))


class AttendanceSystemGUI:
    """Modern GUI application for face recognition attendance system"""
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Face Recognition Attendance System - xAI Enhanced")

        # Fullscreen / maximize: handle cross-platform safely
        try:
            import platform
            if platform.system() == 'Windows':
                # Maximized window on Windows
                try:
                    self.window.state('zoomed')
                except Exception:
                    pass
            else:
                # On many Linux window managers the '-zoomed' attribute works.
                # Try it, but don't raise if unsupported (avoids TclError).
                try:
                    self.window.attributes('-zoomed', True)
                except Exception:
                    # Fallback: don't force maximize on unknown platforms
                    pass
        except Exception:
            # Be conservative: if platform detection fails, skip maximizing
            pass

        self.window.minsize(1400, 900)
        self.window.configure(bg='#0d1117')  # Dark theme background
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Enable F11 toggle fullscreen
        self.window.bind('<F11>', lambda e: self.toggle_fullscreen())
        self.window.bind('<Escape>', lambda e: self.window.attributes('-fullscreen', False))
        self.fullscreen = False
        
        # Configure styles
        self.setup_styles()
        
        # Core state
        self.cap = None
        self.running = False
        self.multiple_faces_warning = False
        
        # Processing control - optimized for performance
        self.frame_count = 0
        self.PROCESS_EVERY_N_FRAMES = max(5, PROCESS_EVERY_N_FRAMES)  # Process every 5 frames
        self.EMOTION_EVERY_N_FRAMES = 5  # Process emotion/liveness every 5 frames
        self.last_processed_frame = 0
        
        # Multi-face tracking for concurrent processing
        self.face_identities = []  # Store identity for each processed face
        self.face_confidences = []  # Store confidence for each processed face
        self.face_emotions = []  # Store emotion for each processed face
        self.face_liveness = []  # Store liveness for each processed face
        
        # Identity lock system for seamless check-in
        self.identity_lock_buffer = []  # List of (timestamp, identity, confidence) tuples
        self.LOCK_DURATION = 2.0  # Accumulate verifications for 2 seconds (real time) - reduced for faster check-in
        self.LOCK_MIN_VERIFICATIONS = 3  # Require 3 verifications for check-in - reduced for faster response
        self.locked_identity = None  # Currently locked identity
        self.lock_timestamp = 0  # When the identity was locked
        self.LOCK_DISPLAY_TIME = 2.0  # Show locked identity for 2 seconds before auto-reset
        
        # Phase 8: Attendance logger
        self.attendance_logger = AttendanceLogger(
            csv_path=OUTPUT_DIR / 'attendance_log.csv',
            cooldown_minutes=60
        )
        self.last_attendance_message = ""
        self.attendance_attempt_cache = {}  # Track last attempt time per person to prevent repeated I/O
        
        # Registration state tracking
        self.registration_mode = False
        self.registration_name = ""
        self.registration_state = None  # Will hold state machine data during registration
        self.registration_feedback = ""  # Persistent feedback message
        self.registration_feedback_color = (255, 165, 0)  # Default orange
        
        # Thread-safe queue for frames - optimized queue size
        self.frame_queue = queue.Queue(maxsize=1)  # Smaller queue to reduce lag
        
        # Camera lock to prevent race conditions
        self.camera_lock = threading.Lock()
        
        # Thread safety locks
        self.emotion_lock = threading.Lock()
        self.liveness_lock = threading.Lock()
        self.liveness_state_lock = threading.RLock()  # NEW: Protect liveness state variables
        
        # Emotion analysis failure tracking for graceful degradation
        self.emotion_failure_count = 0
        self.emotion_analysis_enabled = True  # Auto-enable for lightweight liveness detection
        
        # Initialize missing attributes
        self.confidence_buffer = []
        self.no_face_frames = 0
        # Last-known display values (prevent AttributeError when display updates run before first detection)
        self.last_identity = "No face detected"
        self.last_emotion = "Neutral"
        self.last_liveness = "Unknown"
        self.last_liveness_confidence = 0.0
        self.last_distance = 0.0
        self.last_confidence = 0.0
        
        # Recognition smoothing to reduce flickering
        self.recognition_history = []  # Store last N recognition results
        self.SMOOTHING_WINDOW = 7  # Increased from 3 to 7 for better stability
        
        # Missing attributes initialization
        self.matched_pose_index = -1  # Initialize pose matching index
        self.ear_canvas = None  # Will be set during UI setup
        
        # EAR detection optimization - Balanced thresholds for reliable blink detection
        self.ear_threshold = 0.45  # Optimized threshold based on your data (was 0.5)
        from collections import deque
        self.ear_history = deque(maxlen=50)  # Store last 50 EAR values (~1.5 seconds at 30fps)
        self.ear_lock = threading.Lock()  # Thread safety for EAR data
        self.ear_baseline = 0.55  # Expected open-eye EAR baseline
        self.ear_blink_sensitivity = 0.75  # Sensitivity multiplier (0.45 = 0.55 * 0.75)
        self.current_ear = 0.0  # Current EAR value
        self.current_blinks = 0  # Current blink count
        
        # Face tracking for consistent identity assignment
        self.face_trackers = {}  # Track face positions across frames
        self.next_face_id = 0
        self.TRACKING_DISTANCE_THRESHOLD = 100  # Max distance to consider same face
        self.TRACKER_TIMEOUT_FRAMES = 10  # Remove tracker after N frames without detection
        
        # ========== MULTI-FACE VERIFICATION SYSTEM ==========
        # Industry Standard: Primary face gets full processing (liveness + verification)
        #                   Secondary faces get verification only
        # TO REVERT: Set ENABLE_MULTI_FACE_VERIFICATION = False
        self.ENABLE_MULTI_FACE_VERIFICATION = True  # Master feature flag
        
        if self.ENABLE_MULTI_FACE_VERIFICATION:
            self.primary_face_id = None  # Primary face gets full processing
            self.face_verification_states = {}  # Per-face verification tracking
            self.face_areas = {}  # Track face sizes for primary selection
            self.PRIMARY_FACE_SELECTION_FRAMES = 30  # Frames to determine primary face
            self.primary_selection_counter = 0
            
            # Per-face verification tracking (lightweight - only verification, not liveness)
            self.face_identities_verified = {}  # face_id -> latest verified identity
            self.face_confidences_verified = {}  # face_id -> latest confidence
            self.face_verification_history = {}  # face_id -> history of verifications
            
            print("[MULTI-FACE] Enhanced multi-face verification enabled (Primary + Secondary)")
        else:
            print("[MULTI-FACE] Using legacy multi-face processing")
        
        # Synchronous blink detection (no threading needed)
        self.face_mesh = face_mesh_detector  # MediaPipe Face Mesh for landmarks
        self.current_landmarks = None  # Store landmarks for blink detection
        
        # Initialize lightweight blink detector and liveness system
        self.blink_detector = BlinkDetector()
        self.verification_start_time = None
        self.lightweight_liveness = True  # Default to lightweight mode
        
        # Spoof detection state tracking
        self.last_spoof_detection_time = 0
        self.SPOOF_WARNING_DISPLAY_TIME = 8.0  # Show spoof warning for 8 seconds (increased)
        self.spoof_warning_shown = False  # Track if warning has been shown
        self.last_emotion_check_frame = 0
        
        # Hysteresis for liveness (reduce flicker from single-frame noise)
        self.consec_spoof_count = 0
        self.consec_real_count = 0
        self.CONSEC_REQUIRED = 6  # Require 6 consecutive same results before declaring spoof (increased)
        
        # EAR (Eye Aspect Ratio) tracking for blink detection visualization
        from collections import deque
        self.ear_history = deque(maxlen=150)  # Store last 150 EAR values (~5 seconds at 30fps)
        self.ear_threshold = 0.5  # Blink detection threshold
        self.ear_lock = threading.Lock()  # Thread safety for EAR data
        self.ear_graph_overlay = None  # Cached graph image
        self.ear_graph_update_counter = 0  # Update graph every N frames
        
        # Recognition statistics
        self.recognition_stats = {
            'total_detections': 0,
            'successful_recognitions': 0,
            'unique_faces_today': set()
        }
        
        # xAI: Explainability engine
        self.explainer = None  # Initialize after model loads
        self.current_face_tensor = None  # Store current face for explanation
        self.current_face_image = None  # Store current face image
        self.current_explanation = None  # Store current explanation data
        self.knn_neighbors = None  # Store kNN neighbor info
        
        # Performance optimized processing pipeline
        self.async_processor = None  # Initialize after model loads
        self.vectorized_knn = None  # Initialize after database loads
        self.performance_monitor = PerformanceMonitor()
        self.processing_batch_buffer = []
        self.batch_processing_active = False
        
        # Animation states for smooth transitions
        self.verification_animation = {
            'active': False,
            'type': None,  # 'success' or 'failure'
            'frame_count': 0,
            'max_frames': 30  # ~1 second at 30fps
        }
        self.box_color_transition = {
            'current_color': (128, 128, 128),
            'target_color': (128, 128, 128),
            'frame': 0,
            'transition_frames': 10
        }
        
        self.setup_ui()

    def update_face_tracking(self, face_positions):
        """Update face tracking to maintain consistent IDs across frames."""
        current_time = time.time()
        
        # Clean up expired trackers
        expired_ids = []
        for face_id, tracker in self.face_trackers.items():
            if current_time - tracker['last_seen'] > self.TRACKER_TIMEOUT_FRAMES / 30.0:  # Assuming 30fps
                expired_ids.append(face_id)
        
        for face_id in expired_ids:
            del self.face_trackers[face_id]
        
        # Calculate center positions for detected faces
        detected_centers = []
        for i, (x, y, w, h) in enumerate(face_positions):
            center_x = x + w // 2
            center_y = y + h // 2
            detected_centers.append((center_x, center_y))
        
        # Assign consistent IDs to detected faces
        face_assignments = {}
        unassigned_faces = list(range(len(detected_centers)))
        
        # First pass: match to existing trackers
        for face_id, tracker in list(self.face_trackers.items()):
            best_match = None
            best_distance = float('inf')
            
            for i in unassigned_faces:
                distance = ((detected_centers[i][0] - tracker['center'][0]) ** 2 + 
                           (detected_centers[i][1] - tracker['center'][1]) ** 2) ** 0.5
                
                if distance < self.TRACKING_DISTANCE_THRESHOLD and distance < best_distance:
                    best_match = i
                    best_distance = distance
            
            if best_match is not None:
                # Update existing tracker
                self.face_trackers[face_id]['center'] = detected_centers[best_match]
                self.face_trackers[face_id]['last_seen'] = current_time
                face_assignments[best_match] = face_id
                unassigned_faces.remove(best_match)
        
        # Second pass: create new trackers for unassigned faces
        for i in unassigned_faces:
            face_id = self.next_face_id
            self.next_face_id += 1
            
            self.face_trackers[face_id] = {
                'center': detected_centers[i],
                'last_seen': current_time,
                'recognition_history': []
            }
            face_assignments[i] = face_id
        
        return face_assignments
    
    def get_face_recognition_history(self, face_id):
        """Get recognition history for a specific face ID."""
        if face_id in self.face_trackers:
            return self.face_trackers[face_id]['recognition_history']
        return []
    
    def update_face_recognition(self, face_id, identity, confidence):
        """Update recognition result for a specific face ID."""
        if face_id in self.face_trackers:
            history = self.face_trackers[face_id]['recognition_history']
            history.append({
                'identity': identity,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
            # Keep only recent history
            if len(history) > self.SMOOTHING_WINDOW:
                history.pop(0)
    
    def match_landmarks_to_face(self, face_bbox, landmarks_results):
        """Improved landmark-to-face matching with size and scale consideration."""
        if not landmarks_results or not landmarks_results.multi_face_landmarks:
            return None
        
        x, y, w, h = face_bbox
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        face_area = w * h
        
        best_landmark_match = None
        best_score = float('inf')  # Lower is better
        
        for landmark_set in landmarks_results.multi_face_landmarks:
            # Get landmark bounding box
            xs = [lm.x for lm in landmark_set.landmark]
            ys = [lm.y for lm in landmark_set.landmark]
            
            # Convert to pixel coordinates (assuming frame dimensions)
            frame_h, frame_w = 480, 640  # Standard resolution
            landmark_xs = [x * frame_w for x in xs]
            landmark_ys = [y * frame_h for y in ys]
            
            # Calculate landmark bounding box
            lm_min_x, lm_max_x = min(landmark_xs), max(landmark_xs)
            lm_min_y, lm_max_y = min(landmark_ys), max(landmark_ys)
            lm_w = lm_max_x - lm_min_x
            lm_h = lm_max_y - lm_min_y
            lm_center_x = (lm_min_x + lm_max_x) / 2
            lm_center_y = (lm_min_y + lm_max_y) / 2
            lm_area = lm_w * lm_h
            
            # Multi-factor matching score
            center_distance = ((face_center_x - lm_center_x) ** 2 + 
                             (face_center_y - lm_center_y) ** 2) ** 0.5
            
            # Normalize by face size
            normalized_distance = center_distance / max(w, h)
            
            # Size similarity factor (closer to 1.0 is better)
            size_ratio = min(face_area, lm_area) / max(face_area, lm_area) if max(face_area, lm_area) > 0 else 0
            
            # Combined matching score (lower is better)
            matching_score = normalized_distance + (1.0 - size_ratio)
            
            # Only consider if reasonable overlap
            if normalized_distance < 1.0 and size_ratio > 0.3:  # Must have reasonable overlap and size
                if matching_score < best_score:
                    best_score = matching_score
                    best_landmark_match = landmark_set
        
        return best_landmark_match
    
    def select_primary_face(self, faces, face_assignments):
        """Select primary face for intensive processing using multiple criteria."""
        if not faces:
            return None
        
        current_time = time.time()
        
        # Update face areas tracking
        for i, (x, y, w, h) in enumerate(faces):
            face_id = face_assignments.get(i, -1)
            if face_id >= 0:
                area = w * h
                self.face_areas[face_id] = area
        
        # If we have a current primary face that's still detected, keep it for stability
        if (self.primary_face_id is not None and 
            self.primary_face_id in [face_assignments.get(i, -1) for i in range(len(faces))]):
            # Check if primary face is still reasonably large
            current_primary_area = self.face_areas.get(self.primary_face_id, 0)
            largest_area = max(self.face_areas.values()) if self.face_areas else 0
            
            # Keep primary if it's at least 70% of the largest face
            if largest_area > 0 and current_primary_area / largest_area >= 0.7:
                return self.primary_face_id
        
        # Select new primary face based on multiple criteria
        face_scores = {}
        frame_center_x, frame_center_y = 320, 240  # Assuming 640x480 frame
        
        for i, (x, y, w, h) in enumerate(faces):
            face_id = face_assignments.get(i, -1)
            if face_id < 0:
                continue
            
            # Criteria for primary face selection
            area = w * h
            center_x, center_y = x + w//2, y + h//2
            
            # Distance from frame center (closer is better)
            center_distance = ((center_x - frame_center_x) ** 2 + (center_y - frame_center_y) ** 2) ** 0.5
            normalized_center_distance = center_distance / 400  # Normalize by max possible distance
            
            # Combined score (higher is better)
            area_score = area / 10000  # Normalize area
            centrality_score = 1.0 - min(normalized_center_distance, 1.0)
            
            # Weight: 70% area, 30% centrality (configurable)
            total_score = (0.7 * area_score + 0.3 * centrality_score)
            face_scores[face_id] = total_score
        
        # Select face with highest score
        if face_scores:
            new_primary_id = max(face_scores.items(), key=lambda x: x[1])[0]
            self.primary_face_id = new_primary_id
            return new_primary_id
        
        return None
        
    def setup_styles(self):
        """Setup dark theme TTK styles with high contrast for professional xAI display"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Dark theme color palette (GitHub Dark inspired)
        # Background: #0d1117, Surface: #161b22, Primary: #58a6ff, Success: #3fb950
        # Warning: #d29922, Danger: #f85149, Text: #c9d1d9
        
        # Configure custom styles for dark theme
        self.style.configure('Title.TLabel', font=('Arial', 20, 'bold'), foreground='#58a6ff', background='#0d1117')
        self.style.configure('Header.TLabel', font=('Arial', 13, 'bold'), foreground='#c9d1d9', background='#0d1117')
        self.style.configure('Status.TLabel', font=('Arial', 11), foreground='#8b949e', background='#161b22')
        self.style.configure('Error.TLabel', font=('Arial', 11, 'bold'), foreground='#f85149', background='#161b22')
        self.style.configure('Info.TLabel', font=('Arial', 9), foreground='#8b949e', background='#161b22')
        self.style.configure('Success.TLabel', font=('Arial', 11, 'bold'), foreground='#3fb950', background='#161b22')
        self.style.configure('Warning.TLabel', font=('Arial', 11, 'bold'), foreground='#d29922', background='#161b22')
        
        # Dark theme button styles with high contrast
        self.style.configure('Primary.TButton', 
                           font=('Arial', 10, 'bold'),
                           background='#58a6ff',
                           foreground='#0d1117',
                           borderwidth=1,
                           relief='flat',
                           padding=(20, 10))
        self.style.map('Primary.TButton',
                      background=[('active', '#79c0ff'), ('pressed', '#388bfd')])
        
        self.style.configure('Success.TButton', 
                           font=('Arial', 10, 'bold'),
                           background='#3fb950',
                           foreground='#0d1117',
                           borderwidth=1,
                           relief='flat',
                           padding=(20, 10))
        self.style.map('Success.TButton',
                      background=[('active', '#56d364'), ('pressed', '#2ea043')])
        
        self.style.configure('Warning.TButton', 
                           font=('Arial', 10, 'bold'),
                           background='#d29922',
                           foreground='#0d1117',
                           borderwidth=1,
                           relief='flat',
                           padding=(20, 10))
        self.style.map('Warning.TButton',
                      background=[('active', '#e2b340'), ('pressed', '#bb8009')])
        
        self.style.configure('Danger.TButton', 
                           font=('Arial', 10, 'bold'),
                           background='#f85149',
                           foreground='#ffffff',
                           borderwidth=1,
                           relief='flat',
                           padding=(20, 10))
        self.style.map('Danger.TButton',
                      background=[('active', '#ff7b72'), ('pressed', '#da3633')])
        
        # xAI button style for explanation features
        self.style.configure('XAI.TButton', 
                           font=('Arial', 10, 'bold'),
                           background='#a371f7',
                           foreground='#ffffff',
                           borderwidth=1,
                           relief='flat',
                           padding=(15, 8))
        self.style.map('XAI.TButton',
                      background=[('active', '#b583f8'), ('pressed', '#8957e5')])
        
        # Disabled button state - dark gray that stands out
        self.style.map('TButton',
                      background=[('disabled', '#95a5a6')],
                      foreground=[('disabled', '#555555')])
        self.style.map('Primary.TButton',
                      background=[('disabled', '#7f8c8d')],
                      foreground=[('disabled', '#34495e')])
        self.style.map('Success.TButton',
                      background=[('disabled', '#7f8c8d')],
                      foreground=[('disabled', '#34495e')])
        
        # LabelFrame styles for dark theme
        self.style.configure('TLabelframe', background='#161b22', borderwidth=1, relief='solid')
        self.style.configure('TLabelframe.Label', 
                           font=('Arial', 11, 'bold'), 
                           foreground='#58a6ff',
                           background='#0d1117')
    
    def setup_ui(self):
        """Setup simplified user-facing UI"""
        # Main container with dark theme
        main_container = tk.Frame(self.window, bg='#0d1117')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 4. Right Panel: Checked-in Identities (New Requirement)
        # Pack RIGHT first so it claims the right side
        right_panel = tk.Frame(main_container, bg='#0d1117', width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Header
        log_header = tk.Frame(right_panel, bg='#161b22', height=40)
        log_header.pack(fill=tk.X, pady=(0, 10))
        tk.Label(log_header, text="RECENT CHECK-INS", font=('Arial', 11, 'bold'), 
                fg='#58a6ff', bg='#161b22').pack(pady=10)
        
        # Listbox for logs
        self.log_listbox = tk.Listbox(right_panel, bg='#161b22', fg='#c9d1d9', 
                                     font=('Arial', 10), borderwidth=0, highlightthickness=0)
        self.log_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # EAR Debug Panel with Canvas for Real-time Graph
        ear_debug_frame = ttk.LabelFrame(right_panel, text="BLINK DETECTION DEBUG")
        ear_debug_frame.pack(fill=tk.X, pady=(0, 10))
        
        ear_container = tk.Frame(ear_debug_frame, bg='#161b22', height=120)
        ear_container.pack(fill=tk.X, padx=10, pady=10)
        ear_container.pack_propagate(False)
        
        # Canvas for EAR visualization
        self.ear_canvas = tk.Canvas(ear_container, bg='#0d0d0d', height=80,
                                   highlightthickness=1, highlightbackground='#444c56')
        self.ear_canvas.pack(fill=tk.X, pady=(0, 5))
        
        # Simple text-based debug for additional info
        self.ear_debug_text = tk.Text(ear_container, bg='#0d0d0d', fg='#c9d1d9', 
                                     height=2, font=('Arial', 8), wrap=tk.WORD,
                                     highlightthickness=1, highlightbackground='#444c56')
        self.ear_debug_text.pack(fill=tk.X, pady=(0, 5))
        
        # EAR status labels
        ear_status_frame = tk.Frame(ear_container, bg='#161b22')
        ear_status_frame.pack(fill=tk.X)
        
        self.ear_current_label = tk.Label(ear_status_frame, text="Current EAR: --", 
                                         bg='#161b22', fg='#c9d1d9', font=('Arial', 8))
        self.ear_current_label.pack(side=tk.LEFT)
        
        self.blink_count_label = tk.Label(ear_status_frame, text="Blinks: 0", 
                                         bg='#161b22', fg='#c9d1d9', font=('Arial', 8))
        self.blink_count_label.pack(side=tk.RIGHT)

        # 5. Employee Management (CRUD) - Bottom of Right Panel
        crud_frame = ttk.LabelFrame(right_panel, text="EMPLOYEE MANAGEMENT")
        crud_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 0))
        
        crud_container = tk.Frame(crud_frame, bg='#161b22')
        crud_container.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(crud_container, text="View Employees", command=self.view_employees).pack(fill=tk.X, pady=(0, 5))
        # ttk.Button(crud_container, text="View Attendance", command=self.view_attendance).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(crud_container, text="Edit Employee", command=self.edit_employee).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(crud_container, text="Delete Employee", command=self.delete_employee, style='Danger.TButton').pack(fill=tk.X)

        # Left Container for Video and Controls
        left_container = tk.Frame(main_container, bg='#0d1117')
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 1. Video Feed (Top/Center)
        video_frame = ttk.LabelFrame(left_container, text="LIVE CAMERA FEED")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Video container
        video_container = tk.Frame(video_frame, bg='#0d0d0d', relief='solid', borderwidth=2,
                                  highlightthickness=0)
        video_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.video_label = tk.Label(video_container, bg='#0d0d0d', 
                                   text="Camera Feed\nClick 'Start Camera' to begin", 
                                   fg='#8b949e', font=('Arial', 14), justify=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Store video display dimensions
        self.video_width = 640
        self.video_height = 480
        
        # 2. Controls (Below Video)
        controls_frame = ttk.LabelFrame(left_container, text="CONTROLS")
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        control_buttons_frame = tk.Frame(controls_frame, bg='#161b22')
        control_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Center the buttons
        button_container = tk.Frame(control_buttons_frame, bg='#161b22')
        button_container.pack(anchor=tk.CENTER)
        
        self.start_button = ttk.Button(button_container, text="START CAMERA", 
                                      command=self.start_camera, style='Primary.TButton', width=20)
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = ttk.Button(button_container, text="STOP CAMERA", 
                                     command=self.stop_camera, state=tk.DISABLED, style='Warning.TButton', width=20)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        self.register_button = ttk.Button(button_container, text="REGISTER NEW USER", 
                                         command=self.start_registration, state=tk.DISABLED, style='Success.TButton', width=20)
        self.register_button.pack(side=tk.LEFT, padx=10)
        
        # Commented out blink detection button
        # self.blink_button = ttk.Button(button_container, text="ENABLE BLINK DETECTION", 
        #                              command=self.enable_blink_detection, state=tk.DISABLED, width=20)
        # self.blink_button.pack(side=tk.LEFT, padx=10)

        # Lightweight detection toggle
        toggle_frame = tk.Frame(controls_frame, bg='#f0f0f0')
        toggle_frame.pack(fill=tk.X, pady=5)
        
        self.lightweight_var = tk.BooleanVar(value=True)  # Default to lightweight
        self.lightweight_checkbox = ttk.Checkbutton(
            toggle_frame, 
            text="⚡ Lightweight Liveness (Blink-only, 30+ FPS)", 
            variable=self.lightweight_var,
            command=self.toggle_liveness_mode
        )
        self.lightweight_checkbox.pack(side=tk.LEFT, padx=10)
        
        # Status label for detection mode
        self.detection_mode_label = tk.Label(
            toggle_frame, 
            text="Mode: Lightweight (Recommended)", 
            font=('Arial', 9, 'italic'),
            fg='green',
            bg='#f0f0f0'
        )
        self.detection_mode_label.pack(side=tk.LEFT, padx=10)

        # 3. Hidden components (to prevent logic errors in existing update methods)
        # These are created but NOT packed into the visible UI
        self._setup_hidden_components()

    def _setup_hidden_components(self):
        """Initialize UI components that are referenced in logic but hidden in this view"""
        hidden_frame = tk.Frame(self.window) # Not packed
        
        # Status
        self.status_text = tk.StringVar(value="")
        self.status_label = ttk.Label(hidden_frame, textvariable=self.status_text)
        self.status_indicator = tk.Label(hidden_frame)
        self.accuracy_display = tk.Label(hidden_frame)
        
        # Detection Info
        self.identity_label = tk.Label(hidden_frame)
        self.emotion_var = tk.StringVar()
        self.liveness_var = tk.StringVar()
        self.liveness_label = tk.Label(hidden_frame)
        self.liveness_confidence_var = tk.StringVar()
        self.distance_var = tk.StringVar()
        self.confidence_var = tk.StringVar()
        
        # Debug
        self.pose_var = tk.StringVar()
        self.threshold_display = tk.Label(hidden_frame)
        
        # Stats
        self.total_det_label = tk.Label(hidden_frame)
        self.success_rate_label = tk.Label(hidden_frame)
        self.unique_faces_label = tk.Label(hidden_frame)
        self.db_size_label = tk.Label(hidden_frame)
        
        # Performance metrics
        self.fps_var = tk.StringVar(value="FPS: --")
        self.latency_var = tk.StringVar(value="Latency: --")
        self.fps_label = tk.Label(hidden_frame, textvariable=self.fps_var)
        self.latency_label = tk.Label(hidden_frame, textvariable=self.latency_var)
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.fullscreen = not self.fullscreen
        self.window.attributes('-fullscreen', self.fullscreen)
    
    def update_stats_display(self):
        """Update session statistics display with visual metrics"""
        total = self.recognition_stats['total_detections']
        success = self.recognition_stats['successful_recognitions']
        unique = len(self.recognition_stats['unique_faces_today'])
        accuracy = (success / total * 100) if total > 0 else 0
        
        # Update individual metric labels
        self.total_det_label.config(text=str(total))
        self.success_rate_label.config(text=f"{accuracy:.1f}%")
        self.unique_faces_label.config(text=str(unique))
        self.db_size_label.config(text=str(len(employee_db)))
        
        # Update performance metrics every 30 frames (roughly once per second)
        if self.frame_count % 30 == 0:
            stats = self.performance_monitor.get_stats()
            
            if 'total' in stats:
                fps = stats['total'].get('fps', 0)
                avg_ms = stats['total'].get('avg_ms', 0)
                self.fps_var.set(f"FPS: {fps:.1f}")
                self.latency_var.set(f"Latency: {avg_ms:.1f}ms")
            else:
                # Fallback fps calculation
                if hasattr(self, 'last_fps_time'):
                    current_time = time.time()
                    fps = 30 / (current_time - self.last_fps_time)
                    self.fps_var.set(f"FPS: {fps:.1f}")
                    self.last_fps_time = current_time
                else:
                    self.last_fps_time = time.time()
    
    def update_debug_panel(self):
        """Update debug panel with verification details"""
        threshold = _load_gui_threshold(OPTIMAL_THRESHOLD_GUI)
        self.threshold_display.config(text=f"Threshold: {threshold:.3f}")
        
        # Update matched pose
        if self.matched_pose_index >= 0:
            pose_names = ["Center", "Left", "Right", "Up", "Down"]
            pose_name = pose_names[self.matched_pose_index] if self.matched_pose_index < 5 else f"Pose {self.matched_pose_index + 1}"
            self.pose_var.set(pose_name)
        else:
            self.pose_var.set("N/A")
        
    def update_debug_display(self):
        """Update debug information display (legacy stats)"""
        threshold = _load_gui_threshold(OPTIMAL_THRESHOLD_GUI)
        debug_text = f"""Recognition Threshold: {threshold:.3f}
Process Every N Frames: {self.PROCESS_EVERY_N_FRAMES}
Model: {"Loaded" if verification_model else "Not Loaded"}
Camera Status: {"Active" if self.running else "Inactive"}
Registration Mode: {"ON" if self.registration_mode else "OFF"}"""
        if hasattr(self, 'debug_text'):
            self.debug_text.set(debug_text)
    
    def update_ear_debug_panel(self):
        """Update the EAR debug panel with balanced blink detection analysis"""
        try:
            # Safely get EAR data
            with self.ear_lock:
                ear_list = list(self.ear_history)
                
            if not ear_list:
                # No data yet
                if hasattr(self, 'ear_current_label'):
                    self.ear_current_label.config(text="Current EAR: --", fg='#8b949e')
                    self.blink_count_label.config(text="Blinks: --", fg='#8b949e')
                return
            
            # Calculate balanced EAR metrics
            current_ear = ear_list[-1]
            recent_ears = ear_list[-5:] if len(ear_list) >= 5 else ear_list  # Last 5 frames
            avg_recent_ear = sum(recent_ears) / len(recent_ears)
            ear_variance = sum((x - avg_recent_ear) ** 2 for x in recent_ears) / len(recent_ears)
            
            # Dynamic threshold based on recent baseline
            if len(ear_list) >= 10:
                baseline_ears = [e for e in ear_list[-10:] if e > self.ear_threshold]  # Open-eye samples
                if baseline_ears:
                    dynamic_baseline = sum(baseline_ears) / len(baseline_ears)
                    adaptive_threshold = dynamic_baseline * self.ear_blink_sensitivity
                    # Smooth threshold changes
                    self.ear_threshold = 0.7 * self.ear_threshold + 0.3 * adaptive_threshold
            
            # Determine blink state with hysteresis
            is_likely_blink = current_ear < self.ear_threshold
            blink_confidence = max(0, (self.ear_threshold - current_ear) / self.ear_threshold) if is_likely_blink else 0
            
            # Update labels with enhanced info
            if hasattr(self, 'ear_current_label'):
                ear_color = '#f85149' if is_likely_blink else '#3fb950'  # Red for blink, Green for open
                self.ear_current_label.config(text=f"EAR: {current_ear:.3f} (Avg: {avg_recent_ear:.3f})", fg=ear_color)
                
                # Get blink count from detector if available
                if hasattr(self, '_fast_blink_detector'):
                    blink_count = self._fast_blink_detector.blink_count
                    confidence_text = f" | Conf: {blink_confidence:.1%}" if is_likely_blink else ""
                    self.blink_count_label.config(text=f"Blinks: {blink_count}{confidence_text}", fg='#58a6ff')
            
            # Enhanced canvas visualization (only if canvas exists)
            if hasattr(self, 'ear_canvas') and self.ear_canvas:
                try:
                    self.ear_canvas.delete("all")  # Clear previous drawing
                    canvas_width = self.ear_canvas.winfo_width()
                    canvas_height = self.ear_canvas.winfo_height()
                    
                    if canvas_width <= 1 or canvas_height <= 1:
                        return  # Canvas not ready yet
                    
                    # Draw background
                    self.ear_canvas.create_rectangle(0, 0, canvas_width, canvas_height, 
                                                   fill='#0d0d0d', outline='#444c56')
                    
                    # Draw dynamic threshold line (adaptive color)
                    threshold_y = canvas_height - (self.ear_threshold / 0.8 * canvas_height)
                    threshold_color = '#d29922' if not is_likely_blink else '#f85149'
                    self.ear_canvas.create_line(0, threshold_y, canvas_width, threshold_y, 
                                              fill=threshold_color, width=2)
                    self.ear_canvas.create_text(5, threshold_y - 10, text=f"Thresh: {self.ear_threshold:.3f}", 
                                              fill=threshold_color, anchor='nw', font=('Arial', 7))
                    
                    # Draw baseline reference (higher line)
                    baseline_y = canvas_height - (self.ear_baseline / 0.8 * canvas_height)
                    self.ear_canvas.create_line(0, baseline_y, canvas_width, baseline_y, 
                                              fill='#58a6ff', width=1, dash=(2, 2))
                    
                    # Plot EAR values with enhanced visualization
                    if len(ear_list) > 1:
                        points = []
                        for i, ear_value in enumerate(ear_list):
                            x = i * canvas_width / max(len(ear_list) - 1, 1)
                            y = canvas_height - (min(ear_value, 0.8) / 0.8 * canvas_height)
                            points.extend([x, y])
                        
                        # Draw the line graph
                        if len(points) >= 4:  # Need at least 2 points (4 coordinates)
                            line_color = '#f85149' if is_likely_blink else '#58a6ff'
                            self.ear_canvas.create_line(points, fill=line_color, width=2, smooth=True)
                        
                        # Enhanced current state indicator
                        last_y = canvas_height - (min(current_ear, 0.8) / 0.8 * canvas_height)
                        point_size = 4 if is_likely_blink else 3
                        point_color = '#f85149' if is_likely_blink else '#3fb950'
                        self.ear_canvas.create_oval(canvas_width - point_size, last_y - point_size, 
                                                  canvas_width + point_size, last_y + point_size, 
                                                  fill=point_color, outline='#ffffff', width=1)
                except Exception as canvas_error:
                    # Canvas-specific error, continue with text updates
                    pass
            
            # Update debug text with detailed metrics
            if hasattr(self, 'ear_debug_text'):
                debug_info = f"Variance: {ear_variance:.4f} | Sensitivity: {self.ear_blink_sensitivity:.2f}\n"
                debug_info += f"State: {'BLINK' if is_likely_blink else 'OPEN'} | Confidence: {blink_confidence:.1%}"
                self.ear_debug_text.delete(1.0, tk.END)
                self.ear_debug_text.insert(tk.END, debug_info)
            
        except Exception as e:
            print(f"[EAR PANEL] Error updating debug panel: {e}")
    
    def update_ear_debug_display(self, current_ear, blink_count):
        """Thread-safe method to update EAR display from worker threads"""
        try:
            # This method is called via window.after() to ensure thread safety
            self.update_ear_debug_panel()
        except Exception as e:
            print(f"[EAR DISPLAY] Error updating EAR debug display: {e}")
    
    def _open_camera_with_fallback(self):
        """Try camera indices with fallback options"""
        # Try multiple camera indices (try configured index first)
        preferred_indices = [CAMERA_INDEX, 0, 1, 2]

        for idx in preferred_indices:
            cap = None
            try:
                print(f"Trying camera {idx}...")
                # Use platform-default backend; on Linux cv2 will pick V4L2. If a backend is required,
                # consider passing cv2.CAP_V4L2 as the second arg.
                cap = cv2.VideoCapture(idx)

                # VideoCapture always returns an object; guard by checking isOpened
                if not cap or not cap.isOpened():
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    continue

                # Wait for camera to initialize
                time.sleep(0.8)

                # Test if we can actually read a frame
                try:
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and getattr(test_frame, 'size', 0) > 0:
                        print(f"[OK] Successfully opened camera {idx}")
                        return cap
                except Exception as read_err:
                    print(f"  Frame read error on camera {idx}: {read_err}")

                # Release and don't retry to avoid memory issues
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                time.sleep(0.3)

            except Exception as e:
                print(f"  Error with camera {idx}: {e}")
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                continue
        
        print("[ERROR] Failed to open any camera")
        return None

    def toggle_liveness_mode(self):
        """Toggle between lightweight and heavy liveness detection"""
        self.lightweight_liveness = self.lightweight_var.get()
        
        if self.lightweight_liveness:
            mode_text = "Mode: Lightweight (Recommended)"
            mode_color = "green"
            print("[LIVENESS] Switched to LIGHTWEIGHT mode (blink-only, high performance)")
        else:
            mode_text = "Mode: Heavy (Multi-method analysis, slower)"
            mode_color = "orange"
            print("[LIVENESS] Switched to HEAVY mode (texture+color+blink analysis, lower performance)")
        
        self.detection_mode_label.config(text=mode_text, fg=mode_color)
        
        # Reset verification timer when switching modes
        self.verification_start_time = None
        if hasattr(self, 'blink_detector') and self.blink_detector:
            try:
                self.blink_detector.reset()
                print(f"[LIVENESS] Blink detector reset for {mode_text}")
            except Exception as e:
                print(f"[LIVENESS] Could not reset blink detector: {e}")

    def enable_blink_detection(self):
        """Safely enable blink detection after camera stabilizes"""
        if not self.running:
            print("[BLINK] Camera must be running first!")
            return
            
        self.emotion_analysis_enabled = True
        self.emotion_failure_count = 0  # Reset failure count
        self.blink_button.config(text="BLINK DETECTION ON", state=tk.DISABLED, style='Success.TButton')
        print("[BLINK] Blink detection enabled")
    
    def disable_blink_detection_ui(self):
        """Update UI when blink detection gets disabled due to errors"""
        self.blink_button.config(text="ENABLE BLINK DETECTION", state=tk.NORMAL, style='TButton')
        print("[BLINK] UI updated - blink detection disabled")
    
    def start_camera(self):
        """Start camera and begin processing"""
        # Update status to show we're trying
        self.status_text.set("● Initializing camera...")
        self.window.update_idletasks()
        print("[CAMERA] Starting camera initialization...")
        
        self.cap = self._open_camera_with_fallback()
        if not self.cap or not self.cap.isOpened():
            error_msg = (
                "Failed to open camera.\n\n"
                "Troubleshooting steps:\n"
                "1. Check if camera is connected\n"
                "2. Close other apps using the camera\n"
                "3. Try unplugging and replugging the camera\n"
                "4. Check Windows privacy settings (Camera access)\n"
                "5. Try restarting the application"
            )
            print("[ERROR] Camera initialization failed")
            messagebox.showerror("Camera Error", error_msg)
            self.status_text.set("● Camera initialization failed. Check connection.")
            return
        
        print("[CAMERA] Camera opened successfully")
        
        # Optimize capture settings
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.register_button.config(state=tk.NORMAL)
        # self.blink_button.config(state=tk.NORMAL)  # Commented out
        
        # Reset confidence buffer when camera starts
        self.confidence_buffer.clear()
        self.no_face_frames = 0
        
        # Update UI indicators
        self.status_text.set("● Camera started. Face recognition active...")
        self.status_indicator.config(fg='#3fb950')  # Green
        
        # Initialize explainer if not done
        if self.explainer is None and verification_model is not None:
            try:
                self.explainer = ExplainabilityEngine(verification_model, DEVICE)
                print("[OK] Explainability engine initialized")
            except Exception as e:
                print(f"[WARNING] Failed to initialize explainer: {e}")
        
        # Initialize optimized processing pipeline
        if verification_model is not None and OPTIMIZED_CORE_AVAILABLE:
            try:
                self.async_processor, self.vectorized_knn = create_optimized_pipeline(
                    verification_model, DEVICE, employee_db
                )
                if self.async_processor is not None:
                    self.status_text.set("● Camera started with OPTIMIZED processing pipeline...")
                    print("[PERFORMANCE] Optimized pipeline initialized - expect 3-4x faster processing")
                else:
                    self.status_text.set("● Camera started with standard processing...")
            except Exception as e:
                print(f"[WARNING] Could not initialize optimized pipeline: {e}")
                self.async_processor = None
                self.vectorized_knn = None
                self.status_text.set("● Camera started with standard processing...")
        else:
            self.status_text.set("● Camera started with standard processing...")
        
        self.video_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.video_thread.start()
        self.update_display()
        
        # Blink detection disabled
        # self.window.after(2000, self.enable_blink_detection)  # Enable after 2 seconds
    
    def handle_camera_failure(self):
        """Handle camera failure gracefully"""
        if self.running:
            self.status_text.set("● Camera error detected. Stopping camera.")
            messagebox.showwarning(
                "Camera Connection Lost",
                "Camera connection was lost or too many errors occurred.\n\n"
                "Click 'Start' to retry."
            )
            self.stop_camera()
    
    def stop_camera(self):
        """Stop camera and clean up"""
        self.running = False
        
        # Wait for threads to finish gracefully
        # No emotion thread to wait for (using synchronous detection)
        
        # Wait for capture thread to finish
        time.sleep(0.2)
        
        # Cleanup optimized components
        if self.async_processor is not None:
            self.async_processor.cleanup()
            self.async_processor = None
        self.vectorized_knn = None
        
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
                self.cap = None  # Clear reference to prevent reuse
            except Exception as e:
                print(f"Error releasing camera: {e}")
                self.cap = None
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.register_button.config(state=tk.DISABLED)
        # Blink button disabled by default
        
        # Update UI indicators
        self.status_text.set("● Camera stopped. Click 'Start' to resume.")
        self.status_indicator.config(fg='#f85149')  # Red
        self.video_label.config(image='', text="Camera Feed\nStopped", 
                               fg='#8b949e', font=('Arial', 14), justify=tk.CENTER)
        
        # Reset detection displays
        self.identity_label.config(text="No face detected", bg='#1f6feb', fg='#ffffff')
        self.emotion_var.set("Neutral")
        self.liveness_var.set("Unknown")
        self.liveness_confidence_var.set("")
        self.distance_var.set("N/A")
        
        # Reset performance displays
        self.fps_var.set("FPS: --")
        self.latency_var.set("Latency: --")
        self.liveness_label.config(fg='#58a6ff')
        self.confidence_var.set("N/A")
        
        self.update_stats_display()
        self.update_debug_display()
    
    def start_registration(self):
        """Start employee registration with enhanced dialog"""
        # Create custom registration dialog
        dialog = tk.Toplevel(self.window)
        dialog.title("👤 Register New Employee")
        dialog.geometry("400x400")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.window.winfo_rootx() + 50, self.window.winfo_rooty() + 50))
        
        # Header
        header = tk.Frame(dialog, bg='#3498db', height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="👤 Register New Employee", font=('Arial', 14, 'bold'), 
                fg='white', bg='#3498db').pack(pady=15)
        
        # Content
        content = tk.Frame(dialog, bg='#f0f0f0')
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(content, text="Enter employee name:", font=('Arial', 11, 'bold'), 
                bg='#f0f0f0').pack(anchor=tk.W, pady=(0, 10))
        
        name_var = tk.StringVar()
        entry = tk.Entry(content, textvariable=name_var, font=('Arial', 12), width=30)
        entry.pack(fill=tk.X, pady=(0, 15))
        entry.focus()
        
        # Instructions
        instructions = tk.Frame(content, bg='#e8f5e8', relief='solid', borderwidth=1)
        instructions.pack(fill=tk.X, pady=(0, 15), ipadx=10, ipady=0)
        instructions.pack_propagate(True)

        tk.Label(instructions, text="📋 Registration Instructions:", 
                font=('Arial', 10, 'bold'), fg='#27ae60', bg='#e8f5e8').pack(anchor=tk.W, padx=10, pady=(10, 5))

        inst_text = """• Position yourself 2-3 feet from camera (closer = better quality)\n• Look directly at the camera\n• Make sure you have good lighting\n• Keep your face centered and still\n• Follow the on-screen pose instructions\n• Hold each pose steady when prompted\n• Registration captures 3 optimized poses for stationary setup"""

        tk.Label(instructions, text=inst_text, font=('Arial', 9), 
                fg='#2d5a2d', bg='#e8f5e8', justify=tk.LEFT, wraplength=350).pack(anchor=tk.W, padx=10, pady=(0, 10))
        
        def start_registration_process():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("❌ Invalid Input", "Please enter a valid name.")
                return
            
            if name in employee_db:
                messagebox.showerror("❌ Duplicate Employee", f"Employee '{name}' is already registered!")
                return
            
            # Import config constants
            try:
                from config import (
                    REGISTRATION_POSES_QUICK, REGISTRATION_INSTRUCTIONS_QUICK
                )
                
                # OPTIMIZED: Use 3 poses for stationary camera (center, left, right)
                # Skip up/down as camera is stationary - focus on side angles for better embedding separation
                poses = REGISTRATION_POSES_QUICK  # ["center", "right", "left"]
                instructions = REGISTRATION_INSTRUCTIONS_QUICK  # ["Face forward", "Turn left", "Turn right"]
                
                self.registration_mode = True
                self.registration_name = name
                self.registration_state = {
                    'step': 0,
                    'poses_required': poses,
                    'instructions': instructions,
                    'frames': [],
                    'embeddings': [],
                    'hold_frames': 0,
                    'last_check_frame': 0,
                    'completing': False  # Flag to prevent multiple completion calls
                }
                # Clear any previous feedback
                self.registration_feedback = ""
                self.status_text.set(f"🔵 Step 1/{len(poses)}: {instructions[0]} (Position closer for better quality)")
                dialog.destroy()
            except Exception as e:
                print(f"Registration initialization error: {e}")
                messagebox.showerror("Error", f"Failed to start registration: {e}")
                return
        
        # Buttons
        button_frame = tk.Frame(content, bg='#f0f0f0')
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="✓ Start Registration", command=start_registration_process).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="✗ Cancel", command=dialog.destroy).pack(side=tk.LEFT)
        
        # Bind Enter key
        dialog.bind('<Return>', lambda e: start_registration_process())
    
    def complete_registration(self):
        """Show thumbnail confirmation dialog and save embeddings"""
        if not self.registration_state or len(self.registration_state['embeddings']) == 0:
            messagebox.showerror("Error", "No registration data available")
            self.registration_mode = False
            return
        
        dialog = tk.Toplevel(self.window)
        dialog.title("✓ Review Registration")
        dialog.geometry("600x600")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Header
        header = tk.Frame(dialog, bg='#27ae60', height=60)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text=f"✓ Review: {self.registration_name}",
                font=('Arial', 14, 'bold'), bg='#27ae60', fg='white').pack(pady=15)
        
        # Thumbnail grid
        canvas_frame = tk.Frame(dialog, bg='#f0f0f0')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        num_poses = len(self.registration_state['frames'])
        cols = 3
        
        for idx, (frame, instruction) in enumerate(zip(self.registration_state['frames'], 
                                                       self.registration_state['instructions'])):
            row = idx // cols
            col = idx % cols
            
            # Convert frame to PhotoImage
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            pil_img = pil_img.resize((150, 150))
            photo = ImageTk.PhotoImage(pil_img)
            
            # Create thumbnail container
            thumb_frame = tk.Frame(canvas_frame, bg='white', relief=tk.RAISED, borderwidth=2)
            thumb_frame.grid(row=row, column=col, padx=10, pady=10)
            
            # Image label
            img_label = tk.Label(thumb_frame, image=photo, bg='white')
            img_label.image = photo  # Keep reference
            img_label.pack()
            
            # Pose label
            tk.Label(thumb_frame, text=instruction, font=('Arial', 8), 
                    bg='white', fg='#2c3e50').pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(dialog, bg='#f0f0f0')
        button_frame.pack(side=tk.BOTTOM, pady=20)
        
        def save_to_database():
            try:
                # Save all embeddings as list
                employee_db[self.registration_name] = self.registration_state['embeddings']
                torch.save(employee_db, EMPLOYEE_DB_PATH)
                
                # Save image copies of captured frames
                identities_dir = Path("identities")
                identities_dir.mkdir(exist_ok=True)
                
                # Create user-specific folder
                user_dir = identities_dir / self.registration_name
                user_dir.mkdir(exist_ok=True)
                
                # Save each captured frame as an image
                for idx, frame in enumerate(self.registration_state['frames']):
                    # Generate filename with frame number
                    filename = f"frame_{idx+1:02d}.jpg"
                    image_path = user_dir / filename
                    
                    # Save the frame as JPEG image
                    success = cv2.imwrite(str(image_path), frame)
                    if success:
                        print(f"Saved frame {idx+1} to: {image_path}")
                    else:
                        print(f"Failed to save frame {idx+1} to: {image_path}")
                
                print(f"[OK] Registered '{self.registration_name}' with {len(self.registration_state['embeddings'])} poses")
                print(f"[OK] Saved {len(self.registration_state['frames'])} images to: {user_dir}")
                self.status_text.set(f"[OK] Registered '{self.registration_name}' successfully!")
                messagebox.showinfo("Success", f"Employee '{self.registration_name}' registered!\nImages saved to: identities/{self.registration_name}/")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
                print(f"Registration save error: {e}")
            finally:
                self.registration_mode = False
                self.registration_state = None
                # Reset spoof detection for next user
                self.reset_spoof_detection("registration completed")
        
        def cancel_registration():
            self.registration_mode = False
            self.registration_state = None
            self.status_text.set("Registration cancelled")
            dialog.destroy()
        
        ttk.Button(button_frame, text="✓ Save to Database", command=save_to_database,
                  style='Success.TButton', width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="✗ Cancel", command=cancel_registration,
                  style='Danger.TButton', width=15).pack(side=tk.LEFT, padx=5)
    
    def reset_spoof_detection(self, reason="identity change"):
        """Reset spoof detection state for new verification cycle"""
        try:
            print(f"[SPOOF] Resetting spoof detection state - reason: {reason}")
            with self.liveness_state_lock:
                self.last_liveness = 'Unknown'
                self.last_liveness_confidence = 0.0
                self.spoof_warning_shown = False
                self.last_spoof_detection_time = 0
                self.consec_spoof_count = 0
                self.consec_real_count = 0
            
            # Smart blink detector reset: Allow reset for next user, but prevent rapid resets during verification
            if hasattr(self, 'blink_detector') and self.blink_detector:
                current_time = time.time()
                
                # For identity lock timeout (next user), ensure minimum gap between resets
                if reason == "identity lock timeout":
                    if hasattr(self.blink_detector, 'last_reset_time'):
                        time_since_reset = current_time - getattr(self.blink_detector, 'last_reset_time', 0)
                        if time_since_reset > 3.0:  # Minimum 3 seconds between user resets
                            self.blink_detector.reset()
                            self.blink_detector.last_reset_time = current_time
                            print(f"[BLINK] Reset for next user (gap: {time_since_reset:.1f}s)")
                        else:
                            print(f"[BLINK] Skip rapid reset - next user can continue current verification")
                    else:
                        self.blink_detector.reset()
                        self.blink_detector.last_reset_time = current_time
                        print(f"[BLINK] Initial reset for next user")
                else:
                    # For other reasons, use longer cooldown to prevent interruption
                    if hasattr(self.blink_detector, 'last_reset_time'):
                        time_since_reset = current_time - getattr(self.blink_detector, 'last_reset_time', 0)
                        if time_since_reset > 10.0:  # Longer cooldown for other resets
                            self.blink_detector.reset()
                            self.blink_detector.last_reset_time = current_time
                            print(f"[BLINK] Reset due to: {reason}")
                    else:
                        self.blink_detector.reset()
                        self.blink_detector.last_reset_time = current_time
                        print(f"[BLINK] Initial reset due to: {reason}")
            
            # Reset external liveness detector
            from src.emotion import reset_liveness_detector
            reset_liveness_detector()
            
            # Reset verification timer
            self.verification_start_time = None
            
        except Exception as e:
            print(f"[SPOOF] Error resetting spoof detection: {e}")
    
    def capture_frames(self):
        """Optimized frame capture with async processing pipeline"""
        consecutive_errors = 0
        max_consecutive_errors = 30
        
        # Initialize async event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.running:
            try:
                # Check if camera is still valid
                if self.cap is None or not self.cap.isOpened():
                    print("[ERROR] Camera is no longer available")
                    self.window.after(0, lambda: self.handle_camera_failure())
                    break
                
                # Thread-safe camera read with lock
                with self.camera_lock:
                    ret, frame = self.cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    consecutive_errors += 1
                    if consecutive_errors == 1:  # Log first error
                        frame_info = 'None' if frame is None else f'shape={frame.shape}' if frame is not None else 'unknown'
                        print(f"[ERROR] Frame read failed: ret={ret}, frame={frame_info}")
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[ERROR] Too many consecutive frame read errors ({consecutive_errors}). Stopping camera.")
                        self.window.after(0, lambda: self.handle_camera_failure())
                        break
                    time.sleep(0.01)
                    continue
                
                # Reset error counter on successful read
                consecutive_errors = 0
                
                # Debug: Log successful frame capture periodically
                if self.frame_count % 300 == 0:  # Every 10 seconds at 30fps
                    print(f"[DEBUG] Frame capture OK: {frame.shape}, frame #{self.frame_count}")
                
            except Exception as e:
                consecutive_errors += 1
                print(f"Frame capture error: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[ERROR] Too many errors. Stopping camera.")
                    self.window.after(0, lambda: self.handle_camera_failure())
                    break
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            self.frame_count += 1
            
            # Performance timing start
            self.process_start = time.time()
            
            # Clear previous frame's face data
            if self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0:
                self.face_identities.clear()
                self.face_confidences.clear()
                self.face_emotions.clear()
                self.face_liveness.clear()
            
            if self.registration_mode and self.registration_state:
                state = self.registration_state
                current_step = state['step']
                
                if current_step >= len(state['poses_required']):
                    # All poses captured, complete registration (once only)
                    if not state.get('completing', False):
                        state['completing'] = True
                        self.window.after(100, self.complete_registration)
                    # Don't process any more frames during completion
                    continue
                
                # Clean, centered registration overlay above camera feed
                h, w = frame.shape[:2]
                instruction_text = state['instructions'][current_step]
                
                # Use sharp, modern font (FONT_HERSHEY_DUPLEX for sharper text)
                font = cv2.FONT_HERSHEY_DUPLEX
                
                # Calculate text size for instruction
                (inst_w, inst_h), _ = cv2.getTextSize(instruction_text, font, 0.9, 2)
                
                # Center-top position with more breathing room
                card_w = min(w - 100, 500)
                card_h = 80
                x_offset = (w - card_w) // 2
                y_offset = 20
                
                # Draw modern card with high contrast
                overlay = frame.copy()
                draw_rounded_rectangle(overlay, (x_offset, y_offset), 
                                     (x_offset + card_w, y_offset + card_h), 
                                     (26, 26, 46), -1, radius=12)  # Dark background #1a1a2e
                cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
                
                # Circular progress indicator on left side
                progress = current_step / len(state['poses_required'])
                circle_x = x_offset + 50
                circle_y = y_offset + card_h // 2
                radius = 28
                
                # Progress circle background
                cv2.circle(frame, (circle_x, circle_y), radius, (60, 60, 80), -1)
                # Progress arc
                angle = int(360 * progress)
                cv2.ellipse(frame, (circle_x, circle_y), (radius - 3, radius - 3), 
                           -90, 0, angle, (46, 204, 113), 4)  # #2ecc71 green
                # Progress number with high contrast
                progress_text = f"{current_step + 1}/{len(state['poses_required'])}"
                (prog_w, prog_h), _ = cv2.getTextSize(progress_text, font, 0.6, 1)
                cv2.putText(frame, progress_text, 
                           (circle_x - prog_w//2, circle_y + prog_h//2), 
                           font, 0.6, (255, 255, 255), 1)
                
                # Instruction text - centered with high contrast white (lighter weight)
                text_x = x_offset + 100
                text_y = y_offset + (card_h + inst_h) // 2
                cv2.putText(frame, instruction_text, (text_x, text_y), 
                           font, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Simple feedback text at bottom - minimal processing
                if hasattr(self, 'registration_feedback') and self.registration_feedback:
                    cv2.putText(frame, self.registration_feedback, (20, h - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.registration_feedback_color, 2, cv2.LINE_AA)
                
                # Check every 30 frames (~1 second at 30fps) to reduce processing load and flickering
                if self.frame_count - state['last_check_frame'] >= 30:
                    state['last_check_frame'] = self.frame_count
                    
                    faces = detect_faces(frame)
                    if len(faces) == 0:
                        self.registration_feedback = "No face detected"
                        self.registration_feedback_color = (231, 76, 60)
                        state['hold_frames'] = 0
                    else:
                        x, y, w, h = faces[0]
                        cropped_face = crop_face_with_padding(frame, x, y, w, h)
                        
                        if cropped_face.size == 0 or cropped_face.shape[0] < 50:
                            self.registration_feedback = "Face too small"
                            self.registration_feedback_color = (231, 76, 60)
                            state['hold_frames'] = 0
                        else:
                            # Import quality check functions safely
                            try:
                                from utils import check_image_blur, check_image_lighting, estimate_head_pose_angles, validate_pose_for_target
                                
                                # ENHANCED: Quality checks optimized for closer distance
                                blur_var, blur_ok, blur_msg = check_image_blur(cropped_face, threshold=50)  # Slightly higher blur threshold
                                brightness, contrast, lighting_ok, lighting_msg = check_image_lighting(cropped_face, 30, 220, 35)  # Tighter lighting control
                                
                                # Face size guidance for optimal distance
                                face_size = min(cropped_face.shape[0], cropped_face.shape[1])
                                optimal_size = face_size >= 150  # Encourage larger faces
                                size_feedback = "" if optimal_size else " (Move closer for better quality)"
                                
                                # Pose validation
                                target_pose = state['poses_required'][state['step']]
                                yaw, pitch, roll, _, _, _ = estimate_head_pose_angles(cropped_face)
                                pose_ok, _, pose_feedback = validate_pose_for_target(yaw, pitch, target_pose, is_strict=False)
                            except Exception as util_e:
                                print(f"Warning: Quality check error: {util_e}")
                                # Default to accepting the frame if utils fail
                                blur_ok, lighting_ok, pose_ok = True, True, True
                                pose_feedback = "Processing..."
                            
                            if blur_ok and lighting_ok and pose_ok and optimal_size:
                                state['hold_frames'] += 1
                                remaining = 3 - state['hold_frames']
                                
                                if remaining > 0:
                                    self.registration_feedback = f"Perfect! Hold steady... {remaining}{size_feedback}"
                                    self.registration_feedback_color = (0, 255, 0)
                                else:
                                    # Capture this pose
                                    self.registration_feedback = "Captured!"
                                    self.registration_feedback_color = (0, 255, 0)
                                    
                                    try:
                                        cropped_face_resized = cv2.resize(cropped_face, (IMG_SIZE, IMG_SIZE))
                                        rgb = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2RGB)
                                        pil_image = Image.fromarray(rgb).convert('RGB')
                                        image_tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)
                                        
                                        with torch.no_grad():
                                            embedding = verification_model(image_tensor, mode='metric')
                                        
                                        state['frames'].append(cropped_face_resized.copy())
                                        state['embeddings'].append(embedding.cpu())
                                        state['step'] += 1
                                        state['hold_frames'] = 0
                                        
                                        if state['step'] < len(state['poses_required']):
                                            self.status_text.set(f"🔵 Step {state['step'] + 1}/{len(state['poses_required'])}: {state['instructions'][state['step']]}")
                                    except Exception as e:
                                        print(f"Capture error: {e}")
                                        cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 100), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            else:
                                state['hold_frames'] = 0
                                
                                if not optimal_size:
                                    self.registration_feedback = "Move closer - face should be larger in frame"
                                    self.registration_feedback_color = (255, 165, 0)
                                elif not pose_ok:
                                    self.registration_feedback = f"Adjust Pose: {pose_feedback}"
                                    self.registration_feedback_color = (255, 165, 0)
                                elif not blur_ok:
                                    self.registration_feedback = f"Too blurry ({blur_var:.1f}) - hold still"
                                    self.registration_feedback_color = (255, 165, 0)
                                elif not lighting_ok:
                                    self.registration_feedback = f"Adjust lighting (brightness: {brightness:.0f}, contrast: {contrast:.0f})"
                                    self.registration_feedback_color = (255, 165, 0)
                else:
                    # Initialize feedback if not set
                    if not hasattr(self, 'registration_feedback'):
                        self.registration_feedback = ""            # ========== OPTIMIZED ASYNC PROCESSING ==========
            process_start = time.time()
            
            # Use optimized async processor if available
            if self.async_processor is not None:
                try:
                    # Run async processing in the loop
                    results = loop.run_until_complete(
                        self.async_processor.process_frame_async(frame)
                    )
                    
                    # Process results efficiently
                    if results:
                        processed_frame = self.process_async_results(results, frame)
                        self.queue_frame_for_display(processed_frame)
                    
                    process_time = (time.time() - process_start) * 1000
                    self.performance_monitor.log_timing('total', process_time)
                    
                    if self.frame_count % 30 == 0:
                        stats = self.performance_monitor.get_stats()
                        print(f"[PERFORMANCE] Async processing: {process_time:.1f}ms | FPS: {stats['total']['fps']:.1f}")
                    
                    continue  # Skip legacy processing
                except Exception as e:
                    print(f"[ASYNC ERROR] Falling back to legacy processing: {e}")
            
            # ========== LEGACY PROCESSING (Fallback) ==========
            detect_start = time.time()
            faces = detect_faces(frame)
            detect_time = (time.time() - detect_start) * 1000
            
            # Update face tracking for consistent identity assignment
            face_assignments = self.update_face_tracking(faces)
            
            if self.frame_count % 30 == 0:
                print(f"[LEGACY] Face Detection: {detect_time:.2f}ms | Faces: {len(faces)} | Tracked: {len(self.face_trackers)}")
            
            h, w = frame.shape[:2]
            self.multiple_faces_warning = len(faces) > MAX_CONCURRENT_FACES
            faces_to_process = faces[:MAX_CONCURRENT_FACES]
            
            # Multi-face verification: Select primary face for intensive processing
            if self.ENABLE_MULTI_FACE_VERIFICATION and len(faces) > 0:
                try:
                    selected_primary_id = self.select_primary_face(faces, face_assignments)
                    if self.frame_count % 30 == 0 and selected_primary_id != self.primary_face_id:
                        print(f"[PRIMARY] Selected face ID {selected_primary_id} as primary (from {len(faces)} faces)")
                    self.primary_face_id = selected_primary_id
                except Exception as e:
                    print(f"[PRIMARY] Error in face selection, falling back to face 0: {e}")
                    self.primary_face_id = face_assignments.get(0, 0) if faces else None
            
            # Debug: Log multi-face processing when multiple faces detected
            if len(faces) > 1 and self.frame_count % 30 == 0:
                print(f"[MULTI-FACE] Processing {len(faces_to_process)} out of {len(faces)} detected faces")
                for i, (fx, fy, fw, fh) in enumerate(faces):
                    face_id = face_assignments.get(i, -1)
                    print(f"  Face {i} (ID:{face_id}): bbox=({fx}, {fy}, {fw}, {fh}) center=({fx + fw//2}, {fy + fh//2}) area={fw*fh}")
            elif len(faces) == 0 and self.frame_count % 60 == 0:  # Less frequent for no faces
                print(f"[FACE-DETECT] No faces detected at frame {self.frame_count}")

            for face_idx, (x, y, w, h) in enumerate(faces_to_process):
                is_processing_face = face_idx < MAX_CONCURRENT_FACES
                current_face_id = face_assignments.get(face_idx, -1)  # Get tracked face ID
                
                # Determine processing type: primary (full) vs secondary (verification only)
                if self.ENABLE_MULTI_FACE_VERIFICATION:
                    is_primary_face = (current_face_id == self.primary_face_id)
                    is_secondary_face = (not is_primary_face and current_face_id >= 0)
                else:
                    # Legacy mode: first face is primary, others are secondary
                    is_primary_face = (face_idx == 0)
                    is_secondary_face = (face_idx > 0)
                
                # Color coding: Primary=Green, Secondary=Orange, Unknown=Red
                if is_primary_face:
                    box_color = (0, 255, 0)  # Green for primary
                elif is_secondary_face:
                    box_color = (255, 165, 0)  # Orange for secondary
                else:
                    box_color = (0, 0, 255)  # Red for untracked

                # Skip verification during registration mode - always show as unregistered
                if self.registration_mode:
                    if face_idx == 0:  # Only first face for registration
                        self.last_identity = "Registering..."
                        box_color = (255, 165, 0)  # Orange for registration
                # Process all faces within limit for verification
                elif is_processing_face and self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0:
                    print(f"[VERIFICATION] Processing face {face_idx} at frame {self.frame_count}")
                    # ========== TIMING: Processing Frame ==========
                    process_start = time.time()
                    
                    crop_start = time.time()
                    cropped_face = crop_face_with_padding(frame, x, y, w, h)
                    crop_time = (time.time() - crop_start) * 1000

                    if cropped_face.size > 0 and cropped_face.shape[0] >= 50:
                        resize_start = time.time()
                        cropped_face_resized = cv2.resize(cropped_face, (IMG_SIZE, IMG_SIZE))
                        resize_time = (time.time() - resize_start) * 1000

                        # Extract face landmarks (only for primary face to save performance)
                        face_landmarks = None
                        if is_primary_face:  # Only primary face gets landmark processing
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            landmarks_results = face_mesh_detector.process(frame_rgb)
                            
                            # Use improved landmark matching
                            face_landmarks = self.match_landmarks_to_face((x, y, w, h), landmarks_results)
                            
                            if face_landmarks and self.frame_count % 30 == 0:
                                print(f"[LANDMARKS] ✓ Matched landmarks to primary face {face_idx}")
                            elif self.frame_count % 30 == 0:
                                print(f"[LANDMARKS] ✗ No landmark match for primary face {face_idx}")
                        
                        # Fallback: try on cropped face if no landmarks found
                        if face_landmarks is None:
                            rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                            landmarks_results = face_mesh_detector.process(rgb_face)
                            if landmarks_results and landmarks_results.multi_face_landmarks:
                                face_landmarks = landmarks_results.multi_face_landmarks[0]
                                if self.frame_count % 30 == 0:
                                    print(f"[LANDMARKS] ✓ Detected landmarks on cropped face {face_idx} (fallback)")
                            elif self.frame_count % 30 == 0:
                                print(f"[LANDMARKS] ✗ FAILED to detect landmarks for face {face_idx} (crop size: {cropped_face.shape})")

                        # Emotion and Liveness Analysis (PRIMARY FACE ONLY for performance)
                        if (is_primary_face and self.emotion_analysis_enabled and 
                            (self.frame_count % self.EMOTION_EVERY_N_FRAMES == 0)):
                            try:
                                # Initialize default values
                                emotion = 'Neutral'
                                is_live = False
                                liveness_confidence = 0.0
                                liveness_details = {'error': 'No analysis performed'}
                                
                                if self.lightweight_liveness:
                                    # LIGHTWEIGHT: Blink-only detection (like test_liveness_simple.py)
                                    
                                    # Initialize verification timer
                                    if self.verification_start_time is None:
                                        self.verification_start_time = time.time()
                                        # Let blink detector maintain state across verifications for better continuity
                                    
                                    current_time = time.time()
                                    
                                    # Simple blink detection
                                    if face_landmarks is not None:
                                        try:
                                            blink_detected, current_ear, total_blinks = self.blink_detector.detect_blink(face_landmarks)
                                            has_blinked, blinks_needed = self.blink_detector.requires_blink(
                                                self.verification_start_time, current_time, min_blinks=1
                                            )
                                            
                                            # Blink detection working properly
                                            
                                            elapsed = current_time - self.verification_start_time
                                            
                                            # Simple liveness decision (like test_liveness_simple.py)
                                            if has_blinked:
                                                is_live = True
                                                liveness_confidence = 0.95
                                                emotion = 'Neutral'  # Skip emotion analysis for performance
                                            elif elapsed < 10.0:  # Extended timeout for better UX
                                                is_live = None  # Still waiting
                                                liveness_confidence = 0.5
                                                emotion = 'Neutral'
                                            else:
                                                # Extended timeout - reset and allow retry (only after 10s)
                                                self.verification_start_time = time.time()
                                                if hasattr(self, 'blink_detector') and self.blink_detector:
                                                    self.blink_detector.reset()
                                                    print(f"[BLINK] Reset after 10s timeout - fresh start")
                                                is_live = False  # Failed verification
                                                liveness_confidence = 0.1
                                                emotion = 'Neutral'
                                            
                                            # Create lightweight liveness details
                                            liveness_details = {
                                                'method': 'lightweight_blink_only',
                                                'blink': {
                                                    'current_ear': current_ear,
                                                    'total_blinks': total_blinks,
                                                    'has_blinked': has_blinked,
                                                    'blinks_needed': blinks_needed,
                                                    'elapsed_time': elapsed
                                                }
                                            }
                                            
                                            if blink_detected:
                                                print(f"[BLINK DETECTED!] Frame {len(tracked_faces)}, Total blinks: {total_blinks}, EAR: {current_ear:.3f}")
                                                
                                        except Exception as blink_error:
                                            # Use safe defaults on blink detection error
                                            is_live = False
                                            liveness_confidence = 0.0
                                            current_ear = 0.0
                                            total_blinks = 0
                                            liveness_details = {'error': f'Blink detection failed: {blink_error}'}
                                    else:
                                        # No landmarks available
                                        is_live = False
                                        liveness_confidence = 0.0
                                        emotion = 'Neutral'
                                        current_ear = 0.0
                                        total_blinks = 0
                                        liveness_details = {'error': 'No landmarks available'}
                                else:
                                    # ORIGINAL: Heavy emotion+liveness analysis (fallback)
                                    
                                    from src.emotion import analyze_emotion_and_liveness
                                    emotion, is_live, liveness_confidence, liveness_details = analyze_emotion_and_liveness(
                                        cropped_face_resized, face_landmarks
                                    )
                                
                                # Thread-safe update of liveness state
                                with self.liveness_state_lock:
                                    self.last_emotion = emotion
                                    # Handle waiting state (None) properly
                                    if is_live is True:
                                        self.last_liveness = 'Real'
                                    elif is_live is False:
                                        self.last_liveness = 'Spoof'
                                    else:  # is_live is None (waiting for blink)
                                        self.last_liveness = 'Waiting'
                                    
                                    self.last_liveness_confidence = liveness_confidence
                                    
                                    # Extract and store EAR data if available
                                    if isinstance(liveness_details, dict) and 'blink' in liveness_details:
                                        blink_data = liveness_details['blink']
                                        if isinstance(blink_data, dict) and 'current_ear' in blink_data:
                                            current_ear = blink_data['current_ear']
                                            with self.ear_lock:
                                                self.ear_history.append(current_ear)
                                                print(f"[EAR] Stored EAR={current_ear:.3f}, history size: {len(self.ear_history)}")
                                            
                                            # Update EAR debug display (safe UI update)
                                            blink_count = blink_data.get('total_blinks', 0)
                                            self.current_ear = current_ear
                                            self.current_blinks = blink_count
                                            self.window.after(0, self.update_ear_debug_display, current_ear, blink_count)
                                
                                print(f"[ANALYSIS] Emotion: {emotion}, Liveness: {self.last_liveness} ({liveness_confidence:.1%})")
                                        
                            except Exception as e:
                                print(f"[ANALYSIS ERROR] {e}")
                                self.emotion_failure_count += 1
                                if self.emotion_failure_count >= 5:
                                    print(f"[ANALYSIS] Too many failures, disabling emotion analysis")
                                    self.emotion_analysis_enabled = False
                                    self.window.after(0, self.disable_blink_detection_ui)
                                
                                # Set safe defaults on error
                                with self.liveness_state_lock:
                                    self.last_emotion = "Neutral"
                                    self.last_liveness = "Unknown"
                                    self.last_liveness_confidence = 0.0
                        # Thread-safe liveness state reading
                        current_time = time.time()
                        
                        with self.liveness_state_lock:
                            # Apply hysteresis to reduce flickering
                            if self.last_liveness == 'Spoof':
                                self.consec_spoof_count += 1
                                self.consec_real_count = 0
                            elif self.last_liveness == 'Real':
                                self.consec_real_count += 1
                                self.consec_spoof_count = 0
                            
                            # Only change state after CONSEC_REQUIRED consecutive detections
                            stable_liveness = self.last_liveness
                            if self.consec_spoof_count >= self.CONSEC_REQUIRED:
                                stable_liveness = 'Spoof'
                            elif self.consec_real_count >= self.CONSEC_REQUIRED:
                                stable_liveness = 'Real'
                            
                            # Auto-recover from spoof state only when face changes or after extended timeout
                            if stable_liveness == 'Spoof':
                                time_since_detection = current_time - self.last_spoof_detection_time
                                if time_since_detection > (self.SPOOF_WARNING_DISPLAY_TIME * 3):  # Triple the timeout - 24 seconds
                                    print("[SPOOF] Auto-recovery: clearing spoof state after extended timeout")
                                    self.reset_spoof_detection()
                                    stable_liveness = 'Unknown'
                            
                            is_live = (stable_liveness == 'Real')
                            liveness_status = stable_liveness

                        # Check for spoof (thread-safe)
                        if not is_live:
                            self.last_identity = f"SPOOF - {liveness_status}"
                            box_color = (0, 0, 255)
                            with self.liveness_state_lock:
                                if not self.spoof_warning_shown:
                                    self.last_spoof_detection_time = current_time
                                    self.spoof_warning_shown = True
                            # Don't process verification for spoof, but don't freeze either
                        else:
                            # Verification with multi-embedding support
                            try:
                                    # ========== TIMING: Face Verification ==========
                                    verification_start = time.time()
                                    
                                    # Avoid disk I/O: convert cv2 image to PIL directly
                                    preprocess_start = time.time()
                                    rgb = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2RGB)
                                    pil_image = Image.fromarray(rgb).convert('RGB')
                                    image_tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)
                                    preprocess_time = (time.time() - preprocess_start) * 1000

                                    embedding_start = time.time()
                                    with torch.no_grad():
                                        trial_embedding = verification_model(image_tensor, mode='metric').cpu()
                                    embedding_time = (time.time() - embedding_start) * 1000

                                    min_distance = float('inf')
                                    self.last_identity = "Not Registered"
                                    best_match = None

                                    # Multi-embedding comparison
                                    comparison_start = time.time()
                                    best_match_data = None  # Track the employee data for pose matching
                                    for name, saved_data in employee_db.items():
                                        if USE_MULTI_EMBEDDING and isinstance(saved_data, list):
                                            # Compare against all stored embeddings, use minimum distance
                                            distances = [F.pairwise_distance(trial_embedding, emb).item() 
                                                       for emb in saved_data]
                                            distance = min(distances) if distances else float('inf')
                                        else:
                                            # Single embedding (backward compatible)
                                            saved_embedding = saved_data[0] if isinstance(saved_data, list) else saved_data
                                            distance = F.pairwise_distance(trial_embedding, saved_embedding).item()
                                        
                                        if distance < min_distance:
                                            min_distance = distance
                                            best_match = name
                                            best_match_data = saved_data  # Store for pose matching

                                    comparison_time = (time.time() - comparison_start) * 1000
                                    
                                    # Load current threshold (may have been adjusted by user)
                                    current_threshold = _load_gui_threshold(OPTIMAL_THRESHOLD_GUI)
                                    
                                    # Calculate confidence and prepare result
                                    threshold = _load_gui_threshold(OPTIMAL_THRESHOLD_GUI)
                                    confidence = max(0, min(100, (1 - min_distance / threshold) * 100))
                                    
                                    # Only log final result for main face identification
                                    if face_idx == 0:
                                        print(f"Face {face_idx}: {best_match or 'None'} | Distance: {min_distance:.3f} | Confidence: {confidence:.0f}%")
                                    
                                    # Enhanced rejection logic for small databases
                                    adjusted_threshold = current_threshold
                                    if len(employee_db) <= 2:  # Small database - be more strict
                                        adjusted_threshold = current_threshold * UNRECOGNIZED_DISTANCE_MULTIPLIER
                                    
                                    # MULTI-PERSON ENHANCEMENT: Store all verification results for conflict resolution
                                    verification_result = {
                                        'face_idx': face_idx,
                                        'best_match': best_match,
                                        'distance': min_distance,
                                        'confidence': confidence,
                                        'threshold': adjusted_threshold,
                                        'bbox': (x, y, w, h)
                                    }
                                    
                                    # Store in frame-level results for conflict resolution
                                    if not hasattr(self, 'current_frame_verifications'):
                                        self.current_frame_verifications = []
                                    self.current_frame_verifications.append(verification_result)
                                    
                                    # ENHANCED MULTI-PERSON IDENTITY DETERMINATION
                                    if confidence < CONFIDENCE_REJECTION_THRESHOLD * 100:
                                        raw_identity = "Not Registered (Low Confidence)"
                                        if face_idx == 0:
                                            print(f"Rejected: Low confidence ({confidence:.0f}%)")
                                    elif min_distance < adjusted_threshold:
                                        # CONFLICT RESOLUTION: Check if another face has better match for same person
                                        if len(faces_to_process) > 1 and hasattr(self, 'current_frame_verifications'):
                                            # Find if any other face has better confidence for this person
                                            better_match_exists = False
                                            for other_result in self.current_frame_verifications:
                                                if (other_result['best_match'] == best_match and 
                                                    other_result['face_idx'] != face_idx and
                                                    other_result['confidence'] > confidence + 10):  # 10% better threshold
                                                    better_match_exists = True
                                                    break
                                            
                                            if better_match_exists:
                                                raw_identity = "Ambiguous Match"
                                                if face_idx == 0:
                                                    print(f"Conflict: Another face has better match for {best_match}")
                                            else:
                                                raw_identity = best_match
                                        else:
                                            raw_identity = best_match
                                    else:
                                        raw_identity = "Not Registered"
                                    
                                    # Update face-specific recognition tracking
                                    if current_face_id >= 0:
                                        self.update_face_recognition(current_face_id, raw_identity, confidence)
                                        face_history = self.get_face_recognition_history(current_face_id)
                                    else:
                                        face_history = []
                                    
                                    # Apply face-specific recognition smoothing
                                    frame_identity = raw_identity
                                    if len(face_history) >= 2:
                                        # Face-specific smoothing with confidence weighting
                                        identity_scores = {}
                                        for result in face_history:
                                            hist_id = result['identity']
                                            hist_conf = result['confidence']
                                            
                                            # Weight by confidence, prefer recognized identities
                                            weight = hist_conf / 100.0
                                            if hist_id != "Not Registered" and "Low Confidence" not in hist_id:
                                                weight *= 1.3  # Boost recognized identities
                                            
                                            if hist_id not in identity_scores:
                                                identity_scores[hist_id] = {'weight': 0, 'count': 0}
                                            identity_scores[hist_id]['weight'] += weight
                                            identity_scores[hist_id]['count'] += 1
                                        
                                        # Choose best identity for this specific face
                                        best_identity = None
                                        best_score = 0
                                        for identity, data in identity_scores.items():
                                            avg_confidence = (data['weight'] / data['count']) * 100
                                            # More lenient for tracked faces
                                            if data['count'] >= 2 or avg_confidence > 75:
                                            # More lenient for tracked faces
                                                if data['weight'] > best_score:
                                                    best_score = data['weight']
                                                    best_identity = identity
                                        
                                        # Use tracked identity if strong enough
                                        if best_identity and best_identity not in ["Not Registered", "Not Registered (Low Confidence)"]:
                                            frame_identity = best_identity
                                            print(f"[TRACKING] Face {current_face_id}: Using tracked identity '{best_identity}' (score: {best_score:.2f})")
                                    
                                    # Fallback to global smoothing for primary face only (backwards compatibility)
                                    if face_idx == 0 and frame_identity == raw_identity:
                                        self.recognition_history.append((raw_identity, confidence))
                                        if len(self.recognition_history) > self.SMOOTHING_WINDOW:
                                            self.recognition_history.pop(0)
                                    
                                    # ========== IDENTITY LOCK SYSTEM FOR SEAMLESS CHECK-IN ==========
                                    current_time = time.time()
                                    # Store face-specific identity (primary face only for check-in)
                                    if is_primary_face:  # Only primary face uses identity lock system
                                        # Check if we have a locked identity
                                        if self.locked_identity is not None:
                                            # Check if lock display time has expired
                                            if current_time - self.lock_timestamp > self.LOCK_DISPLAY_TIME:
                                                # Auto-reset after display timeout
                                                print("[RESET] Identity lock timeout - preparing for next person")
                                                self.locked_identity = None
                                                self.identity_lock_buffer = []
                                                self.last_identity = "Not Registered"
                                                # Reset detection state for next person
                                                self.reset_spoof_detection("identity lock timeout")
                                            else:
                                                # Keep showing locked identity
                                                self.last_identity = self.locked_identity
                                                box_color = (0, 255, 0)
                                        else:
                                            # No locked identity - accumulate verifications
                                            if raw_identity not in ["Not Registered", "Not Registered (Low Confidence)", "Processing...", "Error"]:
                                                # Add to buffer
                                                self.identity_lock_buffer.append((current_time, raw_identity, confidence))
                                            else:
                                                # No face recognized - clear buffer if it's been too long
                                                if self.identity_lock_buffer:
                                                    oldest_time = min(t for t, _, _ in self.identity_lock_buffer)
                                                    if current_time - oldest_time > self.LOCK_DURATION:
                                                        self.identity_lock_buffer = []
                                            
                                            # Remove old entries (older than LOCK_DURATION)
                                            self.identity_lock_buffer = [
                                                (t, name, conf) for t, name, conf in self.identity_lock_buffer
                                                if current_time - t <= self.LOCK_DURATION
                                            ]
                                            
                                            # Count verifications per identity
                                            identity_counts = {}
                                            for _, name, conf in self.identity_lock_buffer:
                                                if name not in identity_counts:
                                                    identity_counts[name] = []
                                                identity_counts[name].append(conf)
                                            
                                            # Find most verified identity
                                            if identity_counts:
                                                most_verified = max(identity_counts.items(), key=lambda x: len(x[1]))
                                                most_verified_name = most_verified[0]
                                                verification_count = len(most_verified[1])
                                                avg_confidence = sum(most_verified[1]) / len(most_verified[1])
                                                
                                                # Lock if we have enough verifications
                                                if verification_count >= self.LOCK_MIN_VERIFICATIONS:
                                                    self.locked_identity = most_verified_name
                                                    self.lock_timestamp = current_time
                                                    self.last_identity = most_verified_name
                                                    box_color = (0, 255, 0)
                                                    
                                                    print(f"[LOCK] Identity locked: {most_verified_name} ({verification_count} verifications, {avg_confidence:.1f}% avg confidence)")
                                                    
                                                    # Update UI Log immediately (on main thread to ensure it shows)
                                                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                                                    log_entry = f"[{timestamp}] {most_verified_name}"
                                                    self.log_listbox.insert(0, log_entry)
                                                    print(f"✓ Added to check-in log: {log_entry}")
                                                    
                                                    # Mark attendance in background
                                                    def mark_async():
                                                        try:
                                                            success, message = self.attendance_logger.mark_attendance(
                                                                most_verified_name, min_distance, self.last_emotion, self.last_liveness
                                                            )
                                                            if success:
                                                                print(f"✓ {message}")
                                                        except Exception as e:
                                                            print(f"Attendance marking error: {e}")
                                                    
                                                    threading.Thread(target=mark_async, daemon=True).start()
                                                    self.last_attendance_message = f"Checked in: {most_verified_name}"
                                                    
                                                    # Spoof detection continues for next person - will reset on identity timeout
                                                else:
                                                    # Still accumulating - show progress
                                                    buffer_age = current_time - min(t for t, _, _ in self.identity_lock_buffer)
                                                    progress_pct = int((buffer_age / self.LOCK_DURATION) * 100)
                                                    self.last_identity = f"{most_verified_name} (verifying... {verification_count}/{self.LOCK_MIN_VERIFICATIONS})"
                                                    box_color = (255, 165, 0)  # Orange while verifying
                                            else:
                                                self.last_identity = "Not Registered"
                                                box_color = (0, 0, 255)
                                    
                                    # Store results for primary face (backwards compatibility)
                                    if face_idx == 0:
                                        self.last_confidence = confidence
                                        self.last_distance = min_distance
                                        self.last_identity = frame_identity
                                    
                                    # Store results for this specific face
                                    while len(self.face_identities) <= face_idx:
                                        self.face_identities.append("Processing...")
                                        self.face_confidences.append(0.0)
                                        self.face_emotions.append("Unknown")
                                        self.face_liveness.append("Unknown")
                                    
                                    self.face_identities[face_idx] = frame_identity
                                    self.face_confidences[face_idx] = confidence
                                    
                                    # Update per-face verification tracking
                                    if self.ENABLE_MULTI_FACE_VERIFICATION and current_face_id >= 0:
                                        self.face_identities_verified[current_face_id] = frame_identity
                                        self.face_confidences_verified[current_face_id] = confidence
                                        
                                        # Initialize verification history if needed
                                        if current_face_id not in self.face_verification_history:
                                            self.face_verification_history[current_face_id] = []
                                        
                                        # Add to verification history
                                        self.face_verification_history[current_face_id].append({
                                            'identity': frame_identity,
                                            'confidence': confidence,
                                            'timestamp': current_time
                                        })
                                        
                                        # Keep only recent history
                                        if len(self.face_verification_history[current_face_id]) > 10:
                                            self.face_verification_history[current_face_id].pop(0)
                                    
                                    # Set emotion/liveness based on face type
                                    if is_primary_face:
                                        self.face_emotions[face_idx] = self.last_emotion
                                        self.face_liveness[face_idx] = self.last_liveness
                                    else:
                                        # Secondary faces get basic status
                                        self.face_emotions[face_idx] = "N/A"
                                        self.face_liveness[face_idx] = "Verified" if confidence > 50 else "Unknown"
                                    

                                    
                                    # xAI: Store face data for explainability features
                                    xai_start = time.time()
                                    self.current_face_tensor = image_tensor
                                    self.current_face_image = rgb  # RGB numpy array
                                    
                                    # Generate explanation data
                                    if self.explainer is not None:
                                        try:
                                            self.current_explanation = self.explainer.explain_distance(
                                                min_distance, threshold
                                            )
                                            # Add quality analysis
                                            quality_exp = self.explainer.explain_quality_factors(rgb)
                                            self.current_explanation['quality_message'] = quality_exp['overall_message']
                                        except Exception as e:
                                            print(f"[WARNING] Explanation generation failed: {e}")
                                    xai_time = (time.time() - xai_start) * 1000
                                    if xai_time > 5:  # Only log if significant
                                        print(f"    [TIMING] xAI Explanation: {xai_time:.2f}ms")
                                    
                                    # Store kNN neighbors (for kNN analysis button)
                                    # TODO: Implement proper kNN tracking when available
                                    self.knn_neighbors = (
                                        [best_match] * 5,  # Placeholder: same name 5 times
                                        [min_distance] * 5  # Placeholder: same distance
                                    )
                                    
                                    # Find which pose matched (for multi-embedding)
                                    if best_match_data and USE_MULTI_EMBEDDING and isinstance(best_match_data, list):
                                        distances_with_idx = [(F.pairwise_distance(trial_embedding, emb).item(), idx) 
                                                             for idx, emb in enumerate(best_match_data)]
                                        _, self.matched_pose_index = min(distances_with_idx, key=lambda x: x[0])
                                    else:
                                        self.matched_pose_index = 0
                                    
                                    # ========== TOTAL TIMING ==========
                                    total_process_time = (time.time() - process_start) * 1000
                                    # Only log slow processing times
                                    if total_process_time > 100:  # Log only if > 100ms
                                        print(f"  Processing time: {total_process_time:.0f}ms")
                            except Exception as e:
                                print(f"Verification error: {e}")
                                self.last_identity = "Error"
                    else:
                        self.last_identity = "Face too small"
                        box_color = (0, 0, 255)

                # Display proper label for each face with primary/secondary status
                if face_idx < MAX_CONCURRENT_FACES:
                    if self.registration_mode and face_idx == 0:
                        display_text = "Registering..."
                    elif is_primary_face:
                        # Primary face gets full status display
                        with self.liveness_state_lock:
                            display_text = f"PRIMARY: {self.last_identity} ({self.last_emotion} | {self.last_liveness})"
                    elif is_secondary_face:
                        # Secondary faces get verification-only display
                        if hasattr(self, 'face_identities') and face_idx < len(self.face_identities):
                            face_identity = self.face_identities[face_idx]
                            face_confidence = self.face_confidences[face_idx] if face_idx < len(self.face_confidences) else 0
                            display_text = f"SEC: {face_identity} ({face_confidence:.0f}%)"
                        else:
                            display_text = f"SECONDARY: Processing..."
                    else:
                        # Untracked faces
                        if hasattr(self, 'face_identities') and face_idx < len(self.face_identities):
                            display_text = f"UNTRACKED: {self.face_identities[face_idx]}"
                        else:
                            display_text = f"Face {face_idx+1}: Processing..."
                else:
                    display_text = "Face Limit Exceeded"
                
                # Color coding for multiple faces
                if face_idx < MAX_CONCURRENT_FACES:
                    # Smooth color transition for first face (backwards compatibility)
                    if face_idx == 0 and hasattr(self, 'box_color_transition'):
                        trans = self.box_color_transition
                        trans['target_color'] = box_color
                        
                        if trans['current_color'] != trans['target_color']:
                            trans['frame'] += 1
                            if trans['frame'] >= trans['transition_frames']:
                                trans['current_color'] = trans['target_color']
                                trans['frame'] = 0
                            else:
                                factor = trans['frame'] / trans['transition_frames']
                                trans['current_color'] = interpolate_color(
                                    trans['current_color'], trans['target_color'], factor
                                )
                        box_color = trans['current_color']
                    elif face_idx > 0:
                        # Different colors for additional faces
                        face_colors = [(0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]  # Green, Magenta, Cyan, Yellow
                        box_color = face_colors[face_idx % len(face_colors)]
                else:
                    box_color = (128, 128, 128)  # Gray for faces beyond limit
                
                # Draw modern rounded rectangle with thicker line for recognized faces
                current_identity = self.face_identities[face_idx] if face_idx < len(self.face_identities) else "Processing..."
                is_recognized = current_identity not in ["Not Registered", "Not Registered (Low Confidence)", "Error", "Spoof Detected", "Face too small", "Registering...", "Processing..."]
                thickness = 3 if is_recognized else 2
                draw_rounded_rectangle(frame, (x, y), (x+w, y+h), box_color, thickness, radius=12)
                
                # Label with high contrast dark background for better readability
                font = cv2.FONT_HERSHEY_DUPLEX  # Sharper font
                (text_w, text_h), baseline = cv2.getTextSize(display_text, font, 0.6, 2)
                label_y = max(y - text_h - 18, 10)
                
                # Dark semi-transparent background for maximum contrast
                label_overlay = frame.copy()
                draw_rounded_rectangle(label_overlay, (x - 2, label_y), (x + text_w + 24, label_y + text_h + 12), 
                                     (20, 20, 30), -1, radius=8)  # Dark background
                cv2.addWeighted(label_overlay, 0.85, frame, 0.15, 0, frame)
                
                # Bright white text with anti-aliasing (lighter weight)
                cv2.putText(frame, display_text, (x + 10, label_y + text_h + 6), 
                           font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Update statistics (each processed face)
                if face_idx < MAX_CONCURRENT_FACES and self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0:
                    self.recognition_stats['total_detections'] += 1
                    current_identity = self.face_identities[face_idx] if face_idx < len(self.face_identities) else "Processing..."
                    if current_identity not in ["Not Registered", "Not Registered (Low Confidence)", "Error", "Spoof Detected", "Face too small", "Processing..."]:
                        self.recognition_stats['successful_recognitions'] += 1
                        self.recognition_stats['unique_faces_today'].add(current_identity)
            
            # Display state accumulation/lock status indicator
            if not self.registration_mode:
                h, w = frame.shape[:2]
                font_status = cv2.FONT_HERSHEY_DUPLEX
                
                if len(self.confidence_buffer) > 0:
                    # Show accumulation progress
                    buffer_size = len(self.confidence_buffer)
                    progress_pct = min(100, int((buffer_size / 20) * 100))
                    status_text = f"Analyzing: {progress_pct}%"
                    status_color = (255, 193, 7)  # Amber
                    
                    (tw, th), _ = cv2.getTextSize(status_text, font_status, 0.5, 1)
                    
                    # Draw indicator in top-right corner
                    x_pos = w - tw - 40
                    y_pos = 10
                    
                    overlay = frame.copy()
                    draw_rounded_rectangle(overlay, (x_pos - 10, y_pos), 
                                         (x_pos + tw + 20, y_pos + th + 16), 
                                         status_color, -1, radius=10)
                    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                    cv2.putText(frame, status_text, (x_pos + 5, y_pos + th + 5), 
                               font_status, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Draw progress bar below text
                    bar_x = x_pos - 10
                    bar_y = y_pos + th + 20
                    bar_w = tw + 30
                    bar_h = 6
                    
                    # Background bar
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                                 (60, 60, 80), -1)
                    # Progress bar
                    progress_w = int(bar_w * (progress_pct / 100))
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h), 
                                 status_color, -1)
            
            # Display warning banner if face limit exceeded
            if len(faces) > MAX_CONCURRENT_FACES:
                warning_text = f"⚠ {len(faces)} faces detected - processing {MAX_CONCURRENT_FACES} max"
                font_warn = cv2.FONT_HERSHEY_DUPLEX
                (tw, th), _ = cv2.getTextSize(warning_text, font_warn, 0.6, 1)
                overlay = frame.copy()
                draw_rounded_rectangle(overlay, (5, 5), (tw + 30, th + 22), (255, 165, 0), -1, radius=10)
                cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
                cv2.putText(frame, warning_text, (18, th + 14), 
                           font_warn, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            elif len(faces) > 1:
                # ENHANCED: Show multi-person verification with primary/secondary breakdown
                if self.ENABLE_MULTI_FACE_VERIFICATION:
                    primary_count = 1 if self.primary_face_id is not None else 0
                    secondary_count = len([fid for fid in self.face_identities_verified.keys() 
                                         if fid != self.primary_face_id])
                    verified_people = list(self.face_identities_verified.values())
                    recognized_count = len([p for p in verified_people if "Not Registered" not in p])
                    info_text = f"✓ {len(faces)} faces: 1 PRIMARY + {secondary_count} SEC | {recognized_count} recognized"
                else:
                    # Fallback to original multi-face display
                    info_text = f"✓ Processing {len(faces)} faces simultaneously"
                    
                font_info = cv2.FONT_HERSHEY_DUPLEX
                (tw, th), _ = cv2.getTextSize(info_text, font_info, 0.5, 1)
                overlay = frame.copy()
                draw_rounded_rectangle(overlay, (5, 5), (tw + 25, th + 18), (0, 255, 0), -1, radius=8)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                cv2.putText(frame, info_text, (15, th + 12), 
                           font_info, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Log performance metrics
            if hasattr(self, 'process_start'):
                process_time = (time.time() - self.process_start) * 1000
                self.performance_monitor.log_timing('total', process_time)
            
            # Only put frame if queue is not full (prevents backup and lag)
            try:
                self.frame_queue.put_nowait(frame)
                # Debug: Log frame queue success periodically
                if self.frame_count % 300 == 0:
                    print(f"[DEBUG] Frame {self.frame_count} queued successfully")
            except queue.Full:
                # Skip this frame to prevent lag
                if self.frame_count % 30 == 0:
                    print(f"[WARNING] Frame queue full! Skipping frame {self.frame_count} to prevent lag.")
                pass

    def update_display(self):
        """Update display on main thread - optimized for performance"""
        if not self.running:
            return

        try:
            display_start = time.time()
            # Clear queue if backed up to prevent lag
            frame = None
            frames_skipped = 0
            while not self.frame_queue.empty():
                try:
                    if frame is not None:
                        frames_skipped += 1
                    frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            if frames_skipped > 0 and self.frame_count % 30 == 0:
                print(f"[UI] Skipped {frames_skipped} frame(s) to catch up")
            
            if frame is not None:
                try:
                    render_start = time.time()
                    # Resize frame to fit display window
                    frame_resized = cv2.resize(frame, (self.video_width, self.video_height))
                    
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    render_time = (time.time() - render_start) * 1000

                    # Update video display
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk, text="")
                    
                    if render_time > 10 and self.frame_count % 30 == 0:  # Log if render is slow
                        print(f"[UI TIMING] Frame Rendering: {render_time:.2f}ms")
                        
                except Exception as e:
                    print(f"[ERROR] Frame rendering failed: {e}")
            else:
                # No frame available - show status message
                if self.frame_count % 60 == 0:  # Every 2 seconds
                    print("[DEBUG] No frame available for display")

                # Update UI elements less frequently (every 3 display updates)
                if self.frame_count % 3 == 0:
                    update_ui_start = time.time()
                    self.update_detection_display()
                    self.update_stats_display()
                    self.update_debug_display()
                    self.update_ear_debug_panel()
                    update_ui_time = (time.time() - update_ui_start) * 1000
                    if update_ui_time > 10 and self.frame_count % 30 == 0:
                        print(f"[UI TIMING] UI Elements Update: {update_ui_time:.2f}ms")
            
        except Exception as e:
            print(f"Display error: {e}")

        # Reduced update frequency to 40ms (25 FPS) for better performance
        self.window.after(40, self.update_display)
    
    def update_detection_display(self):
        """Update the detection information display"""
        if self.last_identity != "Not Registered" and self.last_identity != "Error":
            # Successful recognition
            self.identity_label.config(text=f"✅ {self.last_identity}", 
                                     bg='#d5f4e6', fg='#27ae60')
            status_text = f"🟢 Recognition successful: {self.last_identity}"
        elif self.last_identity == "Error":
            # Error state
            self.identity_label.config(text="❌ Recognition Error", 
                                     bg='#fdeaea', fg='#e74c3c')
            status_text = "🔴 Error occurred during recognition"
        else:
            # Unknown person or no face
            if self.last_identity == "Not Registered":
                self.identity_label.config(text="❓ Unknown Person", 
                                         bg='#fff3cd', fg='#856404')
                if len(self.confidence_buffer) > 0:
                    progress = min(100, int((len(self.confidence_buffer) / 20) * 100))
                    status_text = f"📊 Analyzing face data... {progress}%"
                else:
                    status_text = "🟡 Face detected but not recognized"
            else:
                self.identity_label.config(text="👤 No face detected", 
                                         bg='white', fg='#2c3e50')
                status_text = "⭕ No face in camera view"
        
        # Update individual components
        self.emotion_var.set(self.last_emotion)
        self.liveness_var.set(self.last_liveness)
        
        # Color-code liveness and show confidence
        if self.last_liveness == "Real":
            self.liveness_label.config(fg='#27ae60')
            self.liveness_confidence_var.set(f"{self.last_liveness_confidence:.0%} confidence")
        elif self.last_liveness == "Spoof":
            self.liveness_label.config(fg='#e74c3c')
            # Show inverse confidence for spoof (100% - confidence = certainty of spoof)
            spoof_certainty = 1.0 - self.last_liveness_confidence
            self.liveness_confidence_var.set(f"{spoof_certainty:.0%} certainty")
        else:
            self.liveness_label.config(fg='#3498db')
            self.liveness_confidence_var.set("")
        
        # Update distance
        if self.last_distance != float('inf'):
            self.distance_var.set(f"{self.last_distance:.3f}")
        else:
            self.distance_var.set("N/A")
        
        
        # Update debug panel
        self.update_debug_panel()
        
        # Update status (include attendance message if recent)
        if self.last_attendance_message:
            status_text = f"{status_text} | 📅 {self.last_attendance_message}"
        self.status_text.set(status_text)
    
    def adjust_threshold(self):
        """Adjust verification threshold with enhanced dialog"""
        global OPTIMAL_THRESHOLD_GUI
        
        # Create custom dialog
        dialog = tk.Toplevel(self.window)
        dialog.title("⚙ Adjust Recognition Threshold")
        dialog.geometry("400x500")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.window.winfo_rootx() + 50, self.window.winfo_rooty() + 50))
        
        # Header
        header = tk.Frame(dialog, bg='#3498db', height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="🎯 Recognition Threshold Settings", font=('Arial', 14, 'bold'), 
                fg='white', bg='#3498db').pack(pady=15)
        
        # Content
        content = tk.Frame(dialog, bg='#f0f0f0')
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(content, text=f"Current Threshold: {OPTIMAL_THRESHOLD_GUI:.3f}", 
                font=('Arial', 12, 'bold'), bg='#f0f0f0').pack(pady=(0, 10))
        
        # Threshold descriptions
        desc_frame = tk.Frame(content, bg='#f0f0f0')
        desc_frame.pack(fill=tk.X, pady=(0, 15))
        
        descriptions = [
            ("🔒 Strict (0.5-0.8)", "High security, may reject some valid faces", '#e74c3c'),
            ("⚖ Balanced (0.8-1.2)", "Good balance of security and usability", '#f39c12'),
            ("🔓 Lenient (1.2-2.0)", "Lower security, accepts more variations", '#27ae60')
        ]
        
        for label, desc, color in descriptions:
            frame = tk.Frame(desc_frame, bg='#f0f0f0')
            frame.pack(fill=tk.X, pady=2)
            tk.Label(frame, text=label, font=('Arial', 10, 'bold'), 
                    fg=color, bg='#f0f0f0').pack(side=tk.LEFT)
            tk.Label(frame, text=f": {desc}", font=('Arial', 10), 
                    fg='#2c3e50', bg='#f0f0f0').pack(side=tk.LEFT)
        
        # Threshold input
        tk.Label(content, text="Enter new threshold (0.1-3.0):", 
                font=('Arial', 11), bg='#f0f0f0').pack(anchor=tk.W, pady=(15, 5))
        
        threshold_var = tk.StringVar(value=str(OPTIMAL_THRESHOLD_GUI))
        entry = tk.Entry(content, textvariable=threshold_var, font=('Arial', 12), width=10)
        entry.pack(anchor=tk.W)
        entry.select_range(0, tk.END)
        entry.focus()
        
        # Buttons
        button_frame = tk.Frame(content, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def apply_threshold():
            try:
                new_value = float(threshold_var.get())
                if 0.1 <= new_value <= 3.0:
                    global OPTIMAL_THRESHOLD_GUI
                    OPTIMAL_THRESHOLD_GUI = new_value
                    _save_gui_threshold(OPTIMAL_THRESHOLD_GUI)
                    self.update_debug_display()
                    messagebox.showinfo("Success", f"✅ Threshold saved! New value: {OPTIMAL_THRESHOLD_GUI:.3f}")
                    dialog.destroy()
                else:
                    messagebox.showerror("Invalid Value", "⚠ Please enter a value between 0.1 and 3.0")
            except ValueError:
                messagebox.showerror("Invalid Input", "⚠ Please enter a valid number")
        
        ttk.Button(button_frame, text="✓ Save", command=apply_threshold, 
                  style='Success.TButton').pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="✗ Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    def view_employees(self):
        """View all employees with enhanced dialog"""
        if len(employee_db) == 0:
            messagebox.showinfo("👥 Employee List", "No employees registered yet.\n\nClick 'Register Employee' to add your first employee!")
            return
        
        # Create custom dialog
        dialog = tk.Toplevel(self.window)
        dialog.title("👥 Employee Database")
        dialog.geometry("500x400")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.window)
        
        # Header
        header = tk.Frame(dialog, bg='#27ae60', height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text=f"👥 Registered Employees ({len(employee_db)})", 
                font=('Arial', 14, 'bold'), fg='white', bg='#27ae60').pack(pady=15)
        
        # Content
        content = tk.Frame(dialog, bg='#f0f0f0')
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Employee list with scrollbar
        list_frame = tk.Frame(content)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
                            font=('Arial', 11), height=15)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        for i, name in enumerate(sorted(employee_db.keys()), 1):
            listbox.insert(tk.END, f"{i:2d}. {name}")
        
        # Close button
        ttk.Button(content, text="Close", command=self.view_employees).pack(pady=(10, 0))
    

    
    def edit_employee(self):
        """Batch edit employees with checkboxes"""
        if len(employee_db) == 0:
            messagebox.showinfo("Edit Employees", "No employees registered yet.")
            return
        
        dialog = tk.Toplevel(self.window)
        dialog.title("✏ Edit Employees")
        dialog.geometry("500x600")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Header
        header = tk.Frame(dialog, bg='#f39c12', height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="Batch Edit Employee Names", font=('Segoe UI Semibold', 14, 'bold'), 
                fg='white', bg='#f39c12').pack(pady=15)
        
        # Content
        content = tk.Frame(dialog, bg='#f0f0f0')
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(content, text="Select employees to rename:", font=('Segoe UI', 11, 'bold'), 
                bg='#f0f0f0').pack(anchor=tk.W, pady=(0, 10))
        
        # Employee list with checkboxes
        list_frame = tk.Frame(content, bg='white', relief='solid', borderwidth=1)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        canvas = tk.Canvas(list_frame, bg='white')
        scrollbar = tk.Scrollbar(list_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Checkboxes for each employee
        check_vars = {}
        for name in sorted(employee_db.keys()):
            var = tk.BooleanVar()
            check_vars[name] = var
            cb = tk.Checkbutton(scrollable_frame, text=name, variable=var, 
                               font=('Segoe UI', 10), bg='white', anchor='w')
            cb.pack(fill=tk.X, padx=10, pady=2)
        
        # Selection buttons
        select_frame = tk.Frame(content, bg='#f0f0f0')
        select_frame.pack(fill=tk.X, pady=(0, 10))
        
        def select_all():
            for var in check_vars.values():
                var.set(True)
        
        def deselect_all():
            for var in check_vars.values():
                var.set(False)
        
        tk.Button(select_frame, text="Select All", command=select_all, 
                 font=('Segoe UI', 9), bg='#3498db', fg='white', relief='flat',
                 padx=10, pady=5).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(select_frame, text="Deselect All", command=deselect_all,
                 font=('Segoe UI', 9), bg='#95a5a6', fg='white', relief='flat',
                 padx=10, pady=5).pack(side=tk.LEFT)
        
        def batch_rename():
            selected = [name for name, var in check_vars.items() if var.get()]
            if not selected:
                messagebox.showwarning("⚠ No Selection", "Please select at least one employee.")
                return
            
            renamed_count = 0
            identities_dir = Path("identities")
            
            for old_name in selected:
                new_name = simpledialog.askstring("✏ Rename Employee", 
                                                 f"Enter new name for '{old_name}':",
                                                 parent=dialog)
                if new_name and new_name.strip() and new_name != old_name:
                    new_name = new_name.strip()
                    if new_name in employee_db:
                        messagebox.showerror("❌ Error", f"Employee '{new_name}' already exists! Skipping.")
                        continue
                    
                    # Update database
                    employee_db[new_name] = employee_db.pop(old_name)
                    
                    # Rename image folder if it exists
                    old_folder = identities_dir / old_name
                    new_folder = identities_dir / new_name
                    
                    if old_folder.exists() and old_folder.is_dir():
                        try:
                            old_folder.rename(new_folder)
                            print(f"Renamed image folder: {old_folder} -> {new_folder}")
                        except Exception as e:
                            print(f"Failed to rename image folder for '{old_name}': {e}")
                            # Continue with database rename even if folder rename fails
                    
                    renamed_count += 1
            
            if renamed_count > 0:
                torch.save(employee_db, EMPLOYEE_DB_PATH)
                messagebox.showinfo("✅ Success", f"Successfully renamed {renamed_count} employee(s) and their image folders!")
                self.update_stats_display()
                self.update_debug_display()
            dialog.destroy()
        
        # Buttons
        button_frame = tk.Frame(content, bg='#f0f0f0')
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="✓ Rename Selected", command=batch_rename,
                  style='Success.TButton').pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="✗ Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    def view_attendance(self):
        """Phase 8: View attendance records"""
        dialog = tk.Toplevel(self.window)
        dialog.title("📅 Attendance Records")
        dialog.geometry("800x600")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.window)
        
        # Header
        header = tk.Frame(dialog, bg='#4a90e2', height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="📅 Attendance Records", font=('Segoe UI Semibold', 14, 'bold'), 
                fg='white', bg='#4a90e2').pack(pady=15)
        
        # Content
        content = tk.Frame(dialog, bg='#f0f0f0')
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Tab selection
        tab_frame = tk.Frame(content, bg='#f0f0f0')
        tab_frame.pack(fill=tk.X, pady=(0, 10))
        
        current_tab = tk.StringVar(value="today")
        
        def show_today():
            current_tab.set("today")
            update_display()
        
        def show_all():
            current_tab.set("all")
            update_display()
        
        tk.Button(tab_frame, text="Today", command=show_today, 
                 font=('Segoe UI', 10), bg='#4a90e2', fg='white', relief='flat',
                 padx=20, pady=5).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(tab_frame, text="All Records", command=show_all,
                 font=('Segoe UI', 10), bg='#2ecc71', fg='white', relief='flat',
                 padx=20, pady=5).pack(side=tk.LEFT)
        
        # Records display
        records_frame = tk.Frame(content, bg='white', relief='solid', borderwidth=1)
        records_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Scrollbar
        scrollbar = tk.Scrollbar(records_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget for records
        text_widget = tk.Text(records_frame, wrap=tk.NONE, font=('Consolas', 9),
                             yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        def update_display():
            text_widget.delete('1.0', tk.END)
            
            if current_tab.get() == "today":
                records = self.attendance_logger.get_today_attendance()
                title = "Today's Attendance"
            else:
                records = self.attendance_logger.get_all_attendance()
                title = "All Attendance Records"
            
            text_widget.insert('1.0', f"{title} ({len(records)} records)\n")
            text_widget.insert(tk.END, "="*80 + "\n\n")
            
            if len(records) == 0:
                text_widget.insert(tk.END, "No attendance records found.\n")
            else:
                # Header
                text_widget.insert(tk.END, f"{'Timestamp':<20} {'Name':<20} {'Distance':<10} {'Emotion':<12} {'Liveness':<10}\n")
                text_widget.insert(tk.END, "-"*80 + "\n")
                
                # Records (already sorted newest first from get_all_attendance)
                for idx, record in records.iterrows():
                    timestamp_str = str(record.get('timestamp', 'N/A'))[:19]  # Remove microseconds
                    name = str(record.get('employee_name', 'Unknown'))[:20]
                    distance = str(record.get('confidence_distance', 'N/A'))[:10]
                    emotion = str(record.get('emotion', 'N/A'))[:12]
                    liveness = str(record.get('liveness_status', 'N/A'))[:10]
                    
                    text_widget.insert(tk.END, f"{timestamp_str:<20} {name:<20} {distance:<10} {emotion:<12} {liveness:<10}\n")
            
            text_widget.config(state=tk.DISABLED)
        
        update_display();
        
        # Summary
        summary_frame = tk.Frame(content, bg='#e8f5e8', relief='solid', borderwidth=1)
        summary_frame.pack(fill=tk.X)
        
        summary = self.attendance_logger.get_attendance_summary(days=7)
        summary_text = "Last 7 Days Summary: "
        if summary:
            summary_text += ", ".join([f"{name}: {count}" for name, count in summary.items()])
        else:
            summary_text += "No records"
        
        tk.Label(summary_frame, text=summary_text, font=('Segoe UI', 9), 
                bg='#e8f5e8', fg='#27ae60').pack(padx=10, pady=8)
        
        # Close button
        tk.Button(content, text="Close", command=dialog.destroy,
                 font=('Segoe UI', 10), bg='#95a5a6', fg='white', relief='flat',
                 padx=20, pady=5).pack()
    
    def delete_employee(self):
        """Batch delete employees with checkboxes"""
        if len(employee_db) == 0:
            messagebox.showinfo("🗑 Delete Employee", "No employees registered yet.")
            return
        
        dialog = tk.Toplevel(self.window)
        dialog.title("🗑 Delete Employee")
        dialog.geometry("400x600")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Header
        header = tk.Frame(dialog, bg='#e74c3c', height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="🗑 Delete Employee", font=('Arial', 14, 'bold'), 
                fg='white', bg='#e74c3c').pack(pady=15)
        
        # Warning
        warning = tk.Frame(dialog, bg='#fff3cd', relief='solid', borderwidth=1)
        warning.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(warning, text="⚠ WARNING: This action cannot be undone!", 
                font=('Arial', 10, 'bold'), fg='#856404', bg='#fff3cd').pack(pady=5)
        
        # Content
        content = tk.Frame(dialog, bg='#f0f0f0')
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        tk.Label(content, text="Select employees to delete:", font=('Segoe UI', 11, 'bold'), 
                fg='#e74c3c', bg='#f0f0f0').pack(anchor=tk.W, pady=(0, 10))
        
        # Employee list with checkboxes
        list_frame = tk.Frame(content, bg='white', relief='solid', borderwidth=1)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        canvas = tk.Canvas(list_frame, bg='white')
        scrollbar = tk.Scrollbar(list_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Checkboxes for each employee
        check_vars = {}
        for name in sorted(employee_db.keys()):
            var = tk.BooleanVar()
            check_vars[name] = var
            cb = tk.Checkbutton(scrollable_frame, text=name, variable=var, 
                               font=('Segoe UI', 10), bg='white', anchor='w')
            cb.pack(fill=tk.X, padx=10, pady=2)
        
        # Selection buttons
        select_frame = tk.Frame(content, bg='#f0f0f0')
        select_frame.pack(fill=tk.X, pady=(0, 10))
        
        def select_all():
            for var in check_vars.values():
                var.set(True)
        
        def deselect_all():
            for var in check_vars.values():
                var.set(False)
        
        tk.Button(select_frame, text="Select All", command=select_all, 
                 font=('Segoe UI', 9), bg='#3498db', fg='white', relief='flat',
                 padx=10, pady=5).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(select_frame, text="Deselect All", command=deselect_all,
                 font=('Segoe UI', 9), bg='#95a5a6', fg='white', relief='flat',
                 padx=10, pady=5).pack(side=tk.LEFT)
        
        def batch_delete():
            selected = [name for name, var in check_vars.items() if var.get()]
            if not selected:
                messagebox.showwarning("⚠ No Selection", "Please select at least one employee.")
                return
            
            confirm = messagebox.askyesno("⚠ Confirm Deletion", 
                                         f"Are you sure you want to delete {len(selected)} employee(s) and their image folders?\n" +
                                         "\n".join(f"• {name}" for name in selected[:5]) +
                                         (f"\n... and {len(selected) - 5} more" if len(selected) > 5 else ""),
                                         icon='warning')
            if confirm:
                identities_dir = Path("identities")
                deleted_count = 0
                
                for name in selected:
                    # Delete from database
                    del employee_db[name]
                    
                    # Delete image folder if it exists
                    employee_folder = identities_dir / name
                    if employee_folder.exists() and employee_folder.is_dir():
                        try:
                            shutil.rmtree(employee_folder)
                            print(f"Deleted image folder: {employee_folder}")
                        except Exception as e:
                            print(f"Failed to delete image folder for '{name}': {e}")
                            # Continue with deletion even if folder deletion fails
                    
                    deleted_count += 1
                
                torch.save(employee_db, EMPLOYEE_DB_PATH)
                messagebox.showinfo("✅ Success", f"Successfully deleted {deleted_count} employee(s) and their image folders!")
                self.update_stats_display()
                self.update_debug_display()
                dialog.destroy()
        
        # Buttons
        button_frame = tk.Frame(content, bg='#f0f0f0')
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="🗑 Delete Selected", command=batch_delete,
                  style='Danger.TButton').pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="✗ Cancel", command=dialog.destroy).pack(side=tk.LEFT)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
                            font=('Arial', 10), height=10)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        for name in sorted(employee_db.keys()):
            listbox.insert(tk.END, name)
        
        def confirm_delete():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("⚠ No Selection", "Please select an employee to delete.")
                return
            
            name = listbox.get(selection[0])
            confirm = messagebox.askyesno("🗑 Confirm Deletion", 
                                        f"Are you sure you want to delete '{name}' and their image folder?\nThis action cannot be undone!", 
                                        parent=dialog)
            
            if confirm:
                # Delete from database
                del employee_db[name]
                
                # Delete image folder if it exists
                identities_dir = Path("identities")
                employee_folder = identities_dir / name
                if employee_folder.exists() and employee_folder.is_dir():
                    try:
                        shutil.rmtree(employee_folder)
                        print(f"Deleted image folder: {employee_folder}")
                    except Exception as e:
                        print(f"Failed to delete image folder for '{name}': {e}")
                        # Continue with deletion even if folder deletion fails
                
                torch.save(employee_db, EMPLOYEE_DB_PATH)
                messagebox.showinfo("✅ Deleted", f"Employee '{name}' and their image folder have been deleted")
                self.recognition_stats['unique_faces_today'].discard(name)
                self.update_stats_display()
                self.update_debug_display()
                dialog.destroy()
        
        # Buttons
        button_frame = tk.Frame(content, bg='#f0f0f0')
        button_frame.pack(fill=tk.X)
    
    def show_attention_map(self):
        """Show attention map visualization for current face with color spectrum legend"""
        if self.current_face_tensor is None or self.current_face_image is None:
            messagebox.showinfo("xAI", "No face detected. Start camera and detect a face first.")
            return
        
        if self.explainer is None:
            messagebox.showinfo("xAI", "Initializing explainability engine...")
            self.explainer = ExplainabilityEngine(verification_model, DEVICE)
        
        try:
            # Generate attention map on full tensor
            attention_map = self.explainer.generate_attention_map(self.current_face_tensor)
            
            # Apply oval mask to suppress edge regions (soft falloff)
            h, w = attention_map.shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            
            # Create oval mask (wider horizontally for face shape)
            mask = ((x - center_x)**2 / (w * 0.35)**2 + 
                    (y - center_y)**2 / (h * 0.4)**2) <= 1
            
            # Smooth mask edges with Gaussian blur
            from scipy.ndimage import gaussian_filter
            mask = gaussian_filter(mask.astype(float), sigma=h*0.05)
            
            # Apply mask to attention map (suppresses edges, keeps center)
            attention_map_masked = attention_map * mask
            
            overlay = self.explainer.overlay_attention_on_image(self.current_face_image, attention_map_masked, alpha=0.6)
            
            # Show in new window
            win = tk.Toplevel(self.window)
            win.title("Attention Map Visualization")
            win.geometry("800x1000")
            win.configure(bg='#0d1117')
            
            # Header
            header = tk.Frame(win, bg='#161b22', relief='solid', borderwidth=1)
            header.pack(fill=tk.X, padx=10, pady=10)
            tk.Label(header, text="Face Verification Attention Map", font=('Arial', 14, 'bold'), 
                    fg='#58a6ff', bg='#161b22').pack(pady=10)
            tk.Label(header, text="Shows which facial regions the model uses for identity matching", 
                    font=('Arial', 10), fg='#8b949e', bg='#161b22').pack(pady=(0, 5))
            tk.Label(header, text="⚠️ Note: This visualizes FACE VERIFICATION (embedding extraction), not emotion detection", 
                    font=('Arial', 9, 'italic'), fg='#f39c12', bg='#161b22').pack(pady=(0, 10))
            
            # Image
            img_frame = tk.Frame(win, bg='#0d1117')
            img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            
            from PIL import Image
            pil_img = Image.fromarray(overlay)
            pil_img = pil_img.resize((600, 600))
            photo = ImageTk.PhotoImage(pil_img)
            
            img_label = tk.Label(img_frame, image=photo, bg='#0d1117')
            img_label.image = photo
            img_label.pack()
            
            # Color spectrum legend
            legend_frame = tk.Frame(win, bg='#161b22', relief='solid', borderwidth=1)
            legend_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            tk.Label(legend_frame, text="Attention Level Color Spectrum", font=('Arial', 11, 'bold'), 
                    fg='#c9d1d9', bg='#161b22').pack(pady=(10, 5))
            
            # Create color gradient bar
            gradient_width = 700
            gradient_height = 40
            gradient_img = Image.new('RGB', (gradient_width, gradient_height))
            draw = ImageDraw.Draw(gradient_img)
            
            # Draw gradient (COLORMAP_JET equivalent: Blue->Cyan->Green->Yellow->Red)
            for x in range(gradient_width):
                value = int((x / gradient_width) * 255)
                # Apply OpenCV JET colormap logic
                if value < 64:
                    r, g, b = 0, 0, 128 + value * 2
                elif value < 128:
                    r, g, b = 0, (value - 64) * 4, 255
                elif value < 192:
                    r, g, b = (value - 128) * 4, 255, 255 - (value - 128) * 4
                else:
                    r, g, b = 255, 255 - (value - 192) * 4, 0
                draw.line([(x, 0), (x, gradient_height)], fill=(r, g, b))
            
            gradient_photo = ImageTk.PhotoImage(gradient_img)
            gradient_label = tk.Label(legend_frame, image=gradient_photo, bg='#161b22')
            gradient_label.image = gradient_photo
            gradient_label.pack(pady=5)
            
            # Labels for spectrum
            label_frame = tk.Frame(legend_frame, bg='#161b22')
            label_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            tk.Label(label_frame, text="Low (0%)", font=('Arial', 9), 
                    fg='#8b949e', bg='#161b22').pack(side=tk.LEFT)
            tk.Label(label_frame, text="Medium (50%)", font=('Arial', 9), 
                    fg='#8b949e', bg='#161b22').pack(side=tk.LEFT, expand=True)
            tk.Label(label_frame, text="High (100%)", font=('Arial', 9), 
                    fg='#8b949e', bg='#161b22').pack(side=tk.RIGHT)
            
            ttk.Button(win, text="Close", command=win.destroy, style='Primary.TButton').pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate attention map: {e}")
    
    def show_knn_analysis(self):
        """Show kNN neighbor analysis"""
        if self.knn_neighbors is None:
            messagebox.showinfo("xAI", "No recognition data available. Detect a registered face first.")
            return
        
        try:
            neighbor_names, neighbor_distances = self.knn_neighbors;
            
            # Create analysis window
            win = tk.Toplevel(self.window)
            win.title("kNN Neighbor Analysis")
            win.geometry("500x600")
            win.configure(bg='#0d1117')
            
            # Header
            header = tk.Frame(win, bg='#161b22', relief='solid', borderwidth=1)
            header.pack(fill=tk.X, padx=10, pady=10)
            tk.Label(header, text="k-Nearest Neighbors Analysis", font=('Arial', 14, 'bold'), 
                    fg='#58a6ff', bg='#161b22').pack(pady=10)
            
            # Build proper explanation dict
            neighbor_labels = np.arange(len(neighbor_names))  # Use indices as labels
            neighbor_distances_array = np.array(neighbor_distances)
            
            # Get explanation dict with predicted_label
            explanation_dict = {
                'predicted_label': neighbor_names[0] if neighbor_names else "Unknown",
                'confidence': 0.85,  # Placeholder
                'confidence_level': 'High',
                'num_neighbors': len(neighbor_names),
                'num_matches': len(neighbor_names),
                'avg_distance': float(np.mean(neighbor_distances_array)) if len(neighbor_distances_array) > 0 else 0.0,
                'distance_variance': float(np.var(neighbor_distances_array)) if len(neighbor_distances_array) > 0 else 0.0,
                'separation': 0.5,  # Placeholder
                'vote_distribution': {neighbor_names[0]: len(neighbor_names)} if neighbor_names else {},
                'neighbor_labels': neighbor_names,
                'neighbor_distances': neighbor_distances
            }
            
            # Get explanation text
            explanation_text = get_knn_explanation_text(explanation_dict)
            
            # Content
            content = tk.Frame(win, bg='#161b22', relief='solid', borderwidth=1)
            content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            
            text_widget = tk.Text(content, wrap=tk.WORD, font=('Arial', 11), 
                                 bg='#161b22', fg='#c9d1d9', relief='flat', padx=15, pady=15)
            text_widget.pack(fill=tk.BOTH, expand=True)
            text_widget.insert(tk.END, explanation_text)
            text_widget.config(state=tk.DISABLED)
            
            ttk.Button(win, text="Close", command=win.destroy, style='Primary.TButton').pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate kNN analysis: {e}")
    
    def show_explanation(self):
        """Show comprehensive explanation of the recognition decision"""
        if self.current_explanation is None:
            messagebox.showinfo("xAI", "No explanation available. Detect a face first.")
            return
        
        try:
            exp = self.current_explanation
            
            # Create explanation window
            win = tk.Toplevel(self.window)
            win.title("Decision Explanation")
            win.geometry("600x700")
            win.configure(bg='#0d1117')
            
            # Header
            header = tk.Frame(win, bg='#161b22', relief='solid', borderwidth=1)
            header.pack(fill=tk.X, padx=10, pady=10)
            tk.Label(header, text="Recognition Decision Explained", font=('Arial', 14, 'bold'), 
                    fg='#58a6ff', bg='#161b22').pack(pady=10)
            
            # Content
            content = tk.Frame(win, bg='#0d1117')
            content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            
            # Decision box
            decision_color = '#3fb950' if exp.get('decision') == 'Accept' else '#f85149'
            decision_box = tk.Frame(content, bg=decision_color, relief='raised', borderwidth=2)
            decision_box.pack(fill=tk.X, pady=(0, 10))
            tk.Label(decision_box, text=f"Decision: {exp.get('decision', 'N/A')}", 
                    font=('Arial', 16, 'bold'), fg='#ffffff', bg=decision_color).pack(pady=15)
            
            # Details
            details = tk.Frame(content, bg='#161b22', relief='solid', borderwidth=1)
            details.pack(fill=tk.BOTH, expand=True)
            
            text = f"""
Confidence: {exp.get('confidence', 0):.1f}% ({exp.get('level', 'Unknown')})

Distance: {exp.get('distance', 0):.4f}
Threshold: {exp.get('threshold', 0):.4f}
Margin: {exp.get('margin_text', 'N/A')}

Message:
{exp.get('message', 'No additional information available.')}

Quality Assessment:
{exp.get('quality_message', 'Quality analysis not available.')}
"""
            
            text_widget = tk.Text(details, wrap=tk.WORD, font=('Arial', 11), 
                                 bg='#161b22', fg='#c9d1d9', relief='flat', padx=15, pady=15)
            text_widget.pack(fill=tk.BOTH, expand=True)
            text_widget.insert(tk.END, text)
            text_widget.config(state=tk.DISABLED)
            
            ttk.Button(win, text="Close", command=win.destroy, style='Primary.TButton').pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show explanation: {e}")
    
    def process_async_results(self, results, frame):
        """Process results from async face processor"""
        if not results:
            return frame
        
        processed_frame = frame.copy()
        
        # Clear previous face data
        self.face_identities.clear()
        self.face_confidences.clear()
        self.face_emotions.clear()
        self.face_liveness.clear()
        
        for result in results:
            face_id = result['face_id']
            bbox = result['bbox']
            embedding = result['embedding']
            is_live = result['is_live']
            liveness_confidence = result['liveness_confidence']
            
            # Process identity using vectorized kNN if available
            identity = "Processing..."
            confidence = 0.0
            distance = 1.0
            
            if not is_live:
                identity = "Spoof Detected"
                box_color = (0, 0, 255)  # Red
            elif self.vectorized_knn is not None:
                try:
                    predictions, confidences = self.vectorized_knn.predict_batch_with_confidence(
                        embedding.reshape(1, -1)
                    )
                    
                    if len(predictions) > 0:
                        prediction = predictions[0]
                        confidence = confidences[0]
                        distance = 1.0 - confidence
                        
                        # Map prediction to identity name
                        identity_names = list(employee_db.keys())
                        if 0 <= prediction < len(identity_names) and confidence >= CONFIDENCE_REJECTION_THRESHOLD:
                            identity = identity_names[prediction]
                            box_color = (0, 255, 0)  # Green for recognized
                            
                            # Log attendance
                            self.log_attendance(identity, confidence, "Neutral", "Real")
                        else:
                            identity = "Not Registered"
                            box_color = (0, 165, 255)  # Orange for unknown
                    
                except Exception as e:
                    print(f"[VECTORIZED KNN ERROR] {e}")
                    identity = "Error"
                    box_color = (0, 0, 255)  # Red for error
            else:
                identity = "Not Registered"
                box_color = (0, 165, 255)  # Orange
            
            # Store results for UI display
            self.face_identities[face_id] = identity
            self.face_confidences[face_id] = confidence
            self.face_emotions[face_id] = result.get('emotion', 'Neutral')
            self.face_liveness[face_id] = 'Real' if is_live else 'Spoof'
            
            # Draw bounding box and label
            x, y, w, h = bbox
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), box_color, 2)
            
            emotion = result.get('emotion', 'Neutral')
            liveness = 'Real' if is_live else 'Spoof'
            display_text = f"{identity} ({emotion} | {liveness})"
            cv2.putText(processed_frame, display_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        return processed_frame
    
    def queue_frame_for_display(self, frame):
        """Queue frame for display with error handling"""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Skip frame to prevent lag
            pass
    
    def update_ear_debug_display(self, current_ear, blink_count):
        """Update EAR debug display without freezing camera (thread-safe)"""
        try:
            # Update labels
            self.ear_current_label.config(text=f"Current EAR: {current_ear:.3f}")
            self.blink_count_label.config(text=f"Blinks: {blink_count}")
            
            # Update debug text with recent EAR values
            with self.ear_lock:
                if len(self.ear_history) > 0:
                    recent_ears = list(self.ear_history)[-20:]  # Last 20 values
                    ear_text = "Recent EAR values:\n"
                    ear_text += ", ".join([f"{ear:.2f}" for ear in recent_ears])
                    
                    # Add blink status
                    if current_ear < 0.5:
                        ear_text += "\n\nSTATUS: EYES CLOSED (EAR < 0.5)"
                    else:
                        ear_text += "\n\nSTATUS: Eyes open (EAR > 0.5)"
                    
                    # Update text widget
                    self.ear_debug_text.delete(1.0, tk.END)
                    self.ear_debug_text.insert(tk.END, ear_text)
                    
        except Exception as e:
            print(f"[EAR DEBUG] Update error: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.running:
            self.stop_camera()
        self.window.destroy()
    
    def run(self):
        """Run the GUI"""
        self.window.mainloop()


def main():
    print("=" * 70)
    print("FACE RECOGNITION ATTENDANCE SYSTEM")
    print("=" * 70)
    
    load_model_and_database()
    
    print("\nLaunching GUI...")
    print("=" * 70)
    
    app = AttendanceSystemGUI()
    app.run()


if __name__ == "__main__":
    main()
