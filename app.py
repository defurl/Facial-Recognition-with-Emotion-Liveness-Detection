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
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    DEVICE, IMG_SIZE, EMPLOYEE_DB_PATH, MODEL_METRIC_PATH,
    OPTIMAL_THRESHOLD_GUI, PROCESS_EVERY_N_FRAMES, CAMERA_INDEX, OUTPUT_DIR,
    PRIMARY_FACE_AREA_WEIGHT, PRIMARY_FACE_CENTER_WEIGHT, USE_MULTI_EMBEDDING
)
from models import FaceEmbeddingCNN
from data_loader import get_transforms
from utils import detect_faces, crop_face_with_padding
from emotion import analyze_emotion_and_liveness
from attendance import AttendanceLogger
from explainability import ExplainabilityEngine
from deep_knn import knn_predict_with_confidence, get_knn_explanation_text

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


def select_primary_face(faces, frame_width, frame_height):
    """
    Select primary face from multiple detections using area and centeredness scoring.
    
    Args:
        faces: List of (x, y, w, h) face bounding boxes
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
    
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
        
        # Score each face
        scores = []
        for x, y, w, h in faces:
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
            
            # Weighted combination (60% area, 40% centeredness)
            final_score = PRIMARY_FACE_AREA_WEIGHT * area_score + PRIMARY_FACE_CENTER_WEIGHT * center_score
            scores.append(final_score)
        
        # Return index of highest scoring face
        primary_idx = scores.index(max(scores))
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
        
        # Video capture
        self.cap = None
        self.running = False
        
        # Processing control - optimized for performance
        self.frame_count = 0
        self.PROCESS_EVERY_N_FRAMES = max(3, PROCESS_EVERY_N_FRAMES)  # Min 3 frames to reduce lag
        self.EMOTION_EVERY_N_FRAMES = 15  # Process emotion every 15 frames (~0.5 seconds) for faster updates
        self.last_processed_frame = 0  # Track last processed frame time
        self.last_emotion = "Neutral"
        self.last_liveness = "Unknown"
        self.last_liveness_confidence = 0.0  # Liveness detection confidence (0-1)
        self.last_identity = "Not Registered"
        self.last_distance = float('inf')
        self.last_confidence = 0.0  # Phase 7: Confidence percentage
        self.matched_pose_index = -1  # Phase 7: Which pose matched
        self.multiple_faces_warning = False
        
        # Confidence buffer system for state saving (demo improvement)
        self.CONFIDENCE_BUFFER_DURATION = 7.0  # Accumulate results for 7 seconds
        self.confidence_buffer = []  # List of (timestamp, identity, confidence, emotion, liveness, distance, pose_idx)
        self.locked_state = None  # Locked state: {'identity', 'confidence', 'emotion', 'liveness', 'distance', 'pose_idx', 'locked_at'}
        self.state_locked = False  # Whether state is currently locked
        self.no_face_frames = 0  # Counter for frames without face detection
        self.NO_FACE_RESET_THRESHOLD = 30  # Reset locked state after 30 frames (~1 second) without face
        
        # Phase 8: Attendance logger
        self.attendance_logger = AttendanceLogger(
            csv_path=OUTPUT_DIR / 'attendance_log.csv',
            cooldown_minutes=60
        )
        self.last_attendance_message = ""
        self.attendance_attempt_cache = {}  # Track last attempt time per person to prevent repeated I/O
        
        # Registration mode
        self.registration_mode = False
        self.registration_name = ""
        self.registration_state = None  # Will hold state machine data during registration
        
        # Thread-safe queue for frames - optimized queue size
        self.frame_queue = queue.Queue(maxsize=1)  # Smaller queue to reduce lag
        
        # Camera lock to prevent race conditions
        self.camera_lock = threading.Lock()
        
        # Emotion analysis failure tracking for graceful degradation
        self.emotion_failure_count = 0
        self.emotion_analysis_enabled = True
        
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
        """Setup dark theme UI with xAI visualization panels"""
        # Main container with dark theme
        main_container = tk.Frame(self.window, bg='#0d1117')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title header with accuracy display
        title_frame = tk.Frame(main_container, bg='#0d1117')
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Left: Title
        left_title = tk.Frame(title_frame, bg='#0d1117')
        left_title.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        title_label = ttk.Label(left_title, text="Face Recognition System - xAI Enhanced", 
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Right: Status indicator and model accuracy badge
        right_status = tk.Frame(title_frame, bg='#0d1117')
        right_status.pack(side=tk.RIGHT)
        
        # Model accuracy badge (prominent display)
        accuracy_badge = tk.Frame(right_status, bg='#1f6feb', relief='raised', borderwidth=2)
        accuracy_badge.pack(side=tk.LEFT, padx=(0, 15))
        tk.Label(accuracy_badge, text="Model Accuracy", font=('Arial', 9, 'bold'), 
                fg='#c9d1d9', bg='#1f6feb').pack(padx=10, pady=(5, 0))
        self.accuracy_display = tk.Label(accuracy_badge, text="96.2%", font=('Arial', 18, 'bold'), 
                                        fg='#3fb950', bg='#1f6feb')
        self.accuracy_display.pack(padx=10, pady=(0, 5))
        
        # Status indicator
        self.status_indicator = tk.Label(right_status, text="●", font=('Arial', 20), 
                                        fg='#f85149', bg='#0d1117')
        self.status_indicator.pack(side=tk.LEFT)
        
        # Main content area (3-column layout: video | info | xAI)
        content_frame = tk.Frame(main_container, bg='#0d1117')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (video + controls) - larger for visibility
        left_panel = tk.Frame(content_frame, bg='#0d1117')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video frame
        video_frame = ttk.LabelFrame(left_panel, text="LIVE CAMERA FEED")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
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
        
        # Controls frame
        controls_frame = ttk.LabelFrame(left_panel, text="CAMERA CONTROLS")
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        control_buttons_frame = tk.Frame(controls_frame, bg='#161b22')
        control_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = ttk.Button(control_buttons_frame, text="START", 
                                      command=self.start_camera, style='Primary.TButton', width=12)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(control_buttons_frame, text="STOP", 
                                     command=self.stop_camera, state=tk.DISABLED, style='Warning.TButton', width=12)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.register_button = ttk.Button(control_buttons_frame, text="REGISTER", 
                                         command=self.start_registration, state=tk.DISABLED, style='Success.TButton', width=12)
        self.register_button.pack(side=tk.LEFT)
        
        # Status frame
        status_frame = ttk.LabelFrame(left_panel, text="SYSTEM STATUS")
        status_frame.pack(fill=tk.X)
        
        status_container = tk.Frame(status_frame, bg='#161b22')
        status_container.pack(fill=tk.X, padx=10, pady=10)
        
        self.status_text = tk.StringVar(value="● System ready. Click 'Start' to begin recognition.")
        self.status_label = ttk.Label(status_container, textvariable=self.status_text, 
                                     style='Status.TLabel')
        self.status_label.pack(anchor=tk.W)
        
        # Middle panel (detection info + confidence gauges)
        middle_panel = tk.Frame(content_frame, bg='#0d1117', width=380)
        middle_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        middle_panel.pack_propagate(False)
        
        # Detection results
        detection_frame = ttk.LabelFrame(middle_panel, text="CURRENT DETECTION")
        detection_frame.pack(fill=tk.X, pady=(0, 10))
        
        detection_container = tk.Frame(detection_frame, bg='#161b22')
        detection_container.pack(fill=tk.X, padx=10, pady=10)
        
        # Identity display with larger, prominent text
        identity_frame = tk.Frame(detection_container, bg='#1f6feb', relief='raised', borderwidth=2)
        identity_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.identity_label = tk.Label(identity_frame, text="No face detected", 
                                      font=('Arial', 14, 'bold'), fg='#ffffff', bg='#1f6feb',
                                      padx=12, pady=15)
        self.identity_label.pack(fill=tk.X)
        
        # Detection details grid
        details_frame = tk.Frame(detection_container, bg='#161b22')
        details_frame.pack(fill=tk.X)
        
        # Create 2x2 grid for metrics
        for i in range(2):
            details_frame.grid_columnconfigure(i, weight=1)
        
        # Emotion
        emotion_container = tk.Frame(details_frame, bg='#21262d', relief='solid', borderwidth=1)
        emotion_container.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        tk.Label(emotion_container, text="Emotion", font=('Arial', 8, 'bold'), 
                bg='#21262d', fg='#8b949e').pack(pady=(5, 0))
        self.emotion_var = tk.StringVar(value="Neutral")
        tk.Label(emotion_container, textvariable=self.emotion_var, font=('Arial', 11, 'bold'), 
                bg='#21262d', fg='#3fb950').pack(pady=(0, 5))
        
        # Liveness
        liveness_container = tk.Frame(details_frame, bg='#21262d', relief='solid', borderwidth=1)
        liveness_container.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        tk.Label(liveness_container, text="Liveness", font=('Arial', 8, 'bold'), 
                bg='#21262d', fg='#8b949e').pack(pady=(5, 0))
        self.liveness_var = tk.StringVar(value="Unknown")
        self.liveness_label = tk.Label(liveness_container, textvariable=self.liveness_var, 
                                      font=('Arial', 11, 'bold'), bg='#21262d', fg='#58a6ff')
        self.liveness_label.pack(pady=(0, 0))
        # Liveness confidence percentage
        self.liveness_confidence_var = tk.StringVar(value="")
        tk.Label(liveness_container, textvariable=self.liveness_confidence_var, 
                font=('Arial', 8), bg='#21262d', fg='#8b949e').pack(pady=(0, 5))
        
        # Distance
        distance_container = tk.Frame(details_frame, bg='#21262d', relief='solid', borderwidth=1)
        distance_container.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        tk.Label(distance_container, text="Distance", font=('Arial', 8, 'bold'), 
                bg='#21262d', fg='#8b949e').pack(pady=(5, 0))
        self.distance_var = tk.StringVar(value="N/A")
        tk.Label(distance_container, textvariable=self.distance_var, font=('Arial', 11, 'bold'), 
                bg='#21262d', fg='#a371f7').pack(pady=(0, 5))
        
        # Confidence
        confidence_container = tk.Frame(details_frame, bg='#21262d', relief='solid', borderwidth=1)
        confidence_container.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        self.confidence_var = tk.StringVar(value="N/A")

        # xAI Explainability Controls
        xai_frame = ttk.LabelFrame(middle_panel, text="EXPLAINABILITY (xAI)")
        xai_frame.pack(fill=tk.X, pady=(0, 10))
        
        xai_container = tk.Frame(xai_frame, bg='#161b22')
        xai_container.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(xai_container, text="Show Attention Map", 
                  command=self.show_attention_map, style='XAI.TButton').pack(fill=tk.X, pady=(0, 5))
        ttk.Button(xai_container, text="kNN Neighbor Analysis", 
                  command=self.show_knn_analysis, style='XAI.TButton').pack(fill=tk.X, pady=(0, 5))
        ttk.Button(xai_container, text="Explain Decision", 
                  command=self.show_explanation, style='XAI.TButton').pack(fill=tk.X)
        
        # Debug panel (compact)
        debug_frame = ttk.LabelFrame(middle_panel, text="VERIFICATION DEBUG")
        debug_frame.pack(fill=tk.X, pady=(0, 10))
        
        debug_container = tk.Frame(debug_frame, bg='#161b22')
        debug_container.pack(fill=tk.X, padx=10, pady=10)
        
        # Matched pose
        tk.Label(debug_container, text="Matched Pose:", font=('Arial', 9), 
                bg='#161b22', fg='#8b949e').pack(anchor=tk.W)
        self.pose_var = tk.StringVar(value="N/A")
        tk.Label(debug_container, textvariable=self.pose_var, font=('Arial', 10, 'bold'), 
                bg='#161b22', fg='#58a6ff').pack(anchor=tk.W, pady=(0, 8))
        
        # Threshold display
        self.threshold_display = tk.Label(debug_container, 
                                         text=f"Threshold: {OPTIMAL_THRESHOLD_GUI:.3f}", 
                                         font=('Arial', 9), bg='#161b22', fg='#8b949e')
        self.threshold_display.pack(anchor=tk.W)
        
        # Right panel (employee management + stats)
        right_panel = tk.Frame(content_frame, bg='#0d1117', width=340)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # Employee management
        employee_frame = ttk.LabelFrame(right_panel, text="EMPLOYEE MANAGEMENT")
        employee_frame.pack(fill=tk.X, pady=(0, 10))
        
        emp_container = tk.Frame(employee_frame, bg='#161b22')
        emp_container.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(emp_container, text="View Employees", command=self.view_employees).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(emp_container, text="View Attendance", command=self.view_attendance).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(emp_container, text="Edit Employee", command=self.edit_employee).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(emp_container, text="Delete Employee", command=self.delete_employee, style='Danger.TButton').pack(fill=tk.X, pady=(0, 5))
        ttk.Button(emp_container, text="Adjust Threshold", command=self.adjust_threshold, style='Warning.TButton').pack(fill=tk.X, pady=(0, 5))
        ttk.Button(emp_container, text="Reset State Lock", command=self.reset_state_lock, style='Warning.TButton').pack(fill=tk.X)
        
        # Statistics with prominent metrics
        stats_frame = ttk.LabelFrame(right_panel, text="SESSION STATISTICS")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        stats_container = tk.Frame(stats_frame, bg='#161b22')
        stats_container.pack(fill=tk.X, padx=10, pady=10)
        
        # Metric boxes for key stats
        metrics_grid = tk.Frame(stats_container, bg='#161b22')
        metrics_grid.pack(fill=tk.X)
        
        # Total detections
        det_box = tk.Frame(metrics_grid, bg='#21262d', relief='solid', borderwidth=1)
        det_box.pack(fill=tk.X, pady=(0, 5))
        tk.Label(det_box, text="Total Detections", font=('Arial', 8), 
                bg='#21262d', fg='#8b949e').pack(pady=(5, 0))
        self.total_det_label = tk.Label(det_box, text="0", font=('Arial', 16, 'bold'), 
                                       bg='#21262d', fg='#58a6ff')
        self.total_det_label.pack(pady=(0, 5))
        
        # Success rate
        success_box = tk.Frame(metrics_grid, bg='#21262d', relief='solid', borderwidth=1)
        success_box.pack(fill=tk.X, pady=(0, 5))
        tk.Label(success_box, text="Recognition Rate", font=('Arial', 8), 
                bg='#21262d', fg='#8b949e').pack(pady=(5, 0))
        self.success_rate_label = tk.Label(success_box, text="0.0%", font=('Arial', 16, 'bold'), 
                                          bg='#21262d', fg='#3fb950')
        self.success_rate_label.pack(pady=(0, 5))
        
        # Unique faces
        unique_box = tk.Frame(metrics_grid, bg='#21262d', relief='solid', borderwidth=1)
        unique_box.pack(fill=tk.X, pady=(0, 5))
        tk.Label(unique_box, text="Unique Faces Today", font=('Arial', 8), 
                bg='#21262d', fg='#8b949e').pack(pady=(5, 0))
        self.unique_faces_label = tk.Label(unique_box, text="0", font=('Arial', 16, 'bold'), 
                                          bg='#21262d', fg='#a371f7')
        self.unique_faces_label.pack(pady=(0, 5))
        
        # Database size
        db_box = tk.Frame(metrics_grid, bg='#21262d', relief='solid', borderwidth=1)
        db_box.pack(fill=tk.X)
        tk.Label(db_box, text="Employees in DB", font=('Arial', 8), 
                bg='#21262d', fg='#8b949e').pack(pady=(5, 0))
        self.db_size_label = tk.Label(db_box, text=str(len(employee_db)), font=('Arial', 16, 'bold'), 
                                      bg='#21262d', fg='#d29922')
        self.db_size_label.pack(pady=(0, 5))
    
        
        # Configure grid weights for responsive design
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
    
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
    
    def _open_camera_with_fallback(self):
        """Try camera 1 first (user's camera), then fallback to others"""
        # Camera 1 is the user's actual camera - prioritize it
        preferred_indices = [1, CAMERA_INDEX, 0, 2]  # Try 1 first!
        
        for idx in preferred_indices:
            cap = None
            try:
                print(f"Trying camera {idx}...")
                cap = cv2.VideoCapture(idx)
                
                if cap is None:
                    continue
                    
                if not cap.isOpened():
                    cap.release()
                    continue
                
                # Wait for camera to initialize
                time.sleep(0.8)
                
                # Test if we can actually read a frame
                try:
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        print(f"[OK] Successfully opened camera {idx}")
                        return cap
                except Exception as read_err:
                    print(f"  Frame read error on camera {idx}: {read_err}")
                
                # Release and don't retry to avoid memory issues
                cap.release()
                cap = None
                time.sleep(0.3)
                    
            except Exception as e:
                print(f"  Error with camera {idx}: {e}")
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass
                continue
        
        print("[ERROR] Failed to open any camera")
        return None

    def start_camera(self):
        """Start camera and begin processing"""
        # Update status to show we're trying
        self.status_text.set("● Initializing camera...")
        self.window.update_idletasks()
        
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
            messagebox.showerror("Camera Error", error_msg)
            self.status_text.set("● Camera initialization failed. Check connection.")
            return
        
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
        
        # Reset confidence buffer when camera starts
        self.confidence_buffer = []
        self.locked_state = None
        self.state_locked = False
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
        
        self.video_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.video_thread.start()
        self.update_display()
    
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
        """Stop camera"""
        self.running = False
        
        # Wait for capture thread to finish
        time.sleep(0.2)
        
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

        inst_text = """• Look directly at the camera\n• Make sure you have good lighting\n• Keep your face centered and still\n• Follow the on-screen pose instructions\n• Hold each pose steady when prompted\n• Registration captures 5 different poses"""

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
            from config import (
                REGISTRATION_POSES_FULL, REGISTRATION_INSTRUCTIONS_FULL
            )
            
            # Use 5 poses (center, left, right, up, down) for better coverage
            # Skip the final "center again" from full mode
            poses = REGISTRATION_POSES_FULL[:5]
            instructions = REGISTRATION_INSTRUCTIONS_FULL[:5]
            
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
            self.status_text.set(f"🔵 Step 1/{len(poses)}: {instructions[0]}")
            dialog.destroy()
        
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
                
                print(f"[OK] Registered '{self.registration_name}' with {len(self.registration_state['embeddings'])} poses")
                self.status_text.set(f"[OK] Registered '{self.registration_name}' successfully!")
                messagebox.showinfo("Success", f"Employee '{self.registration_name}' registered!")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
            finally:
                self.registration_mode = False
                self.registration_state = None
        
        def cancel_registration():
            self.registration_mode = False
            self.registration_state = None
            self.status_text.set("Registration cancelled")
            dialog.destroy()
        
        ttk.Button(button_frame, text="✓ Save to Database", command=save_to_database,
                  style='Success.TButton', width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="✗ Cancel", command=cancel_registration,
                  style='Danger.TButton', width=15).pack(side=tk.LEFT, padx=5)
    
    def capture_frames(self):
        """Capture and process video frames (runs in background thread) - optimized"""
        fps_counter = 0
        fps_start_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 30  # Stop after 30 consecutive failures (~1 second)
        
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
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[ERROR] Too many consecutive frame read errors ({consecutive_errors}). Stopping camera.")
                        self.window.after(0, lambda: self.handle_camera_failure())
                        break
                    time.sleep(0.01)
                    continue
                
                # Reset error counter on successful read
                consecutive_errors = 0
                
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
            fps_counter += 1
            
            # Calculate FPS every second
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                fps = fps_counter / (current_time - fps_start_time)
                # Display FPS on frame for debugging
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                fps_counter = 0
                fps_start_time = current_time
            
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
                
                # Check every 15 frames (~0.5 seconds at 30fps) to reduce flickering
                if self.frame_count - state['last_check_frame'] >= 15:
                    state['last_check_frame'] = self.frame_count
                    
                    faces = detect_faces(frame)
                    if len(faces) == 0:
                        cv2.putText(frame, "No face detected", (10, h - 40), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.65, (231, 76, 60), 1, cv2.LINE_AA)
                        state['hold_frames'] = 0
                    else:
                        x, y, w, h = faces[0]
                        cropped_face = crop_face_with_padding(frame, x, y, w, h)
                        
                        if cropped_face.size == 0 or cropped_face.shape[0] < 50:
                            cv2.putText(frame, "Face too small", (10, h - 40), 
                                       cv2.FONT_HERSHEY_DUPLEX, 0.65, (231, 76, 60), 1, cv2.LINE_AA)
                            state['hold_frames'] = 0
                        else:
                            # Import quality check functions
                            from utils import check_image_blur, check_image_lighting
                            
                            # Quick quality checks
                            blur_var, blur_ok, _ = check_image_blur(cropped_face, threshold=40)
                            brightness, contrast, lighting_ok, _ = check_image_lighting(cropped_face, 25, 230, 30)
                            
                            if blur_ok and lighting_ok:
                                state['hold_frames'] += 1
                                remaining = 5 - state['hold_frames']
                                
                                if remaining > 0:
                                    cv2.putText(frame, f"Hold steady... {remaining}", (10, frame.shape[0] - 40),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                                else:
                                    # Capture this pose
                                    cv2.putText(frame, "Captured!", (10, h - 40), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                                    
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
                                if not blur_ok:
                                    cv2.putText(frame, "Image too blurry", (10, h - 40), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 165, 0), 1, cv2.LINE_AA)
                                elif not lighting_ok:
                                    cv2.putText(frame, "Poor lighting", (10, h - 40), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 165, 0), 1, cv2.LINE_AA)
                else:
                    # Show persistent hold counter at bottom (lighter weight)
                    if state.get('hold_frames', 0) > 0:
                        remaining = max(0, 5 - state['hold_frames'])
                        cv2.putText(frame, f"Hold steady... {remaining}", (10, h - 40), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            
            faces = detect_faces(frame)
            h, w = frame.shape[:2]
            
            # Track if multiple faces detected
            self.multiple_faces_warning = len(faces) > 1
            
            # Handle no face detection - reset locked state after threshold
            if len(faces) == 0:
                self.no_face_frames += 1
                if self.no_face_frames > self.NO_FACE_RESET_THRESHOLD:
                    if self.state_locked:
                        print("[State] No face detected for too long, resetting locked state")
                        self.state_locked = False
                        self.locked_state = None
                        self.confidence_buffer = []
            else:
                self.no_face_frames = 0  # Reset counter when face detected
            
            # Select primary face if multiple faces  
            primary_face_idx = select_primary_face(faces, w, h)

            for face_idx, (x, y, w, h) in enumerate(faces):
                is_primary = (face_idx == primary_face_idx)
                box_color = (128, 128, 128) if not is_primary else (0, 255, 0)

                # Apply locked state values FIRST if state is locked
                if is_primary and self.state_locked and self.locked_state and isinstance(self.locked_state, dict):
                    try:
                        self.last_identity = self.locked_state.get('identity', 'Not Registered')
                        self.last_confidence = self.locked_state.get('confidence', 0.0)
                        self.last_emotion = self.locked_state.get('emotion', 'Neutral')
                        self.last_liveness = self.locked_state.get('liveness', 'Unknown')
                        self.last_distance = self.locked_state.get('distance', float('inf'))
                        self.matched_pose_index = self.locked_state.get('pose_idx', -1)
                        box_color = (0, 255, 0) if self.last_identity != "Not Registered" else (0, 0, 255)
                    except Exception as e:
                        print(f"[WARNING] Error applying locked state: {e}")
                        self.state_locked = False
                        self.locked_state = None

                # Skip verification during registration mode - always show as unregistered
                if self.registration_mode:
                    if is_primary:
                        self.last_identity = "Registering..."
                        box_color = (255, 165, 0)  # Orange for registration
                # Only process primary face for verification when NOT in registration mode AND not locked
                elif is_primary and not self.state_locked and self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0:
                    cropped_face = crop_face_with_padding(frame, x, y, w, h)

                    if cropped_face.size > 0 and cropped_face.shape[0] >= 50:
                        cropped_face_resized = cv2.resize(cropped_face, (IMG_SIZE, IMG_SIZE))

                        # Multi-method liveness detection + emotion analysis
                        is_live = True  # Default to Real
                        liveness_confidence = 0.0
                        # Check emotion and liveness independently of PROCESS_EVERY_N_FRAMES
                        should_check_emotion = (self.frame_count % self.EMOTION_EVERY_N_FRAMES == 0)
                        if should_check_emotion and self.emotion_analysis_enabled:
                            try:
                                rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                                emo, is_live, liveness_confidence, liveness_details = analyze_emotion_and_liveness(rgb_face)
                                self.last_emotion = emo
                                self.last_liveness = 'Real' if is_live else 'Spoof'
                                self.last_liveness_confidence = liveness_confidence  # Store for UI
                                self.emotion_failure_count = 0  # Reset on success
                                
                                # Log with liveness confidence
                                print(f"[Emotion] {emo} | Liveness: {'Real' if is_live else 'Spoof'} "
                                      f"({liveness_confidence:.1%} confidence)")
                                if not is_live:
                                    print(f"[Liveness Details] {liveness_details}")
                            except Exception as e:
                                self.emotion_failure_count += 1
                                print(f"[WARNING] Emotion/liveness analysis error ({self.emotion_failure_count}/10): {e}")
                                
                                # Graceful degradation: disable after 10 consecutive failures
                                if self.emotion_failure_count >= 10:
                                    self.emotion_analysis_enabled = False
                                    print("[ERROR] Emotion/liveness analysis disabled due to repeated failures. Recognition will continue.")
                                
                                is_live = True
                                self.last_liveness = 'Real'
                        else:
                            # Reuse last liveness result between emotion checks
                            is_live = (self.last_liveness == 'Real')

                        if not is_live:
                            self.last_identity = "Spoof Detected"
                            box_color = (0, 0, 255)
                        else:
                            # Verification with multi-embedding support
                            try:
                                    # Avoid disk I/O: convert cv2 image to PIL directly
                                    rgb = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2RGB)
                                    pil_image = Image.fromarray(rgb).convert('RGB')
                                    image_tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)

                                    with torch.no_grad():
                                        trial_embedding = verification_model(image_tensor, mode='metric').cpu()

                                    min_distance = float('inf')
                                    self.last_identity = "Not Registered"
                                    best_match = None

                                    # Multi-embedding comparison
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

                                    # Load current threshold (may have been adjusted by user)
                                    current_threshold = _load_gui_threshold(OPTIMAL_THRESHOLD_GUI)
                                    
                                    # Calculate confidence and prepare result
                                    threshold = _load_gui_threshold(OPTIMAL_THRESHOLD_GUI)
                                    confidence = max(0, min(100, (1 - min_distance / threshold) * 100))
                                    
                                    # Determine identity for this frame
                                    frame_identity = best_match if min_distance < current_threshold else "Not Registered"
                                    
                                    # Add result to confidence buffer (if not locked yet)
                                    if not self.state_locked:
                                        current_time = time.time()
                                        self.confidence_buffer.append((
                                            current_time,
                                            frame_identity,
                                            confidence,
                                            self.last_emotion,
                                            self.last_liveness,
                                            min_distance,
                                            -1  # pose_idx will be set below if applicable
                                        ))
                                        
                                        # Check if we have enough samples to lock
                                        if len(self.confidence_buffer) >= 20:  # At least 20 samples collected
                                            # Check if enough time has passed since first sample
                                            oldest_time = min(entry[0] for entry in self.confidence_buffer)
                                            time_elapsed = current_time - oldest_time
                                            
                                            # Lock after collecting samples for at least 5 seconds
                                            if time_elapsed >= 5.0:
                                                try:
                                                    # Use voting-based approach: pick identity that appears most frequently
                                                    # Group by identity and count occurrences with confidence weighting
                                                    identity_votes = {}
                                                    identity_data = {}  # Store best sample for each identity
                                                    
                                                    for entry in self.confidence_buffer:
                                                        if not entry or len(entry) < 7:
                                                            continue  # Skip malformed entries
                                                        identity = entry[1]
                                                        confidence = entry[2]
                                                        
                                                        # Count votes (frequency)
                                                        if identity not in identity_votes:
                                                            identity_votes[identity] = 0
                                                            identity_data[identity] = []
                                                        
                                                        identity_votes[identity] += 1
                                                        identity_data[identity].append(entry)
                                                    
                                                    # Ensure we have valid data
                                                    if not identity_votes:
                                                        print("[WARNING] No valid votes in buffer, skipping lock")
                                                        self.confidence_buffer = []
                                                        continue
                                                    
                                                    # Pick identity with most votes
                                                    most_common_identity = max(identity_votes, key=identity_votes.get)
                                                    vote_count = identity_votes[most_common_identity]
                                                    
                                                    # From that identity, pick the sample with highest confidence
                                                    best_entry = max(identity_data[most_common_identity], key=lambda x: x[2])
                                                    
                                                    self.locked_state = {
                                                        'identity': str(best_entry[1]),
                                                        'confidence': float(best_entry[2]),
                                                        'emotion': str(best_entry[3]),
                                                        'liveness': str(best_entry[4]),
                                                        'distance': float(best_entry[5]),
                                                        'pose_idx': int(best_entry[6]),
                                                        'locked_at': current_time
                                                    }
                                                    self.state_locked = True
                                                    print(f"[State] Locked: {self.locked_state['identity']} (votes: {vote_count}/{len(self.confidence_buffer)}, confidence: {self.locked_state['confidence']:.1f}%)")
                                                    
                                                    # Clear buffer after locking
                                                    self.confidence_buffer = []
                                                except Exception as e:
                                                    print(f"[ERROR] Failed to lock state: {e}")
                                                    self.confidence_buffer = []
                                            else:
                                                # Still accumulating - show progress
                                                print(f"[State] Accumulating: {len(self.confidence_buffer)} samples over {time_elapsed:.1f}s (need 5s)")
                                    
                                    # Use locked state if available, otherwise use current frame result
                                    if self.state_locked and self.locked_state and isinstance(self.locked_state, dict):
                                        try:
                                            self.last_identity = self.locked_state.get('identity', 'Not Registered')
                                            self.last_confidence = self.locked_state.get('confidence', 0.0)
                                            self.last_emotion = self.locked_state.get('emotion', 'Neutral')
                                            self.last_liveness = self.locked_state.get('liveness', 'Unknown')
                                            self.last_distance = self.locked_state.get('distance', float('inf'))
                                            self.matched_pose_index = self.locked_state.get('pose_idx', -1)
                                            box_color = (0, 255, 0) if self.last_identity != "Not Registered" else (0, 0, 255)
                                        except Exception as e:
                                            print(f"[WARNING] Error reading locked state: {e}")
                                            self.state_locked = False
                                            self.locked_state = None
                                            # Fall back to current frame
                                            self.last_identity = frame_identity
                                            self.last_confidence = confidence
                                            self.last_distance = min_distance
                                            box_color = (0, 255, 0) if frame_identity != "Not Registered" else (0, 0, 255)
                                    else:
                                        # Use current frame result (accumulation phase)
                                        self.last_identity = frame_identity
                                        self.last_confidence = confidence
                                        self.last_distance = min_distance
                                        box_color = (0, 255, 0) if frame_identity != "Not Registered" else (0, 0, 255)
                                        
                                        # Debug output during accumulation
                                        if self.frame_count % 30 == 0:
                                            buffer_size = len(self.confidence_buffer)
                                            if buffer_size > 0:
                                                print(f"[State] Accumulating: {buffer_size} samples, Identity: {frame_identity}, Confidence: {confidence:.1f}%")
                                    
                                    # Trigger animations only when not locked
                                    if not self.state_locked:
                                        if min_distance < current_threshold:
                                            if not self.verification_animation['active']:
                                                self.verification_animation = {
                                                    'active': True,
                                                    'type': 'success',
                                                    'frame_count': 0,
                                                    'max_frames': 30
                                                }
                                        else:
                                            if not self.verification_animation['active']:
                                                self.verification_animation = {
                                                    'active': True,
                                                    'type': 'failure',
                                                    'frame_count': 0,
                                                    'max_frames': 30
                                                }
                                    
                                    # xAI: Store face data for explainability features
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
                                    
                                    # Phase 8: Auto-mark attendance for recognized faces
                                    if self.last_identity != "Not Registered":
                                        # Only attempt once per cooldown period to avoid repeated I/O
                                        current_time = time.time()
                                        last_attempt = self.attendance_attempt_cache.get(self.last_identity, 0)
                                        
                                        # Attempt marking only if it's been more than 5 seconds since last attempt
                                        if current_time - last_attempt > 5:
                                            self.attendance_attempt_cache[self.last_identity] = current_time
                                            
                                            # Run attendance marking in background thread to avoid blocking GUI
                                            def mark_async():
                                                try:
                                                    success, message = self.attendance_logger.mark_attendance(
                                                        self.last_identity, min_distance, self.last_emotion, self.last_liveness
                                                    )
                                                    self.last_attendance_message = message
                                                    if success:
                                                        print(f"✓ {message}")
                                                    else:
                                                        print(f"ℹ {message}")
                                                except Exception as e:
                                                    print(f"Attendance marking error: {e}")
                                            
                                            threading.Thread(target=mark_async, daemon=True).start()
                            except Exception as e:
                                print(f"Verification error: {e}")
                                self.last_identity = "Error"
                    else:
                        self.last_identity = "Face too small"
                        box_color = (0, 0, 255)
                elif not is_primary:
                    # Non-primary faces shown in gray
                    self.last_identity = "Secondary Face"

                # Display proper label for each face
                if is_primary:
                    if self.registration_mode:
                        display_text = "Registering..."
                    else:
                        display_text = f"{self.last_identity} ({self.last_emotion} | {self.last_liveness})"
                else:
                    display_text = "Secondary Face"
                
                # Smooth color transition for primary face
                if is_primary and hasattr(self, 'box_color_transition'):
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
                
                # Draw modern rounded rectangle with thicker line for recognized faces
                thickness = 3 if (is_primary and self.last_identity not in ["Not Registered", "Error", "Spoof Detected", "Face too small", "Registering..."]) else 2
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
                
                # Update statistics (primary face only)
                if is_primary:
                    self.recognition_stats['total_detections'] += 1
                    if self.last_identity not in ["Not Registered", "Error", "Spoof Detected", "Face too small", "Secondary Face"]:
                        self.recognition_stats['successful_recognitions'] += 1
                        self.recognition_stats['unique_faces_today'].add(self.last_identity)
            
            # Display state accumulation/lock status indicator
            if not self.registration_mode:
                h, w = frame.shape[:2]
                font_status = cv2.FONT_HERSHEY_DUPLEX
                
                if self.state_locked:
                    # Show locked state indicator with lock icon
                    status_text = "LOCKED"
                    status_color = (46, 204, 113)  # Green
                    (tw, th), _ = cv2.getTextSize(status_text, font_status, 0.6, 2)
                    
                    # Draw indicator in top-right corner
                    x_pos = w - tw - 40
                    y_pos = 10
                    
                    overlay = frame.copy()
                    draw_rounded_rectangle(overlay, (x_pos - 10, y_pos), 
                                         (x_pos + tw + 20, y_pos + th + 16), 
                                         status_color, -1, radius=10)
                    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                    cv2.putText(frame, status_text, (x_pos + 5, y_pos + th + 5), 
                               font_status, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                elif len(self.confidence_buffer) > 0:
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
            
            # Display warning banner if multiple faces detected
            if self.multiple_faces_warning and len(faces) > 1:
                warning_text = "⚠ Multiple faces - processing primary only"
                font_warn = cv2.FONT_HERSHEY_DUPLEX
                (tw, th), _ = cv2.getTextSize(warning_text, font_warn, 0.6, 1)
                overlay = frame.copy()
                draw_rounded_rectangle(overlay, (5, 5), (tw + 30, th + 22), (255, 165, 0), -1, radius=10)
                cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
                cv2.putText(frame, warning_text, (18, th + 14), 
                           font_warn, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Only put frame if queue is not full (prevents backup and lag)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                # Skip this frame to prevent lag
                pass

    def update_display(self):
        """Update display on main thread - optimized for performance"""
        if not self.running:
            return

        try:
            # Clear queue if backed up to prevent lag
            frame = None
            while not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            if frame is not None:
                # Resize frame to fit display window
                frame_resized = cv2.resize(frame, (self.video_width, self.video_height))

                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk, text="")

                # Update UI elements less frequently (every 3 display updates)
                if self.frame_count % 3 == 0:
                    self.update_detection_display()
                    self.update_stats_display()
                    self.update_debug_display()
            
        except Exception as e:
            print(f"Display error: {e}")

        # Reduced update frequency to 40ms (25 FPS) for better performance
        self.window.after(40, self.update_display)
    
    def update_detection_display(self):
        """Update the detection information display"""
        if self.last_identity != "Not Registered" and self.last_identity != "Error":
            # Successful recognition
            lock_indicator = " 🔒" if self.state_locked else ""
            self.identity_label.config(text=f"✅ {self.last_identity}{lock_indicator}", 
                                     bg='#d5f4e6', fg='#27ae60')
            if self.state_locked:
                status_text = f"Locked: {self.last_identity}"
            else:
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
        ttk.Button(content, text="Close", command=dialog.destroy).pack(pady=(10, 0))
    
    def reset_state_lock(self):
        """Reset the locked state to allow re-accumulation"""
        if self.state_locked:
            self.state_locked = False
            self.locked_state = None
            self.confidence_buffer = []
            messagebox.showinfo("🔓 State Unlocked", 
                              "Recognition state has been reset.\n\n"
                              "The system will now re-accumulate verification data\n"
                              "for the next 7 seconds before locking again.")
            print("[State] Manually reset by user")
        else:
            messagebox.showinfo("ℹ️ State Not Locked", 
                              "The recognition state is currently not locked.\n\n"
                              "The system is already accumulating data.")
    
    def edit_employee(self):
        """Batch edit employees with checkboxes"""
        if len(employee_db) == 0:
            messagebox.showinfo("✏ Edit Employees", "No employees registered yet.")
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
        
        tk.Label(header, text="✏ Batch Edit Employee Names", font=('Segoe UI Semibold', 14, 'bold'), 
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
            for old_name in selected:
                new_name = simpledialog.askstring("✏ Rename Employee", 
                                                 f"Enter new name for '{old_name}':",
                                                 parent=dialog)
                if new_name and new_name.strip() and new_name != old_name:
                    new_name = new_name.strip()
                    if new_name in employee_db:
                        messagebox.showerror("❌ Error", f"Employee '{new_name}' already exists! Skipping.")
                        continue
                    
                    employee_db[new_name] = employee_db.pop(old_name)
                    renamed_count += 1
            
            if renamed_count > 0:
                torch.save(employee_db, EMPLOYEE_DB_PATH)
                messagebox.showinfo("✅ Success", f"Successfully renamed {renamed_count} employee(s)!")
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
        
        update_display()
        
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
                                         f"Are you sure you want to delete {len(selected)} employee(s)?\\n\\n" +
                                         "\\n".join(f"• {name}" for name in selected[:5]) +
                                         (f"\\n... and {len(selected) - 5} more" if len(selected) > 5 else ""),
                                         icon='warning')
            if confirm:
                for name in selected:
                    del employee_db[name]
                
                torch.save(employee_db, EMPLOYEE_DB_PATH)
                messagebox.showinfo("✅ Success", f"Successfully deleted {len(selected)} employee(s)!")
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
                                        f"Are you sure you want to delete '{name}'?\n\nThis action cannot be undone!", 
                                        parent=dialog)
            
            if confirm:
                del employee_db[name]
                torch.save(employee_db, EMPLOYEE_DB_PATH)
                messagebox.showinfo("✅ Deleted", f"Employee '{name}' has been deleted")
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
            neighbor_names, neighbor_distances = self.knn_neighbors
            
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
