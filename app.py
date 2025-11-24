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

import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
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
        print(f"✓ Loaded model from {MODEL_METRIC_PATH}")
    else:
        print(f"⚠ Warning: Model not found at {MODEL_METRIC_PATH}")
        print("  Please train the model first using: python scripts/train_metric.py")
    
    # Load transforms
    _, val_transform = get_transforms()
    
    # Load database
    if EMPLOYEE_DB_PATH.exists():
        employee_db = torch.load(EMPLOYEE_DB_PATH)
        print(f"✓ Loaded {len(employee_db)} employees from database")
        for name in employee_db:
            print(f"    • {name}")
    else:
        print("ℹ No existing employee database. Starting fresh.")
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
        self.window.title("🎭 Face Recognition Attendance System")
        self.window.geometry("1200x800")
        self.window.minsize(1000, 700)
        self.window.configure(bg='#f8f9fa')
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure styles
        self.setup_styles()
        
        # Video capture
        self.cap = None
        self.running = False
        
        # Processing control
        self.frame_count = 0
        self.PROCESS_EVERY_N_FRAMES = PROCESS_EVERY_N_FRAMES
        self.EMOTION_EVERY_N_FRAMES = 30  # Process emotion every 30 frames (~1 second) to reduce lag
        self.last_emotion = "Neutral"
        self.last_liveness = "Unknown"
        self.last_identity = "Not Registered"
        self.last_distance = float('inf')
        self.last_confidence = 0.0  # Phase 7: Confidence percentage
        self.matched_pose_index = -1  # Phase 7: Which pose matched
        self.multiple_faces_warning = False
        
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
        
        # Thread-safe queue for frames
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Recognition statistics
        self.recognition_stats = {
            'total_detections': 0,
            'successful_recognitions': 0,
            'unique_faces_today': set()
        }
        
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
        """Setup modern TTK styles with sharp, clean fonts"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure custom styles with sharp, modern fonts
        self.style.configure('Title.TLabel', font=('Segoe UI Semibold', 20, 'bold'), foreground='#1a1a2e')
        self.style.configure('Header.TLabel', font=('Segoe UI Semibold', 13, 'bold'), foreground='#16213e')
        self.style.configure('Status.TLabel', font=('Segoe UI', 11), foreground='#0f3460')
        self.style.configure('Error.TLabel', font=('Segoe UI Semibold', 11), foreground='#e94560')
        self.style.configure('Info.TLabel', font=('Segoe UI', 9), foreground='#6c757d')
        
        # Modern button styles with sharp fonts
        self.style.configure('Primary.TButton', 
                           font=('Segoe UI Semibold', 10, 'bold'),
                           background='#4a90e2',
                           foreground='white',
                           borderwidth=0,
                           relief='flat',
                           padding=(20, 10))
        self.style.map('Primary.TButton',
                      background=[('active', '#3a7bd5'), ('pressed', '#2d5f9f')])
        
        self.style.configure('Success.TButton', 
                           font=('Segoe UI Semibold', 10, 'bold'),
                           background='#2ecc71',
                           foreground='white',
                           borderwidth=0,
                           relief='flat',
                           padding=(20, 10))
        self.style.map('Success.TButton',
                      background=[('active', '#27ae60'), ('pressed', '#1e8449')])
        
        self.style.configure('Warning.TButton', 
                           font=('Segoe UI', 10),
                           background='#f39c12',
                           foreground='white',
                           borderwidth=0,
                           relief='flat',
                           padding=(20, 10))
        self.style.map('Warning.TButton',
                      background=[('active', '#e67e22'), ('pressed', '#d35400')])
        
        self.style.configure('Danger.TButton', 
                           font=('Segoe UI', 10),
                           background='#e74c3c',
                           foreground='white',
                           borderwidth=0,
                           relief='flat',
                           padding=(20, 10))
        self.style.map('Danger.TButton',
                      background=[('active', '#c0392b'), ('pressed', '#a93226')])
        
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
        
        # LabelFrame styles - simple configuration without custom layout
        self.style.configure('TLabelframe', background='#ffffff', borderwidth=1)
        self.style.configure('TLabelframe.Label', 
                           font=('Segoe UI', 11, 'bold'), 
                           foreground='#2c3e50',
                           background='#f8f9fa')
    
    def setup_ui(self):
        """Setup the modern user interface with minimalist design"""
        # Main container with padding
        main_container = tk.Frame(self.window, bg='#f8f9fa')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title header with gradient-like effect
        title_frame = tk.Frame(main_container, bg='#f8f9fa')
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, text="🎭 Face Recognition System", 
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Status indicator with smooth animation
        self.status_indicator = tk.Label(title_frame, text="●", font=('Segoe UI', 20), 
                                        fg='#e94560', bg='#f8f9fa')
        self.status_indicator.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Main content area
        content_frame = tk.Frame(main_container, bg='#f8f9fa')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (video + controls)
        left_panel = tk.Frame(content_frame, bg='#f8f9fa')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        # Video frame with modern flat styling
        video_frame = ttk.LabelFrame(left_panel, text="📹 Live Camera Feed")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        video_frame.configure(relief='flat', borderwidth=1)
        
        # Video container with subtle shadow effect
        video_container = tk.Frame(video_frame, bg='#1a1a2e', relief='flat', borderwidth=0,
                                  highlightthickness=2, highlightbackground='#e0e0e0')
        video_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.video_label = tk.Label(video_container, bg='#16213e', 
                                   text="Camera Feed\nClick 'Start Camera' to begin", 
                                   fg='#ffffff', font=('Segoe UI', 14), justify=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Store video display dimensions
        self.video_width = 640
        self.video_height = 480
        
        # Controls frame with card styling
        controls_frame = ttk.LabelFrame(left_panel, text="🎮 Camera Controls")
        controls_frame.pack(fill=tk.X, pady=(0, 15))
        controls_frame.configure(relief='flat', borderwidth=1)
        
        control_buttons_frame = tk.Frame(controls_frame, bg='white')
        control_buttons_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.start_button = ttk.Button(control_buttons_frame, text="▶ Start Camera", 
                                      command=self.start_camera, style='Primary.TButton')
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))
        
        self.stop_button = ttk.Button(control_buttons_frame, text="⏹ Stop Camera", 
                                     command=self.stop_camera, state=tk.DISABLED, style='Warning.TButton')
        self.stop_button.pack(side=tk.LEFT, padx=(0, 8))
        
        self.register_button = ttk.Button(control_buttons_frame, text="👤 Register New Face", 
                                         command=self.start_registration, state=tk.DISABLED, style='Success.TButton')
        self.register_button.pack(side=tk.LEFT)
        
        # Status frame with modern card design
        status_frame = ttk.LabelFrame(left_panel, text="📊 System Status")
        status_frame.pack(fill=tk.X)
        status_frame.configure(relief='flat', borderwidth=1)
        
        status_container = tk.Frame(status_frame, bg='white')
        status_container.pack(fill=tk.X, padx=15, pady=15)
        
        self.status_text = tk.StringVar(value="🔴 System ready. Click 'Start Camera' to begin recognition.")
        self.status_label = ttk.Label(status_container, textvariable=self.status_text, 
                                     style='Status.TLabel', background='white')
        self.status_label.pack(anchor=tk.W)
        
        # Right panel (employee management + detection info)
        right_panel = tk.Frame(content_frame, bg='#f8f9fa', width=360)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # Detection results with modern card
        detection_frame = ttk.LabelFrame(right_panel, text="🎯 Current Detection")
        detection_frame.pack(fill=tk.X, pady=(0, 15))
        detection_frame.configure(relief='flat', borderwidth=1)
        
        detection_container = tk.Frame(detection_frame, bg='white')
        detection_container.pack(fill=tk.X, padx=15, pady=15)
        
        # Identity display with accent
        identity_frame = tk.Frame(detection_container, bg='#4a90e2', relief='flat', borderwidth=0,
                                 highlightthickness=0)
        identity_frame.pack(fill=tk.X, pady=(0, 12))
        
        self.identity_label = tk.Label(identity_frame, text="👤 No face detected", 
                                      font=('Segoe UI', 13, 'bold'), fg='white', bg='#4a90e2',
                                      padx=15, pady=12)
        self.identity_label.pack(fill=tk.X)
        
        # Detection details with modern styling
        details_frame = tk.Frame(detection_container, bg='white')
        details_frame.pack(fill=tk.X)
        
        # Emotion
        emotion_frame = tk.Frame(details_frame, bg='white')
        emotion_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Label(emotion_frame, text="😊 Emotion:", font=('Segoe UI', 10, 'bold'), 
                bg='white', fg='#6c757d').pack(side=tk.LEFT)
        self.emotion_var = tk.StringVar(value="Neutral")
        tk.Label(emotion_frame, textvariable=self.emotion_var, font=('Segoe UI', 10), 
                bg='white', fg='#2ecc71').pack(side=tk.LEFT, padx=(10, 0))
        
        # Liveness
        liveness_frame = tk.Frame(details_frame, bg='white')
        liveness_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Label(liveness_frame, text="🔍 Liveness:", font=('Segoe UI', 10, 'bold'), 
                bg='white', fg='#6c757d').pack(side=tk.LEFT)
        self.liveness_var = tk.StringVar(value="Unknown")
        self.liveness_label = tk.Label(liveness_frame, textvariable=self.liveness_var, 
                                      font=('Segoe UI', 10), bg='white', fg='#4a90e2')
        self.liveness_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Distance
        distance_frame = tk.Frame(details_frame, bg='white')
        distance_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Label(distance_frame, text="📏 Distance:", font=('Segoe UI', 10, 'bold'), 
                bg='white', fg='#6c757d').pack(side=tk.LEFT)
        self.distance_var = tk.StringVar(value="N/A")
        tk.Label(distance_frame, textvariable=self.distance_var, font=('Segoe UI', 10), 
                bg='white', fg='#9b59b6').pack(side=tk.LEFT, padx=(10, 0))
        
        # Phase 7: Confidence
        confidence_frame = tk.Frame(details_frame, bg='white')
        confidence_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Label(confidence_frame, text="💯 Confidence:", font=('Segoe UI', 10, 'bold'), 
                bg='white', fg='#6c757d').pack(side=tk.LEFT)
        self.confidence_var = tk.StringVar(value="N/A")
        self.confidence_label = tk.Label(confidence_frame, textvariable=self.confidence_var, 
                                        font=('Segoe UI', 10, 'bold'), bg='white', fg='#2ecc71')
        self.confidence_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Phase 7: Debug panel
        debug_frame = ttk.LabelFrame(right_panel, text="🔍 Verification Debug")
        debug_frame.pack(fill=tk.X, pady=(0, 15))
        debug_frame.configure(relief='flat', borderwidth=1)
        
        debug_container = tk.Frame(debug_frame, bg='white')
        debug_container.pack(fill=tk.X, padx=15, pady=15)
        
        # Threshold info
        threshold_frame = tk.Frame(debug_container, bg='white')
        threshold_frame.pack(fill=tk.X, pady=(0, 5))
        tk.Label(threshold_frame, text="Threshold:", font=('Segoe UI', 9), 
                bg='white', fg='#6c757d').pack(side=tk.LEFT)
        self.threshold_display = tk.Label(threshold_frame, text=f"{OPTIMAL_THRESHOLD_GUI:.3f}", 
                                         font=('Segoe UI', 9, 'bold'), bg='white', fg='#4a90e2')
        self.threshold_display.pack(side=tk.LEFT, padx=(5, 0))
        
        # Matched pose
        pose_frame = tk.Frame(debug_container, bg='white')
        pose_frame.pack(fill=tk.X, pady=(0, 5))
        tk.Label(pose_frame, text="Matched Pose:", font=('Segoe UI', 9), 
                bg='white', fg='#6c757d').pack(side=tk.LEFT)
        self.pose_var = tk.StringVar(value="N/A")
        tk.Label(pose_frame, textvariable=self.pose_var, font=('Segoe UI', 9), 
                bg='white', fg='#2ecc71').pack(side=tk.LEFT, padx=(5, 0))
        
        # Confidence bar
        tk.Label(debug_container, text="Confidence:", font=('Segoe UI', 9), 
                bg='white', fg='#6c757d', anchor=tk.W).pack(fill=tk.X, pady=(5, 2))
        
        bar_frame = tk.Frame(debug_container, bg='#e0e0e0', height=20, relief='flat')
        bar_frame.pack(fill=tk.X)
        bar_frame.pack_propagate(False)
        
        self.confidence_bar = tk.Frame(bar_frame, bg='#2ecc71', height=20)
        self.confidence_bar.place(relwidth=0.0, relheight=1.0)
        
        self.confidence_bar_text = tk.Label(bar_frame, text="0%", font=('Segoe UI', 8, 'bold'), 
                                           bg='#e0e0e0', fg='#555')
        self.confidence_bar_text.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Employee management with card styling
        employee_frame = ttk.LabelFrame(right_panel, text="👥 Employee Management")
        employee_frame.pack(fill=tk.X, pady=(0, 15))
        employee_frame.configure(relief='flat', borderwidth=1)
        
        emp_container = tk.Frame(employee_frame, bg='white')
        emp_container.pack(fill=tk.X, padx=15, pady=15)
        
        ttk.Button(emp_container, text="👁 View Employees", command=self.view_employees).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(emp_container, text="📅 View Attendance", command=self.view_attendance).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(emp_container, text="✏ Edit Employee", command=self.edit_employee).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(emp_container, text="🗑 Delete Employee", command=self.delete_employee, style='Danger.TButton').pack(fill=tk.X, pady=(0, 8))
        ttk.Button(emp_container, text="⚙ Adjust Threshold", command=self.adjust_threshold).pack(fill=tk.X)
        
        # Statistics with modern card
        stats_frame = ttk.LabelFrame(right_panel, text="📈 Session Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        stats_frame.configure(relief='flat', borderwidth=1)
        
        stats_container = tk.Frame(stats_frame, bg='white')
        stats_container.pack(fill=tk.X, padx=15, pady=15)
        
        self.stats_text = tk.StringVar()
        self.update_stats_display()
        ttk.Label(stats_container, textvariable=self.stats_text, style='Info.TLabel', 
                 background='white', justify=tk.LEFT).pack(anchor=tk.W)
    
        
        # Configure grid weights for responsive design
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
    
    def update_stats_display(self):
        """Update session statistics display"""
        total = self.recognition_stats['total_detections']
        success = self.recognition_stats['successful_recognitions']
        unique = len(self.recognition_stats['unique_faces_today'])
        accuracy = (success / total * 100) if total > 0 else 0
        
        stats_text = f"""Total Detections: {total}
Successful Recognition: {success}
Unique Faces Today: {unique}
Recognition Accuracy: {accuracy:.1f}%
Employees in DB: {len(employee_db)}"""
        self.stats_text.set(stats_text)
    
    def update_debug_panel(self):
        """Update Phase 7 debug panel with verification details"""
        threshold = _load_gui_threshold(OPTIMAL_THRESHOLD_GUI)
        self.threshold_display.config(text=f"{threshold:.3f}")
        
        # Update matched pose
        if self.matched_pose_index >= 0:
            pose_names = ["Center", "Left", "Right", "Up", "Down"]
            pose_name = pose_names[self.matched_pose_index] if self.matched_pose_index < 5 else f"Pose {self.matched_pose_index + 1}"
            self.pose_var.set(pose_name)
        else:
            self.pose_var.set("N/A")
        
        # Update confidence bar
        if self.last_confidence > 0:
            confidence_ratio = self.last_confidence / 100.0
            self.confidence_bar.place(relwidth=confidence_ratio, relheight=1.0)
            self.confidence_bar_text.config(text=f"{self.last_confidence:.0f}%")
            
            # Color-code confidence bar
            if self.last_confidence >= 80:
                self.confidence_bar.config(bg='#2ecc71')  # Green
            elif self.last_confidence >= 60:
                self.confidence_bar.config(bg='#f39c12')  # Yellow
            else:
                self.confidence_bar.config(bg='#e74c3c')  # Red
        else:
            self.confidence_bar.place(relwidth=0.0, relheight=1.0)
            self.confidence_bar_text.config(text="0%")
    
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
        preferred_indices = [CAMERA_INDEX, 0, 1, 2]
        tried = []
        for idx in preferred_indices:
            if idx in tried:
                continue
            cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                print(f"Using camera index: {idx}")
                return cap
            tried.append(idx)
            if cap:
                cap.release()
        return None

    def start_camera(self):
        """Start camera and begin processing"""
        self.cap = self._open_camera_with_fallback()
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Camera Error", "❌ Could not open webcam. Please check camera connection.")
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
        
        # Update UI indicators
        self.status_text.set("🟢 Camera started. Face recognition active...")
        self.status_indicator.config(fg='#2ecc71')  # Modern green
        
        self.video_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.video_thread.start()
        self.update_display()
    
    def stop_camera(self):
        """Stop camera"""
        self.running = False
        if self.cap:
            self.cap.release()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.register_button.config(state=tk.DISABLED)
        
        # Update UI indicators
        self.status_text.set("🔴 Camera stopped. Click 'Start Camera' to resume.")
        self.status_indicator.config(fg='#e94560')  # Modern red
        self.video_label.config(image='', text="Camera Feed\nStopped", 
                               fg='white', font=('Segoe UI', 14), justify=tk.CENTER)
        
        # Reset detection displays
        self.identity_label.config(text="👤 No face detected", bg='white', fg='#2c3e50')
        self.emotion_var.set("Neutral")
        self.liveness_var.set("Unknown")
        self.distance_var.set("N/A")
        self.liveness_label.config(fg='#3498db')
        
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
                
                print(f"✓ Registered '{self.registration_name}' with {len(self.registration_state['embeddings'])} poses")
                self.status_text.set(f"✓ Registered '{self.registration_name}' successfully!")
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
        """Capture and process video frames (runs in background thread)"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            self.frame_count += 1
            
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
            
            # Select primary face if multiple faces  
            primary_face_idx = select_primary_face(faces, w, h)

            for face_idx, (x, y, w, h) in enumerate(faces):
                is_primary = (face_idx == primary_face_idx)
                box_color = (128, 128, 128) if not is_primary else (0, 255, 0)

                # Skip verification during registration mode - always show as unregistered
                if self.registration_mode:
                    if is_primary:
                        self.last_identity = "Registering..."
                        box_color = (255, 165, 0)  # Orange for registration
                # Only process primary face for verification when NOT in registration mode
                elif is_primary and self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0:
                    cropped_face = crop_face_with_padding(frame, x, y, w, h)

                    if cropped_face.size > 0 and cropped_face.shape[0] >= 50:
                        cropped_face_resized = cv2.resize(cropped_face, (IMG_SIZE, IMG_SIZE))

                        # DeepFace-based emotion + liveness (run less frequently to avoid lag)
                        is_live = True  # Default to Real
                        if self.frame_count % self.EMOTION_EVERY_N_FRAMES == 0:
                            try:
                                rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                                emo, is_live = analyze_emotion_and_liveness(rgb_face)
                                self.last_emotion = emo
                                self.last_liveness = 'Real' if is_live else 'Spoof'
                            except Exception as e:
                                print(f"DeepFace analyze error: {e}")
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

                                if min_distance < OPTIMAL_THRESHOLD_GUI:
                                    self.last_identity = best_match
                                    box_color = (0, 255, 0)
                                    # Trigger success animation
                                    if not self.verification_animation['active']:
                                        self.verification_animation = {
                                            'active': True,
                                            'type': 'success',
                                            'frame_count': 0,
                                            'max_frames': 30
                                        }
                                else:
                                    self.last_identity = "Not Registered"
                                    box_color = (0, 0, 255)
                                    self.last_confidence = 0.0  # Phase 7: Reset confidence
                                    self.matched_pose_index = -1  # Phase 7: No match
                                    # Trigger failure animation  
                                    if not self.verification_animation['active']:
                                        self.verification_animation = {
                                            'active': True,
                                            'type': 'failure',
                                            'frame_count': 0,
                                            'max_frames': 30
                                        }

                                self.last_distance = min_distance
                                
                                # Phase 7: Calculate confidence and track matched pose
                                threshold = _load_gui_threshold(OPTIMAL_THRESHOLD_GUI)
                                self.last_confidence = max(0, min(100, (1 - min_distance / threshold) * 100))
                                
                                # Find which pose matched (for multi-embedding)
                                if USE_MULTI_EMBEDDING and isinstance(saved_data, list):
                                    distances_with_idx = [(F.pairwise_distance(trial_embedding, emb).item(), idx) 
                                                         for idx, emb in enumerate(saved_data)]
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

            if not self.frame_queue.full():
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

    def update_display(self):
        """Update display on main thread"""
        if not self.running:
            return

        try:
            frame = self.frame_queue.get_nowait()

            # Resize frame to fit display window
            frame_resized = cv2.resize(frame, (self.video_width, self.video_height))

            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk, text="")

            # Update modern UI elements
            self.update_detection_display()
            self.update_stats_display()
            self.update_debug_display()
            
        except queue.Empty:
            pass

        self.window.after(33, self.update_display)
    
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
                status_text = "🟡 Face detected but not recognized"
            else:
                self.identity_label.config(text="👤 No face detected", 
                                         bg='white', fg='#2c3e50')
                status_text = "⭕ No face in camera view"
        
        # Update individual components
        self.emotion_var.set(self.last_emotion)
        self.liveness_var.set(self.last_liveness)
        
        # Color-code liveness
        if self.last_liveness == "Real":
            self.liveness_label.config(fg='#27ae60')
        elif self.last_liveness == "Spoof":
            self.liveness_label.config(fg='#e74c3c')
        else:
            self.liveness_label.config(fg='#3498db')
        
        # Update distance
        if self.last_distance != float('inf'):
            self.distance_var.set(f"{self.last_distance:.3f}")
        else:
            self.distance_var.set("N/A")
        
        # Phase 7: Update confidence display
        if self.last_confidence > 0:
            self.confidence_var.set(f"{self.last_confidence:.1f}%")
            
            # Color-code confidence
            if self.last_confidence >= 80:
                self.confidence_label.config(fg='#2ecc71')  # Green
            elif self.last_confidence >= 60:
                self.confidence_label.config(fg='#f39c12')  # Yellow
            else:
                self.confidence_label.config(fg='#e74c3c')  # Red
        else:
            self.confidence_var.set("N/A")
            self.confidence_label.config(fg='#6c757d')
        
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
