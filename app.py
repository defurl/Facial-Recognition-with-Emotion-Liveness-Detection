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


class AttendanceSystemGUI:
    """Modern GUI application for face recognition attendance system"""
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🎭 Face Recognition Attendance System")
        self.window.geometry("1200x800")
        self.window.minsize(1000, 700)
        self.window.configure(bg='#f0f0f0')
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure styles
        self.setup_styles()
        
        # Video capture
        self.cap = None
        self.running = False
        
        # Processing control
        self.frame_count = 0
        self.PROCESS_EVERY_N_FRAMES = PROCESS_EVERY_N_FRAMES
        self.last_emotion = "Neutral"
        self.last_liveness = "Unknown"
        self.last_identity = "Not Registered"
        self.last_distance = float('inf')
        self.multiple_faces_warning = False
        
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
        
        self.setup_ui()
        
    def setup_styles(self):
        """Setup modern TTK styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure custom styles
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        self.style.configure('Status.TLabel', font=('Arial', 11), foreground='#27ae60')
        self.style.configure('Error.TLabel', font=('Arial', 11), foreground='#e74c3c')
        self.style.configure('Info.TLabel', font=('Arial', 10), foreground='#7f8c8d')
        
        # Button styles
        self.style.configure('Primary.TButton', font=('Arial', 10, 'bold'))
        self.style.configure('Success.TButton', font=('Arial', 10), foreground='#27ae60')
        self.style.configure('Warning.TButton', font=('Arial', 10), foreground='#f39c12')
        self.style.configure('Danger.TButton', font=('Arial', 10), foreground='#e74c3c')
        
        # Frame styles - use default TLabelFrame layout
        self.style.configure('Card.TLabelFrame', 
                           relief='solid', 
                           borderwidth=1,
                           background='#ffffff')
        self.style.configure('Card.TLabelFrame.Label', 
                           font=('Arial', 11, 'bold'), 
                           foreground='#2c3e50')
    
    def setup_ui(self):
        """Setup the modern user interface"""
        # Main container with padding
        main_container = tk.Frame(self.window, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title header
        title_frame = tk.Frame(main_container, bg='#f0f0f0')
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(title_frame, text="🎭 Face Recognition Attendance System", 
                 style='Title.TLabel').pack(side=tk.LEFT)
        
        # Status indicator
        self.status_indicator = tk.Label(title_frame, text="●", font=('Arial', 20), 
                                        fg='#e74c3c', bg='#f0f0f0')
        self.status_indicator.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Main content area
        content_frame = tk.Frame(main_container, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (video + controls)
        left_panel = tk.Frame(content_frame, bg='#f0f0f0')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video frame with modern styling
        video_frame = ttk.LabelFrame(left_panel, text="📹 Live Camera Feed")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Video container
        video_container = tk.Frame(video_frame, bg='#2c3e50', relief='sunken', borderwidth=2)
        video_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.video_label = tk.Label(video_container, bg='#34495e', text="Camera Feed\nClick 'Start Camera' to begin", 
                                   fg='white', font=('Arial', 14), justify=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Store video display dimensions
        self.video_width = 640
        self.video_height = 480
        
        # Controls frame
        controls_frame = ttk.LabelFrame(left_panel, text="🎮 Camera Controls")
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        control_buttons_frame = tk.Frame(controls_frame)
        control_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = ttk.Button(control_buttons_frame, text="▶ Start Camera", 
                                      command=self.start_camera, style='Primary.TButton')
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(control_buttons_frame, text="⏹ Stop Camera", 
                                     command=self.stop_camera, state=tk.DISABLED, style='Warning.TButton')
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.register_button = ttk.Button(control_buttons_frame, text="👤 Register Employee", 
                                         command=self.start_registration, state=tk.DISABLED, style='Success.TButton')
        self.register_button.pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(left_panel, text="📊 System Status")
        status_frame.pack(fill=tk.X)
        
        status_container = tk.Frame(status_frame)
        status_container.pack(fill=tk.X, padx=10, pady=10)
        
        self.status_text = tk.StringVar(value="🔴 System ready. Click 'Start Camera' to begin recognition.")
        self.status_label = ttk.Label(status_container, textvariable=self.status_text, style='Status.TLabel')
        self.status_label.pack(anchor=tk.W)
        
        # Right panel (employee management + detection info)
        right_panel = tk.Frame(content_frame, bg='#f0f0f0', width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # Detection results
        detection_frame = ttk.LabelFrame(right_panel, text="🎯 Current Detection")
        detection_frame.pack(fill=tk.X, pady=(0, 10))
        
        detection_container = tk.Frame(detection_frame)
        detection_container.pack(fill=tk.X, padx=15, pady=15)
        
        # Identity display
        identity_frame = tk.Frame(detection_container, bg='white', relief='raised', borderwidth=1)
        identity_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.identity_label = tk.Label(identity_frame, text="👤 No face detected", 
                                      font=('Arial', 14, 'bold'), fg='#2c3e50', bg='white',
                                      padx=15, pady=10)
        self.identity_label.pack(fill=tk.X)
        
        # Detection details
        details_frame = tk.Frame(detection_container)
        details_frame.pack(fill=tk.X)
        
        # Emotion
        emotion_frame = tk.Frame(details_frame)
        emotion_frame.pack(fill=tk.X, pady=(0, 5))
        tk.Label(emotion_frame, text="😊 Emotion:", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0').pack(side=tk.LEFT)
        self.emotion_var = tk.StringVar(value="Neutral")
        tk.Label(emotion_frame, textvariable=self.emotion_var, font=('Arial', 10), 
                bg='#f0f0f0', fg='#27ae60').pack(side=tk.LEFT, padx=(10, 0))
        
        # Liveness
        liveness_frame = tk.Frame(details_frame)
        liveness_frame.pack(fill=tk.X, pady=(0, 5))
        tk.Label(liveness_frame, text="🔍 Liveness:", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0').pack(side=tk.LEFT)
        self.liveness_var = tk.StringVar(value="Unknown")
        self.liveness_label = tk.Label(liveness_frame, textvariable=self.liveness_var, 
                                      font=('Arial', 10), bg='#f0f0f0', fg='#3498db')
        self.liveness_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Distance
        distance_frame = tk.Frame(details_frame)
        distance_frame.pack(fill=tk.X, pady=(0, 5))
        tk.Label(distance_frame, text="📏 Distance:", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0').pack(side=tk.LEFT)
        self.distance_var = tk.StringVar(value="N/A")
        tk.Label(distance_frame, textvariable=self.distance_var, font=('Arial', 10), 
                bg='#f0f0f0', fg='#9b59b6').pack(side=tk.LEFT, padx=(10, 0))
        
        # Employee management
        employee_frame = ttk.LabelFrame(right_panel, text="👥 Employee Management")
        employee_frame.pack(fill=tk.X, pady=(0, 10))
        
        emp_container = tk.Frame(employee_frame)
        emp_container.pack(fill=tk.X, padx=15, pady=15)
        
        ttk.Button(emp_container, text="👁 View Employees", command=self.view_employees).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(emp_container, text="✏ Edit Employee", command=self.edit_employee).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(emp_container, text="🗑 Delete Employee", command=self.delete_employee, style='Danger.TButton').pack(fill=tk.X, pady=(0, 5))
        ttk.Button(emp_container, text="⚙ Adjust Threshold", command=self.adjust_threshold).pack(fill=tk.X, pady=(0, 5))
        
        # Statistics
        stats_frame = ttk.LabelFrame(right_panel, text="📈 Session Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        stats_container = tk.Frame(stats_frame)
        stats_container.pack(fill=tk.X, padx=15, pady=15)
        
        self.stats_text = tk.StringVar()
        self.update_stats_display()
        ttk.Label(stats_container, textvariable=self.stats_text, style='Info.TLabel', justify=tk.LEFT).pack(anchor=tk.W)
        
        # Debug info
        debug_frame = ttk.LabelFrame(right_panel, text="🔧 Debug Information")
        debug_frame.pack(fill=tk.X)
        
        debug_container = tk.Frame(debug_frame)
        debug_container.pack(fill=tk.X, padx=15, pady=15)
        
        self.debug_text = tk.StringVar()
        self.update_debug_display()
        debug_label = ttk.Label(debug_container, textvariable=self.debug_text, 
                               font=('Courier', 9), style='Info.TLabel', justify=tk.LEFT)
        debug_label.pack(anchor=tk.W)
        
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
    
    def update_debug_display(self):
        """Update debug information display"""
        threshold = _load_gui_threshold(OPTIMAL_THRESHOLD_GUI)
        debug_text = f"""Recognition Threshold: {threshold:.3f}
Process Every N Frames: {self.PROCESS_EVERY_N_FRAMES}
Model: {"Loaded" if verification_model else "Not Loaded"}
Camera Status: {"Active" if self.running else "Inactive"}
Registration Mode: {"ON" if self.registration_mode else "OFF"}"""
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
        self.status_indicator.config(fg='#27ae60')  # Green
        
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
        self.status_indicator.config(fg='#e74c3c')  # Red
        self.video_label.config(image='', text="Camera Feed\nStopped", 
                               fg='white', font=('Arial', 14), justify=tk.CENTER)
        
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
        dialog.geometry("400x250")
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
        dialog.geometry("600x500")
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
                
                # Display current step in top-right corner with background
                h, w = frame.shape[:2]
                instruction_text = f"Step {current_step + 1}/{len(state['poses_required'])}: {state['instructions'][current_step]}"
                name_text = f"Registering: {self.registration_name}"
                
                # Calculate text sizes
                (name_w, name_h), _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                (inst_w, inst_h), _ = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw semi-transparent background box (moved higher to avoid blocking face)
                overlay = frame.copy()
                box_w = max(name_w, inst_w) + 30
                box_h = name_h + inst_h + 30
                y_offset = 50  # Start further down to avoid blocking top of frame
                cv2.rectangle(overlay, (w - box_w - 10, y_offset), (w - 10, y_offset + box_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Draw text
                cv2.putText(frame, name_text, 
                           (w - box_w, y_offset + 20 + name_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, instruction_text, 
                           (w - box_w, y_offset + 20 + name_h + inst_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Check every 15 frames (~0.5 seconds at 30fps) to reduce flickering
                if self.frame_count - state['last_check_frame'] >= 15:
                    state['last_check_frame'] = self.frame_count
                    
                    faces = detect_faces(frame)
                    if len(faces) == 0:
                        cv2.putText(frame, "No face detected", (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        state['hold_frames'] = 0
                    else:
                        x, y, w, h = faces[0]
                        cropped_face = crop_face_with_padding(frame, x, y, w, h)
                        
                        if cropped_face.size == 0 or cropped_face.shape[0] < 50:
                            cv2.putText(frame, "Face too small", (10, 100), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            state['hold_frames'] = 0
                        else:
                            # Import quality check functions
                            from utils import check_image_blur, check_image_lighting
                            
                            # Quick quality checks
                            blur_var, blur_ok, _ = check_image_blur(cropped_face, threshold=80)
                            brightness, contrast, lighting_ok, _ = check_image_lighting(cropped_face, 25, 230, 30)
                            
                            if blur_ok and lighting_ok:
                                state['hold_frames'] += 1
                                remaining = 12 - state['hold_frames']
                                
                                if remaining > 0:
                                    cv2.putText(frame, f"✓ Hold steady... {remaining}", (10, 100), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                else:
                                    # Capture this pose
                                    cv2.putText(frame, "✓ Captured!", (10, 100), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
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
                                    cv2.putText(frame, "Image too blurry", (10, 100), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                elif not lighting_ok:
                                    cv2.putText(frame, "Poor lighting", (10, 100), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    # Show persistent hold counter between checks to reduce flickering
                    if state.get('hold_frames', 0) > 0:
                        remaining = max(0, 12 - state['hold_frames'])
                        cv2.putText(frame, f"✓ Hold steady... {remaining}", (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            faces = detect_faces(frame)
            h, w = frame.shape[:2]
            
            # Track if multiple faces detected
            self.multiple_faces_warning = len(faces) > 1
            
            # Select primary face if multiple faces  
            primary_face_idx = select_primary_face(faces, w, h)

            for face_idx, (x, y, w, h) in enumerate(faces):
                is_primary = (face_idx == primary_face_idx)
                box_color = (128, 128, 128) if not is_primary else (0, 255, 0)

                # Only process primary face  
                if is_primary and self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0 and not self.registration_mode:
                    cropped_face = crop_face_with_padding(frame, x, y, w, h)

                    if cropped_face.size > 0 and cropped_face.shape[0] >= 50:
                        cropped_face_resized = cv2.resize(cropped_face, (IMG_SIZE, IMG_SIZE))

                        # DeepFace-based emotion + liveness
                        try:
                            rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                            emo, is_live = analyze_emotion_and_liveness(rgb_face)
                            self.last_emotion = emo
                            self.last_liveness = 'Real' if is_live else 'Spoof'
                        except Exception as e:
                            print(f"DeepFace analyze error: {e}")
                            is_live = True
                            self.last_liveness = 'Real'

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
                                else:
                                    self.last_identity = "Not Registered"
                                    box_color = (0, 0, 255)

                                self.last_distance = min_distance
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
                    display_text = f"{self.last_identity} ({self.last_emotion} | {self.last_liveness})"
                else:
                    display_text = "Secondary Face"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                cv2.putText(frame, display_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                
                # Update statistics (primary face only)
                if is_primary:
                    self.recognition_stats['total_detections'] += 1
                    if self.last_identity not in ["Not Registered", "Error", "Spoof Detected", "Face too small", "Secondary Face"]:
                        self.recognition_stats['successful_recognitions'] += 1
                        self.recognition_stats['unique_faces_today'].add(self.last_identity)
            
            # Display warning banner if multiple faces detected
            if self.multiple_faces_warning and len(faces) > 1:
                warning_text = "⚠ Multiple faces detected - processing primary face only"
                (tw, th), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (0, 0), (tw + 20, th + 15), (0, 165, 255), -1)
                cv2.putText(frame, warning_text, (10, th + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
        
        # Update status
        self.status_text.set(status_text)
    
    def adjust_threshold(self):
        """Adjust verification threshold with enhanced dialog"""
        global OPTIMAL_THRESHOLD_GUI
        
        # Create custom dialog
        dialog = tk.Toplevel(self.window)
        dialog.title("⚙ Adjust Recognition Threshold")
        dialog.geometry("400x300")
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
                    messagebox.showinfo("Success", f"✅ Threshold updated to {OPTIMAL_THRESHOLD_GUI:.3f}")
                    dialog.destroy()
                else:
                    messagebox.showerror("Invalid Value", "⚠ Please enter a value between 0.1 and 3.0")
            except ValueError:
                messagebox.showerror("Invalid Input", "⚠ Please enter a valid number")
        
        ttk.Button(button_frame, text="✓ Apply", command=apply_threshold).pack(side=tk.LEFT, padx=(0, 10))
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
        """Edit employee with enhanced dialog"""
        if len(employee_db) == 0:
            messagebox.showinfo("✏ Edit Employee", "No employees registered yet.")
            return
        
        dialog = tk.Toplevel(self.window)
        dialog.title("✏ Edit Employee")
        dialog.geometry("400x450")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Header
        header = tk.Frame(dialog, bg='#f39c12', height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="✏ Edit Employee Name", font=('Arial', 14, 'bold'), 
                fg='white', bg='#f39c12').pack(pady=15)
        
        # Content
        content = tk.Frame(dialog, bg='#f0f0f0')
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(content, text="Select employee to edit:", font=('Arial', 11, 'bold'), 
                bg='#f0f0f0').pack(anchor=tk.W, pady=(0, 10))
        
        # Employee list
        list_frame = tk.Frame(content)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
                            font=('Arial', 10), height=12)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        for name in sorted(employee_db.keys()):
            listbox.insert(tk.END, name)
        
        def confirm_edit():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("⚠ No Selection", "Please select an employee to edit.")
                return
            
            old_name = listbox.get(selection[0])
            new_name = simpledialog.askstring("✏ Edit Employee", 
                                            f"Enter new name for '{old_name}':", 
                                            parent=dialog)
            
            if new_name and new_name.strip() and new_name != old_name:
                new_name = new_name.strip()
                if new_name in employee_db:
                    messagebox.showerror("❌ Error", f"Employee '{new_name}' already exists!")
                    return
                
                employee_db[new_name] = employee_db.pop(old_name)
                torch.save(employee_db, EMPLOYEE_DB_PATH)
                messagebox.showinfo("✅ Success", f"Renamed '{old_name}' to '{new_name}'")
                self.update_stats_display()
                self.update_debug_display()
                dialog.destroy()
        
        # Buttons
        button_frame = tk.Frame(content, bg='#f0f0f0')
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="✓ Edit Selected", command=confirm_edit).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="✗ Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    def delete_employee(self):
        """Delete employee with enhanced dialog"""
        if len(employee_db) == 0:
            messagebox.showinfo("🗑 Delete Employee", "No employees registered yet.")
            return
        
        dialog = tk.Toplevel(self.window)
        dialog.title("🗑 Delete Employee")
        dialog.geometry("400x450")
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
        
        tk.Label(content, text="Select employee to delete:", font=('Arial', 11, 'bold'), 
                fg='#e74c3c', bg='#f0f0f0').pack(anchor=tk.W, pady=(0, 10))
        
        # Employee list
        list_frame = tk.Frame(content)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
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
        
        delete_btn = ttk.Button(button_frame, text="🗑 Delete Selected", command=confirm_delete)
        delete_btn.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="✗ Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
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
