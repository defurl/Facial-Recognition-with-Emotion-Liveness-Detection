"""
Registration handler: Encapsulates the registration workflow and state management.

Separates registration concerns from the main GUI orchestration.
"""
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageTk
import time
import json

from src.config import OUTPUT_DIR


class RegistrationHandler:
    """Manages employee registration workflow and state."""
    
    def __init__(self):
        self.registration_mode = False
        self.registration_name = ""
        self.registration_state: Optional[Dict] = None
        self.registration_feedback = ""
        self.registration_feedback_color = (255, 165, 0)  # Default orange
        
        # Registration protocol settings
        self.TARGET_POSES = [
            "facing_forward",
            "looking_left",
            "looking_right"
        ]
        self.POSES_PER_INSTRUCTION = [
            "Face Forward",
            "Look Left",
            "Look Right"
        ]
        self.MIN_POSES = 3
        self.EMBEDDINGS_PER_POSE = 5
    
    def start_registration(self, name: str) -> bool:
        """
        Start registration workflow.
        
        Args:
            name: Employee name to register
            
        Returns:
            True if registration started successfully
        """
        if not name or len(name.strip()) == 0:
            return False
        
        self.registration_mode = True
        self.registration_name = name.strip()
        self.registration_state = {
            'frames': [],
            'embeddings': [],
            'instructions': [],
            'poses': [],
            'start_time': time.time(),
            'current_pose_index': 0,
            'poses_completed': 0,
        }
        self.registration_feedback = "Starting registration. Position yourself 2-3 feet from camera."
        self.registration_feedback_color = (0, 255, 0)  # Green
        
        return True
    
    def add_registration_frame(
        self,
        frame: np.ndarray,
        embedding: Optional[np.ndarray] = None,
        pose_instruction: str = ""
    ) -> bool:
        """
        Add frame and embedding to registration.
        
        Args:
            frame: The frame to add
            embedding: Face embedding (optional)
            pose_instruction: Current pose instruction text
            
        Returns:
            True if added successfully
        """
        if not self.registration_mode or self.registration_state is None:
            return False
        
        try:
            self.registration_state['frames'].append(frame.copy())
            if embedding is not None:
                self.registration_state['embeddings'].append(embedding)
            self.registration_state['instructions'].append(pose_instruction)
            
            return True
        except Exception as e:
            print(f"[REGISTRATION] Error adding frame: {e}")
            return False
    
    def complete_registration(self) -> Dict:
        """
        Complete registration and return captured data.
        
        Returns:
            Dict with frames, embeddings, instructions
        """
        if not self.registration_mode or self.registration_state is None:
            return {}
        
        result = {
            'name': self.registration_name,
            'frames': self.registration_state.get('frames', []),
            'embeddings': self.registration_state.get('embeddings', []),
            'instructions': self.registration_state.get('instructions', []),
            'timestamp': time.time(),
            'duration': time.time() - self.registration_state.get('start_time', time.time()),
            'num_embeddings': len(self.registration_state.get('embeddings', [])),
        }
        
        self.cancel_registration()
        return result
    
    def cancel_registration(self):
        """Cancel ongoing registration."""
        self.registration_mode = False
        self.registration_name = ""
        self.registration_state = None
        self.registration_feedback = ""
    
    def update_registration_feedback(self, message: str, color: Tuple[int, int, int] = (0, 255, 0)):
        """
        Update registration feedback message.
        
        Args:
            message: Feedback message
            color: BGR color tuple
        """
        self.registration_feedback = message
        self.registration_feedback_color = color
    
    def get_current_pose_instruction(self) -> str:
        """Get current pose instruction."""
        if self.registration_state is None:
            return ""
        
        pose_idx = self.registration_state.get('current_pose_index', 0)
        if pose_idx < len(self.POSES_PER_INSTRUCTION):
            return self.POSES_PER_INSTRUCTION[pose_idx]
        
        return "Complete"
    
    def is_registration_active(self) -> bool:
        """Check if registration is active."""
        return self.registration_mode and self.registration_state is not None
    
    def get_registration_progress(self) -> Dict:
        """Get registration progress information."""
        if not self.registration_mode or self.registration_state is None:
            return {}
        
        state = self.registration_state
        embeddings_collected = len(state.get('embeddings', []))
        frames_collected = len(state.get('frames', []))
        poses_completed = state.get('poses_completed', 0)
        
        return {
            'frames': frames_collected,
            'embeddings': embeddings_collected,
            'poses_completed': poses_completed,
            'current_pose': self.get_current_pose_instruction(),
            'feedback': self.registration_feedback,
            'color': self.registration_feedback_color,
        }
    
    def save_registration_to_database(self, employee_db: Dict, embeddings_path: Path) -> bool:
        """
        Save registration to employee database.
        
        Args:
            employee_db: Employee database dict
            embeddings_path: Path to save embeddings
            
        Returns:
            True if saved successfully
        """
        if not self.registration_state or not self.registration_name:
            return False
        
        try:
            embeddings = self.registration_state.get('embeddings', [])
            if not embeddings:
                print("[REGISTRATION] No embeddings to save")
                return False
            
            # Average embeddings
            mean_embedding = np.mean(embeddings, axis=0)
            
            # Save to database
            employee_db[self.registration_name] = {
                'embedding': mean_embedding,
                'num_embeddings': len(embeddings),
                'registered_at': time.time(),
            }
            
            # Save embeddings file
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                embeddings_path,
                **{name: data['embedding'] for name, data in employee_db.items()}
            )
            
            print(f"[REGISTRATION] Saved {self.registration_name} with {len(embeddings)} embeddings")
            return True
            
        except Exception as e:
            print(f"[REGISTRATION] Error saving to database: {e}")
            return False


def create_registration_dialog(parent: tk.Widget, on_start_callback) -> Optional[str]:
    """
    Create and show registration name dialog.
    
    Args:
        parent: Parent widget
        on_start_callback: Callback when registration starts
        
    Returns:
        Employee name or None if cancelled
    """
    dialog = tk.Toplevel(parent)
    dialog.title("👤 Register New Employee")
    dialog.geometry("400x400")
    dialog.configure(bg='#f0f0f0')
    dialog.transient(parent)
    dialog.grab_set()
    
    # Center the dialog
    dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
    
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

    inst_text = """• Position yourself 2-3 feet from camera (closer = better quality)
• Look directly at the camera
• Make sure you have good lighting
• Keep your face centered and still
• Follow the on-screen pose instructions
• Hold each pose steady when prompted
• Registration captures 3 optimized poses"""

    tk.Label(instructions, text=inst_text, font=('Arial', 9), 
            fg='#2d5a2d', bg='#e8f5e8', justify=tk.LEFT, wraplength=350).pack(anchor=tk.W, padx=10, pady=(0, 10))
    
    result = [None]
    
    def start_process():
        name = name_var.get().strip()
        if not name:
            messagebox.showwarning("Invalid Name", "Please enter an employee name")
            return
        result[0] = name
        on_start_callback(name)
        dialog.destroy()
    
    # Buttons
    button_frame = tk.Frame(content, bg='#f0f0f0')
    button_frame.pack(fill=tk.X)
    
    from tkinter import ttk
    ttk.Button(button_frame, text="✓ Start Registration", command=start_process).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Button(button_frame, text="✗ Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    # Bind Enter key
    dialog.bind('<Return>', lambda e: start_process())
    
    dialog.wait_window()
    return result[0]
