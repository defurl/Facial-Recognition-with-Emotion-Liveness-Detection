"""
Display layer: Encapsulates all UI widget updates and visual rendering.

Separates view/display logic from business logic and camera processing.
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Any, List


class DisplayLayer:
    """Manages all UI display updates without business logic."""
    
    def __init__(self, window: tk.Tk):
        self.window = window
        self.video_label: Optional[tk.Label] = None
        self.identity_label: Optional[tk.Label] = None
        self.status_text: Optional[tk.StringVar] = None
        self.emotion_var: Optional[tk.StringVar] = None
        self.liveness_var: Optional[tk.StringVar] = None
        self.distance_var: Optional[tk.StringVar] = None
        self.confidence_var: Optional[tk.StringVar] = None
        self.fps_var: Optional[tk.StringVar] = None
        self.latency_var: Optional[tk.StringVar] = None
        self.log_listbox: Optional[tk.Listbox] = None
        self.ear_canvas: Optional[tk.Canvas] = None
        self.ear_debug_text: Optional[tk.Text] = None
        self.ear_current_label: Optional[tk.Label] = None
        self.blink_count_label: Optional[tk.Label] = None
        self.liveness_label: Optional[tk.Label] = None
        self.detection_mode_label: Optional[tk.Label] = None
        self.start_button: Optional[ttk.Button] = None
        self.stop_button: Optional[ttk.Button] = None
        self.register_button: Optional[ttk.Button] = None
        self.reset_button: Optional[ttk.Button] = None
        self.clear_cache_button: Optional[ttk.Button] = None
    
    def update_video_frame(self, photo_image):
        """Update the video frame display."""
        if self.video_label is not None:
            self.video_label.config(image=photo_image)
            self.video_label.image = photo_image  # Keep reference
    
    def update_status(self, message: str):
        """Update status text."""
        if self.status_text is not None:
            self.status_text.set(message)
    
    def update_identity(self, name: str, bg_color: str, text_color: str):
        """Update identity display with color coding."""
        if self.identity_label is not None:
            self.identity_label.config(text=name, bg=bg_color, fg=text_color)
    
    def update_emotion(self, emotion: str):
        """Update emotion display."""
        if self.emotion_var is not None:
            self.emotion_var.set(emotion)
    
    def update_liveness(self, liveness: str, color: str = '#58a6ff'):
        """Update liveness display with optional color."""
        if self.liveness_var is not None:
            self.liveness_var.set(liveness)
        if self.liveness_label is not None:
            self.liveness_label.config(fg=color)
    
    def update_distance(self, distance: str):
        """Update distance display."""
        if self.distance_var is not None:
            self.distance_var.set(distance)
    
    def update_confidence(self, confidence: str):
        """Update confidence display."""
        if self.confidence_var is not None:
            self.confidence_var.set(confidence)
    
    def update_fps(self, fps: float):
        """Update FPS display."""
        if self.fps_var is not None:
            self.fps_var.set(f"FPS: {fps:.1f}")
    
    def update_latency(self, latency_ms: float):
        """Update latency display."""
        if self.latency_var is not None:
            self.latency_var.set(f"Latency: {latency_ms:.0f}ms")
    
    def update_ear_display(self, current_ear: float, blink_count: int, debug_text: str = ""):
        """Update EAR (Eye Aspect Ratio) display."""
        if self.ear_current_label is not None:
            self.ear_current_label.config(text=f"Current EAR: {current_ear:.3f}")
        if self.blink_count_label is not None:
            self.blink_count_label.config(text=f"Blinks: {blink_count}")
        if self.ear_debug_text is not None and debug_text:
            self.ear_debug_text.config(state=tk.NORMAL)
            self.ear_debug_text.delete(1.0, tk.END)
            self.ear_debug_text.insert(tk.END, debug_text)
            self.ear_debug_text.config(state=tk.DISABLED)
    
    def update_ear_graph(self, canvas_image):
        """Update EAR graph canvas."""
        if self.ear_canvas is not None:
            self.ear_canvas.create_image(0, 0, image=canvas_image, anchor=tk.NW)
            self.ear_canvas.image = canvas_image  # Keep reference
    
    def add_log_entry(self, entry: str):
        """Add entry to recent check-ins log."""
        if self.log_listbox is not None:
            self.log_listbox.insert(0, entry)
            # Keep only last 50 entries
            if self.log_listbox.size() > 50:
                self.log_listbox.delete(50)
    
    def clear_log(self):
        """Clear log entries."""
        if self.log_listbox is not None:
            self.log_listbox.delete(0, tk.END)
    
    def set_detection_mode(self, mode_text: str, color: str = 'green'):
        """Update detection mode label."""
        if self.detection_mode_label is not None:
            self.detection_mode_label.config(text=mode_text, fg=color)
    
    def enable_camera_controls(self):
        """Enable camera control buttons."""
        if self.start_button:
            self.start_button.config(state=tk.DISABLED)
        if self.stop_button:
            self.stop_button.config(state=tk.NORMAL)
        if self.register_button:
            self.register_button.config(state=tk.NORMAL)
        if self.reset_button:
            self.reset_button.config(state=tk.NORMAL)
        if self.clear_cache_button:
            self.clear_cache_button.config(state=tk.NORMAL)
    
    def disable_camera_controls(self):
        """Disable camera control buttons."""
        if self.start_button:
            self.start_button.config(state=tk.NORMAL)
        if self.stop_button:
            self.stop_button.config(state=tk.DISABLED)
        if self.register_button:
            self.register_button.config(state=tk.DISABLED)
        if self.reset_button:
            self.reset_button.config(state=tk.DISABLED)
        if self.clear_cache_button:
            self.clear_cache_button.config(state=tk.DISABLED)
    
    def reset_detections(self):
        """Reset all detection displays to default."""
        if self.identity_label is not None:
            self.identity_label.config(text="No face detected", bg='#1f6feb', fg='#ffffff')
        if self.emotion_var is not None:
            self.emotion_var.set("Neutral")
        if self.liveness_var is not None:
            self.liveness_var.set("Unknown")
        if self.distance_var is not None:
            self.distance_var.set("N/A")
        if self.confidence_var is not None:
            self.confidence_var.set("N/A")
        if self.fps_var is not None:
            self.fps_var.set("FPS: --")
        if self.latency_var is not None:
            self.latency_var.set("Latency: --")
        if self.liveness_label is not None:
            self.liveness_label.config(fg='#58a6ff')
        if self.video_label is not None:
            self.video_label.config(image='', text="Camera Feed\nStopped", 
                                   fg='#8b949e', font=('Arial', 14), justify=tk.CENTER)
    
    def show_camera_stopped(self):
        """Show camera stopped message."""
        if self.video_label is not None:
            self.video_label.config(text="Camera Feed\nStopped", fg='#8b949e')
        self.update_status("● Camera stopped. Click 'Start' to resume.")
    
    def show_camera_initializing(self):
        """Show camera initializing message."""
        if self.video_label is not None:
            self.video_label.config(text="Camera Feed\nInitializing...", fg='#8b949e')
        self.update_status("● Initializing camera...")
    
    def show_camera_active(self):
        """Show camera active message."""
        self.update_status("● Camera started. Face recognition active...")
    
    def update_detection_identity(self, identity: str, bg_color: str, fg_color: str, emoji: str = ""):
        """Update identity display with custom emoji and colors."""
        if self.identity_label is not None:
            display_text = f"{emoji} {identity}" if emoji else identity
            self.identity_label.config(text=display_text, bg=bg_color, fg=fg_color)
    
    def update_liveness_with_confidence(self, liveness: str, color: str, confidence_text: str):
        """Update liveness display with color and confidence text."""
        if self.liveness_var is not None:
            self.liveness_var.set(liveness)
        if self.liveness_label is not None:
            self.liveness_label.config(fg=color)
        if hasattr(self, 'liveness_confidence_var') and self.liveness_confidence_var is not None:
            self.liveness_confidence_var.set(confidence_text)
    
    def update_detection_status(self, status_message: str):
        """Update detection-specific status message."""
        if self.status_text is not None:
            self.status_text.set(status_message)

