#!/usr/bin/env python3
"""
Enhanced Stationary Registration Protocol

Based on analysis showing poor embedding separation (88.7% overlap),
this implements an optimized registration process for stationary camera setups.
Goal: Create distinct, high-quality embeddings that eliminate "mom bias" and improve accuracy.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import cv2
import torch
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from PIL import Image
import json

# Import project modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from config import *
from models import FaceEmbeddingCNN
from data_loader import get_transforms
from utils import detect_faces, crop_face_with_padding, estimate_head_pose_angles, validate_pose_for_target


class EnhancedRegistrationProtocol:
    """Enhanced registration protocol designed for stationary camera setup"""
    
    def __init__(self):
        self.model = None
        self.transform = None
        self.load_system()
        
        # OPTIMIZED: Fewer poses, shorter time, closer distance for better embeddings with pose validation
        self.registration_poses = [
            {
                'id': 'front_neutral',
                'instruction': 'Look straight at camera - neutral expression',
                'description': 'Keep shoulders square, eyes directly at camera',
                'target_pose': 'center',  # For pose validation
                'hold_time': 2,  # Reduced from 3s
                'weight': 1.5,   # Higher weight - most important
                'distance_instruction': 'Move CLOSER - face should fill more of the frame'
            },
            {
                'id': 'slight_left',
                'instruction': 'Turn head left 10-15 degrees',
                'description': 'Keep eyes on camera, slight left turn',
                'target_pose': 'left',  # For pose validation
                'hold_time': 1.5,  # Reduced from 3s
                'weight': 1.0
            },
            {
                'id': 'slight_right', 
                'instruction': 'Turn head right 10-15 degrees',
                'description': 'Keep eyes on camera, slight right turn',
                'target_pose': 'right',  # For pose validation
                'hold_time': 1.5,  # Reduced from 3s
                'weight': 1.0
            }
            # REMOVED: chin_up and chin_down poses to speed up process
            # 3 poses are sufficient for good embedding variation
        ]
        
        # OPTIMIZED: Quality thresholds for closer distance setup
        self.quality_thresholds = {
            'min_face_size': 150,           # Increased - expect larger faces when closer
            'optimal_face_size': 200,       # NEW - encourage even larger faces
            'blur_threshold': 120,          # Slightly relaxed - closer faces may have slight motion blur
            'brightness_min': 70,           # Tighter range for controlled environment
            'brightness_max': 170,          
            'contrast_min': 25,             # Slightly relaxed for closer faces
            'embedding_consistency': 0.40,  # Realistic threshold for real-world conditions
            'required_quality_score': 70,   # Slightly relaxed to balance with closer distance
            'distance_guidance': {
                'too_small_threshold': 120,  # If face < 120px, ask to move closer
                'optimal_threshold': 180,    # If face > 180px, perfect distance
                'too_close_threshold': 300   # If face > 300px, ask to step back slightly
            }
        }
    
    def load_system(self):
        """Load face recognition system"""
        print("Loading face recognition system...")
        
        # Load model
        self.model = FaceEmbeddingCNN(embedding_dim=256, num_classes=4000).to(DEVICE)
        if MODEL_METRIC_PATH.exists():
            self.model.load_state_dict(torch.load(MODEL_METRIC_PATH, map_location=DEVICE))
            self.model.eval()
            print("✓ Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_METRIC_PATH}")
        
        # Load transforms
        _, self.transform = get_transforms()
        print("✓ System ready")
    
    def assess_quality(self, face_crop, previous_embeddings=None):
        """Enhanced quality assessment with embedding consistency check"""
        if face_crop.size == 0:
            return {'score': 0.0, 'issues': ['No face detected'], 'details': {}}
        
        issues = []
        score = 100.0
        details = {}
        
        # ENHANCED: Face size check with distance guidance
        h, w = face_crop.shape[:2]
        face_size = min(h, w)
        details['face_size'] = face_size
        
        # Distance guidance based on face size
        distance_thresholds = self.quality_thresholds['distance_guidance']
        if face_size < distance_thresholds['too_small_threshold']:
            issues.append(f'Move CLOSER to camera ({face_size}px)')
            score -= 30  # Heavy penalty for being too far
        elif face_size < distance_thresholds['optimal_threshold']:
            issues.append(f'Move closer for better quality ({face_size}px)')
            score -= 10  # Light penalty, still acceptable
        elif face_size > distance_thresholds['too_close_threshold']:
            issues.append(f'Step back slightly - too close ({face_size}px)')
            score -= 15
        else:
            # Optimal distance range - bonus points!
            details['distance_optimal'] = True
            score += 5  # Small bonus for optimal distance
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Blur assessment (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        details['blur_score'] = blur_score
        
        if blur_score < self.quality_thresholds['blur_threshold']:
            issues.append(f'Image blurry (score: {blur_score:.0f})')
            score -= 20
        
        # Brightness assessment
        brightness = np.mean(gray)
        details['brightness'] = brightness
        
        if brightness < self.quality_thresholds['brightness_min']:
            issues.append(f'Too dark ({brightness:.0f})')
            score -= 15
        elif brightness > self.quality_thresholds['brightness_max']:
            issues.append(f'Too bright ({brightness:.0f})')
            score -= 15
        
        # Contrast assessment
        contrast = np.std(gray)
        details['contrast'] = contrast
        
        if contrast < self.quality_thresholds['contrast_min']:
            issues.append(f'Low contrast ({contrast:.0f})')
            score -= 10
        
        # Eye detection check
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        details['eyes_detected'] = len(eyes)
        
        if len(eyes) < 2:
            issues.append('Eyes not clearly visible')
            score -= 15
        
        # Embedding consistency check (if previous embeddings exist)
        if previous_embeddings:
            current_embedding = self.extract_embedding(face_crop)
            if current_embedding is not None:
                distances = []
                for prev_emb in previous_embeddings:
                    # Calculate cosine distance
                    similarity = torch.nn.functional.cosine_similarity(
                        current_embedding, prev_emb, dim=0
                    ).item()
                    distance = 1 - similarity
                    distances.append(distance)
                
                if distances:
                    max_distance = max(distances)
                    details['max_embedding_distance'] = max_distance
                    
                    if max_distance > self.quality_thresholds['embedding_consistency']:
                        issues.append(f'Inconsistent with previous captures')
                        score -= 5  # Reduced penalty
        
        score = max(0, score)
        
        return {
            'score': score,
            'issues': issues,
            'details': details
        }
    
    def extract_embedding(self, face_crop):
        """Extract normalized embedding from face crop"""
        try:
            if face_crop.size == 0:
                return None
            
            # Preprocess face
            face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
            rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb).convert('RGB')
            image_tensor = self.transform(pil_image).unsqueeze(0).to(DEVICE)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(image_tensor, mode='metric').cpu().squeeze()
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def run_registration(self, person_name):
        """Run the complete registration process"""
        print(f"\n" + "="*70)
        print(f"ENHANCED REGISTRATION: {person_name}")
        print("="*70)
        
        print(f"\nReady to capture {len(self.registration_poses)} poses. Follow on-screen instructions.")
        print(f"📋 POSE FLOW:")
        for i, pose in enumerate(self.registration_poses, 1):
            if i == 1:
                print(f"  {i}. {pose['instruction']} (automatic capture)")
            else:
                print(f"  {i}. {pose['instruction']} (press SPACE when ready)")
        
        print(f"\n💡 CONTROLS: ESC = cancel, SPACE = confirm pose change")
        
        input("\nPress Enter when you're ready to start...")
        
        # Initialize camera
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return False
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        captured_data = []
        current_pose_idx = 0
        stable_count = 0
        hold_start_time = None
        required_stable_frames = 20  # ~0.7 seconds at 30 FPS
        pose_detection_count = 0  # Count consecutive pose detections
        required_pose_frames = 15  # Need 15 consecutive frames with correct pose
        
        print(f"\nStarting registration in 3 seconds...")
        time.sleep(3)
        
        try:
            while current_pose_idx < len(self.registration_poses) and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading from camera")
                    break
                
                current_pose = self.registration_poses[current_pose_idx]
                display_frame = frame.copy()
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\n⚠️ Registration cancelled by user")
                    break
                
                # Detect faces
                faces = detect_faces(frame)
                
                if len(faces) == 0:
                    # No face detected
                    cv2.putText(display_frame, "NO FACE DETECTED", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    stable_count = 0
                    hold_start_time = None
                    pose_detection_count = 0
                    pose_detection_count = 0  # Reset pose detection when no face
                    
                elif len(faces) > 1:
                    # Multiple faces detected
                    cv2.putText(display_frame, "MULTIPLE FACES - STEP BACK", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    stable_count = 0
                    hold_start_time = None
                    pose_detection_count = 0
                    pose_detection_count = 0  # Reset pose detection when multiple faces
                    
                else:
                    # Single face detected - process it
                    x, y, w, h = faces[0]
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Extract and assess face quality
                    face_crop = crop_face_with_padding(frame, x, y, w, h)
                    
                    if face_crop.size > 0:
                        # Get quality assessment
                        previous_embeddings = [data['embedding'] for data in captured_data]
                        quality_result = self.assess_quality(face_crop, previous_embeddings)
                        
                        quality_score = quality_result['score']
                        issues = quality_result['issues']
                        
                        # Determine status and color
                        if quality_score >= self.quality_thresholds['required_quality_score']:
                            if len(issues) == 0:
                                status = "EXCELLENT"
                                color = (0, 255, 0)  # Green
                                stable_count += 1
                            else:
                                status = "GOOD"
                                color = (0, 255, 255)  # Yellow
                                stable_count += 1
                        else:
                            status = "POOR"
                            color = (0, 0, 255)  # Red
                            stable_count = 0
                            hold_start_time = None
                        
                        # Display pose instruction prominently
                        cv2.putText(display_frame, 
                                  f"POSE {current_pose_idx + 1}/{len(self.registration_poses)}: {current_pose['instruction']}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Display quality info
                        cv2.putText(display_frame, f"Quality: {quality_score:.0f}% - {status}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # ENHANCED: Show current pose instruction more clearly
                        cv2.putText(display_frame, f">>> {current_pose['instruction'].upper()} <<<", 
                                  (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        # Display issues (max 2)
                        for i, issue in enumerate(issues[:2]):
                            cv2.putText(display_frame, f"Issue: {issue}", 
                                      (10, 90 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
                        # ENHANCED: Automatic pose validation using MediaPipe
                        # Get pose angles for validation
                        pose_result = estimate_head_pose_angles(frame)
                        pose_ready = False
                        pose_feedback = "Processing..."
                        
                        if pose_result is not None:
                            yaw, pitch, roll, pose_label, _, _ = pose_result  # Unpack all 6 values
                            target_pose = current_pose['target_pose']
                            
                            # Validate pose using app.py logic
                            pose_matches, tolerance, feedback = validate_pose_for_target(
                                yaw, pitch, target_pose, 
                                strict_tolerance=15.0, relaxed_tolerance=20.0, is_strict=False
                            )
                            
                            if pose_matches:
                                pose_detection_count += 1
                                pose_feedback = f"✓ {feedback} ({pose_detection_count}/{required_pose_frames})"
                                
                                # Need consistent pose detection across multiple frames
                                if pose_detection_count >= required_pose_frames:
                                    pose_ready = True
                            else:
                                pose_detection_count = 0
                                pose_feedback = f"⚠ {feedback} (Turn more)"
                        else:
                            pose_detection_count = 0
                            pose_feedback = "Face landmarks not detected"
                        
                        # Display pose validation feedback
                        pose_color = (0, 255, 0) if pose_ready else (0, 165, 255)  # Green if ready, orange if not
                        cv2.putText(display_frame, pose_feedback, 
                                  (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
                        
                        if pose_ready and stable_count >= required_stable_frames and quality_score >= self.quality_thresholds['required_quality_score']:
                            if hold_start_time is None:
                                hold_start_time = time.time()
                                print(f"✓ Good quality achieved for {current_pose['id']}, hold for {current_pose['hold_time']}s...")
                            
                            # Show countdown
                            elapsed = time.time() - hold_start_time
                            remaining = max(0, current_pose['hold_time'] - elapsed)
                            
                            cv2.putText(display_frame, f"HOLD STEADY: {remaining:.1f}s", 
                                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            # Capture when timer completes
                            if remaining <= 0:
                                embedding = self.extract_embedding(face_crop)
                                if embedding is not None:
                                    # Store capture data
                                    capture_data = {
                                        'pose_id': current_pose['id'],
                                        'instruction': current_pose['instruction'],
                                        'embedding': embedding,
                                        'face_crop': face_crop.copy(),
                                        'quality_score': quality_score,
                                        'quality_details': quality_result['details'],
                                        'timestamp': datetime.now().isoformat()
                                    }
                                    captured_data.append(capture_data)
                                    
                                    print(f"✅ Captured {current_pose['id']} - Quality: {quality_score:.0f}%")
                                    
                                    # Move to next pose
                                    current_pose_idx += 1
                                    stable_count = 0
                                    hold_start_time = None
                                    pose_detection_count = 0  # Reset pose detection counter for next pose
                                    
                                    # Brief pause
                                    time.sleep(1)
                        
                        else:
                            # Reset counters if quality drops
                            if quality_score < self.quality_thresholds['required_quality_score']:
                                stable_count = 0
                                hold_start_time = None
                                pose_detection_count = 0  # Reset pose detection when quality drops
                
                # Progress bar
                progress = (current_pose_idx / len(self.registration_poses)) * 100
                bar_width = display_frame.shape[1] - 40
                bar_height = 20
                bar_y = display_frame.shape[0] - 40
                
                cv2.rectangle(display_frame, (20, bar_y), (20 + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                cv2.rectangle(display_frame, (20, bar_y), (20 + int(bar_width * progress / 100), bar_y + bar_height), (0, 255, 0), -1)
                cv2.putText(display_frame, f"Progress: {progress:.0f}%", 
                          (display_frame.shape[1] - 150, bar_y + 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow(f'Enhanced Registration - {person_name}', display_frame)
                
                # Key handling is done at the top of the loop
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Process results
        if len(captured_data) == len(self.registration_poses):
            return self.save_registration_data(person_name, captured_data)
        else:
            print(f"\n❌ Incomplete registration: {len(captured_data)}/{len(self.registration_poses)} poses captured")
            return False
    
    def save_registration_data(self, person_name, captured_data):
        """Save and validate the registration data"""
        print(f"\n📊 VALIDATING REGISTRATION FOR {person_name}")
        print("-" * 50)
        
        # Extract embeddings for validation
        embeddings = [data['embedding'] for data in captured_data]
        embeddings_tensor = torch.stack(embeddings)
        
        # Calculate embedding consistency
        distances = torch.cdist(embeddings_tensor, embeddings_tensor)
        
        # Get upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones_like(distances, dtype=bool), diagonal=1)
        pairwise_distances = distances[mask]
        
        max_distance = pairwise_distances.max().item()
        mean_distance = pairwise_distances.mean().item()
        
        # Calculate quality metrics
        avg_quality = np.mean([data['quality_score'] for data in captured_data])
        
        print(f"Embedding consistency:")
        print(f"  Max distance between embeddings: {max_distance:.4f}")
        print(f"  Mean distance: {mean_distance:.4f}")
        print(f"  Target: < {self.quality_thresholds['embedding_consistency']:.2f}")
        print(f"  Average quality score: {avg_quality:.1f}%")
        
        # Validation
        validation_passed = True
        warnings = []
        
        if max_distance > self.quality_thresholds['embedding_consistency']:
            warnings.append(f"High embedding variation ({max_distance:.4f})")
            # More lenient - only fail if extremely high variation
            if max_distance > 0.7:
                validation_passed = False
        
        if avg_quality < self.quality_thresholds['required_quality_score']:
            warnings.append(f"Low average quality ({avg_quality:.1f}%)")
            validation_passed = False
        
        if validation_passed:
            print(f"✅ Validation passed! Ready to save.")
        else:
            print(f"⚠️ Validation issues:")
            for warning in warnings:
                print(f"  - {warning}")
            
            choice = input(f"Save anyway? (y/N): ").strip().lower()
            if choice != 'y':
                print("Registration cancelled")
                return False
        
        # Save to database
        try:
            # Load existing database
            if EMPLOYEE_DB_PATH.exists():
                employee_db = torch.load(EMPLOYEE_DB_PATH, weights_only=False)
                # Create backup
                backup_path = Path("outputs") / f"employee_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                torch.save(employee_db, backup_path)
                print(f"  Database backup: {backup_path}")
            else:
                employee_db = {}
            
            # Save embeddings (list format for multi-embedding support)
            employee_db[person_name] = embeddings
            torch.save(employee_db, EMPLOYEE_DB_PATH)
            print(f"  Database updated: {EMPLOYEE_DB_PATH}")
            
            # Save detailed registration log
            registration_log = {
                'person_name': person_name,
                'registration_timestamp': datetime.now().isoformat(),
                'poses': [
                    {
                        'pose_id': data['pose_id'],
                        'instruction': data['instruction'],
                        'quality_score': data['quality_score'],
                        'quality_details': data['quality_details'],
                        'timestamp': data['timestamp']
                    } for data in captured_data
                ],
                'validation': {
                    'max_embedding_distance': max_distance,
                    'mean_embedding_distance': mean_distance,
                    'average_quality_score': avg_quality,
                    'validation_passed': validation_passed,
                    'warnings': warnings
                }
            }
            
            log_path = Path("outputs") / f"registration_log_{person_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_path, 'w') as f:
                json.dump(registration_log, f, indent=2)
            print(f"  Registration log: {log_path}")
            
            # Save identity images
            identity_dir = Path("identities") / person_name
            identity_dir.mkdir(parents=True, exist_ok=True)
            
            for i, data in enumerate(captured_data):
                img_path = identity_dir / f"frame_{i+1:02d}.jpg"
                cv2.imwrite(str(img_path), data['face_crop'])
            print(f"  Identity images: {identity_dir}")
            
            print(f"\n🎉 REGISTRATION SUCCESSFUL!")
            print(f"   Person: {person_name}")
            print(f"   Quality: {avg_quality:.1f}%")
            print(f"   Embedding consistency: {max_distance:.4f}")
            print(f"   Poses captured: {len(captured_data)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving registration: {e}")
            return False


def main():
    """Main registration interface"""
    print("🎯 OPTIMIZED STATIONARY REGISTRATION PROTOCOL v2.0")
    print("="*70)
    print("⚡ FASTER: 3 poses (vs 5), 1.5-2s each (vs 3s)")
    print("🔍 CLOSER: 2-3 feet distance for higher resolution features")
    print("🎯 BETTER: Stronger embeddings, less background noise")
    print("="*70)
    
    try:
        # Initialize registration system
        protocol = EnhancedRegistrationProtocol()
        
        print(f"\n📋 REGISTRATION: {len(protocol.registration_poses)} poses, ~1 minute total")
        
        # Get person to register
        person_name = input(f"\n👤 Enter name of person to register: ").strip()
        if not person_name:
            print("❌ No name provided")
            return
        
        print(f"\n📋 SETUP: Position 2-3 feet from camera, ensure good lighting")
        
        # Run registration
        success = protocol.run_registration(person_name)
        
        if success:
            print(f"\n🎉 REGISTRATION COMPLETE!")
            print(f"  {person_name} registered with optimized protocol")
            print(f"  This should significantly improve recognition accuracy")
            print(f"  Test with your main app: python app.py")
        else:
            print(f"\n😞 Registration incomplete or failed")
            print(f"  You can try again: python {__file__}")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Registration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()