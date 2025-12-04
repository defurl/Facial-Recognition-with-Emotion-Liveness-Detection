#!/usr/bin/env python3
"""
Simple Registration Test
Tests if better registration improves recognition accuracy
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append('src')

from config import *
from models import FaceEmbeddingCNN
from data_loader import get_transforms
from utils import detect_faces, crop_face_with_padding

def test_current_database():
    """Test current database quality"""
    print("🔍 TESTING CURRENT DATABASE QUALITY")
    print("-" * 50)
    
    if not EMPLOYEE_DB_PATH.exists():
        print("❌ No database found")
        return
    
    # Load database
    db = torch.load(EMPLOYEE_DB_PATH, weights_only=False)
    print(f"People in database: {list(db.keys())}")
    
    # Simple quality check
    total_embeddings = 0
    for name, data in db.items():
        if isinstance(data, list):
            num_embeddings = len(data)
        else:
            num_embeddings = 1
        total_embeddings += num_embeddings
        print(f"  {name}: {num_embeddings} embeddings")
    
    print(f"Total embeddings: {total_embeddings}")
    
    # Check embedding similarity within same person
    print(f"\n📊 Checking embedding consistency...")
    
    for name, data in db.items():
        if isinstance(data, list) and len(data) > 1:
            embeddings = torch.stack(data)
            distances = torch.cdist(embeddings, embeddings)
            max_distance = distances.max().item()
            print(f"  {name}: max intra-distance = {max_distance:.3f} {'✓' if max_distance < 0.3 else '❌'}")

def simple_registration_demo():
    """Demo of what good registration should look like"""
    print(f"\n🎯 REGISTRATION DEMO")
    print("-" * 50)
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Camera not available")
        return
    
    print("📷 Camera opened. Press 'q' to quit, 's' to see quality assessment")
    
    # Load model for quality assessment
    model = FaceEmbeddingCNN(embedding_dim=256, num_classes=4000).to(DEVICE)
    if MODEL_METRIC_PATH.exists():
        model.load_state_dict(torch.load(MODEL_METRIC_PATH, map_location=DEVICE))
        model.eval()
        print("✓ Model loaded")
    
    _, transform = get_transforms()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = detect_faces(frame)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Quick quality assessment
                face_crop = crop_face_with_padding(frame, x, y, w, h)
                if face_crop.size > 0:
                    # Size check
                    size_ok = min(face_crop.shape[:2]) >= 100
                    
                    # Blur check
                    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    blur_ok = blur_score > 100
                    
                    # Brightness check
                    brightness = np.mean(gray)
                    brightness_ok = 60 <= brightness <= 180
                    
                    # Overall quality
                    quality_score = sum([size_ok, blur_ok, brightness_ok]) / 3 * 100
                    color = (0, 255, 0) if quality_score > 66 else (0, 255, 255) if quality_score > 33 else (0, 0, 255)
                    
                    cv2.putText(frame, f"Quality: {quality_score:.0f}%", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Registration Quality Demo - Press q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"Current frame quality assessment shown in window")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("🔧 SIMPLE REGISTRATION TEST")
    print("=" * 50)
    print("This helps you understand the registration problem and solution")
    print()
    
    # Test current database
    test_current_database()
    
    # Show what good registration looks like
    print(f"\n💡 SOLUTION: Better registration captures")
    print("   - Consistent poses (5 angles)")
    print("   - Good lighting (60-180 brightness)")
    print("   - Sharp images (blur score > 100)")
    print("   - Large face size (> 100 pixels)")
    print("   - Stable capture (hold pose for 3 seconds)")
    
    choice = input(f"\nWant to see quality demo with your camera? (y/n): ").lower()
    if choice == 'y':
        simple_registration_demo()
    
    print(f"\n📋 NEXT STEP:")
    print(f"   Run: python stationary_registration_protocol.py")
    print(f"   Re-register one person to test improvement")

if __name__ == "__main__":
    main()