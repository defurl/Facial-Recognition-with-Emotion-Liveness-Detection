"""
Explainability Demo Script

Demonstrates the new xAI features:
- Attention map visualization
- kNN confidence scoring
- Natural language explanations
- t-SNE embedding visualization
- Quality factor analysis

Usage:
    python demo_explainability.py --image path/to/face.jpg --name "John Doe"
"""

# Suppress warnings before any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # Suppress PyTorch hook warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)  # Suppress protobuf warnings
warnings.filterwarnings('ignore', message='.*SymbolDatabase.GetPrototype.*')  # Suppress protobuf specific

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import torch
import cv2
import numpy as np
from PIL import Image

from config import DEVICE, MODEL_METRIC_PATH, EMPLOYEE_DB_PATH, IMG_SIZE
from models import FaceEmbeddingCNN
from data_loader import get_transforms
from explainability import ExplainabilityEngine, explain_recognition
from deep_knn import (
    fit_knn, knn_search, knn_predict_with_confidence, 
    get_knn_explanation_text
)
from visualization_dashboard import create_embedding_dashboard
from utils import detect_faces, crop_face_with_padding


def parse_args():
    parser = argparse.ArgumentParser(description="Explainability Demo")
    parser.add_argument('--image', type=str, help='Path to face image')
    parser.add_argument('--name', type=str, default='Query', help='Name for query')
    parser.add_argument('--camera', action='store_true', help='Use camera input')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/explainability'),
                       help='Output directory for visualizations')
    parser.add_argument('--show-attention', action='store_true', default=True,
                       help='Show attention maps')
    parser.add_argument('--show-tsne', action='store_true', default=True,
                       help='Show t-SNE visualization')
    parser.add_argument('--show-knn', action='store_true', default=True,
                       help='Show kNN analysis')
    return parser.parse_args()


def load_model_and_db():
    """Load trained model and employee database."""
    print("Loading model and database...")
    
    # Load model
    model = FaceEmbeddingCNN(embedding_dim=256, num_classes=4000).to(DEVICE)
    if MODEL_METRIC_PATH.exists():
        model.load_state_dict(torch.load(MODEL_METRIC_PATH, map_location=DEVICE))
        model.eval()
        print(f"✓ Loaded model from {MODEL_METRIC_PATH}")
    else:
        print(f"⚠ Warning: Model not found at {MODEL_METRIC_PATH}")
        return None, None, None
    
    # Load database
    if EMPLOYEE_DB_PATH.exists():
        employee_db = torch.load(EMPLOYEE_DB_PATH)
        print(f"✓ Loaded {len(employee_db)} employees from database")
    else:
        print(f"⚠ Warning: Database not found at {EMPLOYEE_DB_PATH}")
        return model, None, None
    
    # Get transforms
    _, val_transform = get_transforms()
    
    return model, employee_db, val_transform


def extract_embeddings_from_db(employee_db):
    """Extract all embeddings and labels from employee database."""
    all_embeddings = []
    all_labels = []
    name_to_label = {}
    label_to_name = {}
    
    for idx, (name, data) in enumerate(sorted(employee_db.items())):
        name_to_label[name] = idx
        label_to_name[idx] = name
        
        if isinstance(data, list):
            embeddings = data
        else:
            embeddings = [data]
        
        for emb in embeddings:
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            # Ensure embedding is 1D
            if emb.ndim > 1:
                emb = emb.flatten()
            all_embeddings.append(emb)
            all_labels.append(idx)
    
    # Convert to numpy arrays with explicit dtype
    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    
    # Validate shape
    if all_embeddings.ndim != 2:
        raise ValueError(f"Embeddings should be 2D array, got shape {all_embeddings.shape}")
    
    return all_embeddings, all_labels, name_to_label, label_to_name


def process_image(image_path, model, employee_db, val_transform, args):
    """Process a single image and generate explanations."""
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        print("Please provide a valid image file path.")
        print(f"Example: python demo_explainability.py --image path/to/your/face.jpg")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect face
    print("Detecting face...")
    faces = detect_faces(image)
    if not faces:
        print("No face detected in image")
        return
    
    x, y, w, h = faces[0]
    face_crop = crop_face_with_padding(image, x, y, w, h, padding_ratio=0.2)
    face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    
    # Prepare tensor
    face_pil = Image.fromarray(face_crop)
    face_tensor = val_transform(face_pil).unsqueeze(0).to(DEVICE)
    
    # Get embedding
    print("Computing embedding...")
    with torch.no_grad():
        query_embedding = model(face_tensor, mode='metric')
    
    # Extract database embeddings
    print("Analyzing against database...")
    try:
        gallery_embeddings, gallery_labels, name_to_label, label_to_name = extract_embeddings_from_db(employee_db)
        print(f"✓ Extracted {len(gallery_embeddings)} embeddings from {len(name_to_label)} identities")
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        return
    
    # Validate we have enough data
    if len(gallery_embeddings) < 2:
        print("Error: Need at least 2 embeddings in database for kNN analysis")
        return
    
    # Fit kNN
    k = min(5, len(gallery_embeddings) - 1)
    knn = fit_knn(gallery_embeddings, metric='cosine', k=k)
    
    # Search
    query_emb_np = query_embedding.cpu().numpy()
    distances, neighbor_labels, neighbor_indices = knn_search(
        knn, gallery_embeddings, gallery_labels, query_emb_np, k=k
    )
    
    # Get predictions with confidence
    predicted_label, knn_confidence, knn_explanation = knn_predict_with_confidence(
        neighbor_labels[0], distances[0], return_explanation=True
    )
    
    predicted_name = label_to_name[predicted_label]
    neighbor_names = [label_to_name[label] for label in neighbor_labels[0]]
    
    print(f"\n{'='*60}")
    print(f"RECOGNITION RESULT")
    print(f"{'='*60}")
    print(f"Predicted Identity: {predicted_name}")
    print(f"kNN Confidence: {knn_confidence:.1%}")
    print(f"Distance to Match: {distances[0][0]:.3f}")
    print(f"{'='*60}\n")
    
    # Create explainability engine
    print("Generating explanations...")
    explainer = ExplainabilityEngine(model, DEVICE)
    
    # Generate comprehensive explanation
    explanation = explainer.generate_comprehensive_explanation(
        face_crop,
        face_tensor,
        predicted_name,
        distances[0][0],
        threshold=1.0,
        neighbor_names=neighbor_names,
        neighbor_distances=distances[0].tolist()
    )
    
    # Print summary
    print("\n" + explanation['summary'])
    
    # Save visualizations
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.show_attention:
        print("\nGenerating attention visualization...")
        vis = explainer.visualize_explanation(
            face_crop,
            explanation,
            save_path=str(args.output_dir / 'explanation_full.png')
        )
        print(f"✓ Saved to {args.output_dir / 'explanation_full.png'}")
        
        # Save attention overlay separately
        attention_overlay = explainer.overlay_attention_on_image(
            face_crop, explanation['attention_map']
        )
        cv2.imwrite(
            str(args.output_dir / 'attention_overlay.png'),
            cv2.cvtColor(attention_overlay, cv2.COLOR_RGB2BGR)
        )
        print(f"✓ Saved attention overlay to {args.output_dir / 'attention_overlay.png'}")
    
    if args.show_knn and knn_explanation:
        print("\nkNN Explanation:")
        print(get_knn_explanation_text(knn_explanation))
    
    if args.show_tsne:
        print("\nGenerating t-SNE visualization...")
        dashboard = create_embedding_dashboard(employee_db, model, DEVICE, val_transform)
        
        # Plot with query
        tsne_vis = dashboard.plot(
            query_embedding=query_emb_np[0],
            query_name=args.name,
            query_label=predicted_label,
            neighbor_indices=neighbor_indices[0].tolist(),
            save_path=str(args.output_dir / 'tsne_visualization.png')
        )
        print(f"✓ Saved t-SNE visualization to {args.output_dir / 'tsne_visualization.png'}")
        
        # Print cluster metrics
        print("\n" + dashboard.get_metrics_summary())


def process_camera(model, employee_db, val_transform, args):
    """Process camera feed with real-time explanations."""
    print("Starting camera feed... Press 'q' to quit, 's' to save explanation")
    
    cap = cv2.VideoCapture(0)
    explainer = ExplainabilityEngine(model, DEVICE)
    
    # Extract database
    try:
        gallery_embeddings, gallery_labels, name_to_label, label_to_name = extract_embeddings_from_db(employee_db)
        print(f"✓ Extracted {len(gallery_embeddings)} embeddings from {len(name_to_label)} identities")
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        cap.release()
        return
    
    # Validate we have enough data
    if len(gallery_embeddings) < 2:
        print("Error: Need at least 2 embeddings in database for kNN analysis")
        cap.release()
        return
    
    k = min(5, len(gallery_embeddings) - 1)
    knn = fit_knn(gallery_embeddings, metric='cosine', k=k)
    
    # Persistent display state (prevents flickering)
    last_face_bbox = None
    last_predicted_name = "Unknown"
    last_confidence = 0.0
    last_attention_overlay = None
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_frame = frame.copy()
        
        # Process every 5th frame
        if frame_count % 5 == 0:
            faces = detect_faces(frame_rgb)
            
            if faces:
                x, y, w, h = faces[0]
                face_crop = crop_face_with_padding(frame_rgb, x, y, w, h, padding_ratio=0.2)
                face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                
                # Get embedding
                face_pil = Image.fromarray(face_crop)
                face_tensor = val_transform(face_pil).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    query_embedding = model(face_tensor, mode='metric')
                    query_emb_np = query_embedding.cpu().numpy()
                
                # kNN search
                distances, neighbor_labels, _ = knn_search(
                    knn, gallery_embeddings, gallery_labels, query_emb_np, k=k
                )
                
                predicted_label, knn_confidence, _ = knn_predict_with_confidence(
                    neighbor_labels[0], distances[0], return_explanation=False
                )
                
                predicted_name = label_to_name[predicted_label]
                
                # Update persistent state
                last_face_bbox = (x, y, w, h)
                last_predicted_name = predicted_name
                last_confidence = knn_confidence
                
                # Generate attention overlay if needed
                if args.show_attention:
                    attention_map = explainer.generate_attention_map(face_tensor)
                    attention_overlay = explainer.overlay_attention_on_image(face_crop, attention_map)
                    attention_overlay_small = cv2.resize(attention_overlay, (128, 128))
                    last_attention_overlay = cv2.cvtColor(attention_overlay_small, cv2.COLOR_RGB2BGR)
        
        # Draw persistent results (even when no face detected this frame)
        if last_face_bbox is not None:
            x, y, w, h = last_face_bbox
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            text = f"{last_predicted_name} ({last_confidence:.1%})"
            cv2.putText(display_frame, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show attention map overlay in top-right corner
            if args.show_attention and last_attention_overlay is not None:
                h_overlay, w_overlay = last_attention_overlay.shape[:2]
                display_frame[10:10+h_overlay, -w_overlay-10:-10] = last_attention_overlay
        
        cv2.imshow('Explainable Face Recognition (Press Q to quit, S to save)', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current explanation
            cv2.imwrite(str(args.output_dir / 'camera_capture.png'), display_frame)
            print(f"✓ Saved frame to {args.output_dir / 'camera_capture.png'}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    
    print("="*60)
    print("EXPLAINABILITY DEMO")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Output directory: {args.output_dir}")
    print("="*60 + "\n")
    
    # Load model and database
    model, employee_db, val_transform = load_model_and_db()
    
    if model is None or employee_db is None:
        print("Error: Could not load model or database")
        return
    
    if args.camera:
        process_camera(model, employee_db, val_transform, args)
    elif args.image:
        process_image(args.image, model, employee_db, val_transform, args)
    else:
        print("\n⚠ No input specified!")
        print("\nOptions:")
        print("  1. Use camera:        python demo_explainability.py --camera")
        print("  2. Use image file:    python demo_explainability.py --image path/to/face.jpg")
        print("\nTo capture a test image, run:")
        print("  python capture_test_image.py")
        print("\nThen use the captured image:")
        print("  python demo_explainability.py --image test_face.jpg --name 'Your Name'")


if __name__ == "__main__":
    main()
