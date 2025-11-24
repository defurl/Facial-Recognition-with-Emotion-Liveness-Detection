#!/usr/bin/env python
"""
Quick verification that demo_explainability.py is working correctly.
Tests the functions without needing camera or image files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from PIL import Image

print("="*60)
print("DEMO VERIFICATION TEST")
print("="*60)

# Test 1: Import demo functions
print("\n[1/4] Testing imports...")
try:
    from demo_explainability import (
        load_model_and_db,
        extract_embeddings_from_db
    )
    print("✓ Demo imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load model and database
print("\n[2/4] Testing model and database loading...")
try:
    model, employee_db, val_transform = load_model_and_db()
    if model is None or employee_db is None:
        print("✗ Failed to load model or database")
        sys.exit(1)
    print(f"✓ Model and database loaded ({len(employee_db)} employees)")
except Exception as e:
    print(f"✗ Loading failed: {e}")
    sys.exit(1)

# Test 3: Extract embeddings
print("\n[3/4] Testing embedding extraction...")
try:
    gallery_embeddings, gallery_labels, name_to_label, label_to_name = extract_embeddings_from_db(employee_db)
    print(f"✓ Extracted {len(gallery_embeddings)} embeddings")
    print(f"  Shape: {gallery_embeddings.shape}")
    print(f"  Dtype: {gallery_embeddings.dtype}")
    print(f"  Identities: {list(name_to_label.keys())}")
except Exception as e:
    print(f"✗ Extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test inference
print("\n[4/4] Testing model inference...")
try:
    # Create dummy face image
    dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    dummy_pil = Image.fromarray(dummy_image)
    dummy_tensor = val_transform(dummy_pil).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run inference
    with torch.no_grad():
        embedding = model(dummy_tensor, mode='metric')
    
    print(f"✓ Inference successful")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {torch.norm(embedding).item():.3f}")
except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("VERIFICATION SUMMARY")
print("="*60)
print("✓ All tests passed!")
print("\nDemo script is ready to use:")
print("  • python demo_explainability.py --camera")
print("  • python demo_explainability.py --image test_face.jpg")
print("="*60)
