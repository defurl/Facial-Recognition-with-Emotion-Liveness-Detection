#!/usr/bin/env python3
"""Rebuild employee database from identities folder.

Usage:
    python scripts/rebuild_db.py
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
import cv2

from src.config import DEVICE, MODEL_METRIC_PATH, OUTPUT_DIR
from src.runtime.models_loader import load_verification_model
from src.data_loader import get_transforms

EMPLOYEE_DB_PATH = OUTPUT_DIR / "employee_db.pt"
IDENTITIES_DIR = ROOT / "identities"


def main():
    print("=" * 60)
    print("REBUILD EMPLOYEE DATABASE FROM IDENTITIES FOLDER")
    print("=" * 60)
    
    # Load model
    print(f"\n[1/3] Loading model from {MODEL_METRIC_PATH}...")
    model = load_verification_model(MODEL_METRIC_PATH, DEVICE, num_classes=4000)
    model.eval()
    
    # Get transforms
    _, val_transform = get_transforms()
    
    # Scan identities folder
    print(f"\n[2/3] Scanning {IDENTITIES_DIR}...")
    employee_db = {}
    
    if not IDENTITIES_DIR.exists():
        print(f"ERROR: Identities folder not found at {IDENTITIES_DIR}")
        return
    
    for person_dir in sorted(IDENTITIES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        
        # Find all images in this person's folder
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        if not images:
            print(f"  [SKIP] {person_dir.name} - no images found")
            continue
        
        print(f"  [PROCESS] {person_dir.name} - {len(images)} images")
        
        embeddings = []
        for img_path in images:
            try:
                # Load and preprocess image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb).convert("RGB")
                tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)
                
                # Extract embedding
                with torch.no_grad():
                    embedding = model(tensor, mode="metric")
                embeddings.append(embedding.cpu())
            except Exception as e:
                print(f"    [ERROR] {img_path.name}: {e}")
        
        if embeddings:
            # Store all embeddings for this person (stacked tensor)
            employee_db[person_dir.name] = torch.cat(embeddings, dim=0)
            print(f"    -> Stored {len(embeddings)} embeddings")
    
    # Save database
    print(f"\n[3/3] Saving database to {EMPLOYEE_DB_PATH}...")
    torch.save(employee_db, EMPLOYEE_DB_PATH)
    
    print(f"\n{'=' * 60}")
    print(f"SUCCESS! Rebuilt database with {len(employee_db)} employees:")
    for name, emb in employee_db.items():
        print(f"  - {name}: {emb.shape[0]} embeddings")
    print("=" * 60)


if __name__ == "__main__":
    main()
