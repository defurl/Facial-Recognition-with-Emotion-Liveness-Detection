#!/usr/bin/env python3
"""
Create improved synthetic liveness dataset with more realistic spoof augmentations.

Key improvements over simple blur/noise:
1. Print attack simulation - Halftone patterns, moire effects
2. Replay attack simulation - Screen refresh patterns, color shifts
3. Paper/print artifacts - Flat lighting, no skin texture
4. Boundary/edge artifacts - Sharp cutoffs, paper edges
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import random
from tqdm import tqdm

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from config import BASE_DIR

EMOTION_DIR = BASE_DIR / 'dataset' / 'emotion' / 'DATASET'
OUTPUT_DIR = BASE_DIR / 'dataset' / 'liveness' / 'improved_synthetic'

random.seed(42)
np.random.seed(42)


def add_moire_pattern(img: Image.Image, intensity: float = 0.3) -> Image.Image:
    """Add moiré pattern (common in photos of screens)."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    
    # Create interference pattern
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    freq = random.uniform(0.1, 0.3)
    angle = random.uniform(0, np.pi)
    
    pattern = np.sin(2 * np.pi * freq * (x * np.cos(angle) + y * np.sin(angle)))
    pattern = (pattern * intensity * 30).astype(np.float32)
    
    # Apply to all channels
    arr[:, :, 0] = np.clip(arr[:, :, 0] + pattern, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + pattern * 0.8, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + pattern * 0.6, 0, 255)
    
    return Image.fromarray(arr.astype(np.uint8))


def add_halftone_effect(img: Image.Image, dot_size: int = 3) -> Image.Image:
    """Add halftone pattern (printed photos)."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    
    # Create dot grid
    result = arr.copy()
    
    for y in range(0, h, dot_size * 2):
        for x in range(0, w, dot_size * 2):
            if random.random() < 0.3:  # 30% chance of visible dot
                # Slightly darken this region
                y_end = min(y + dot_size, h)
                x_end = min(x + dot_size, w)
                result[y:y_end, x:x_end] = np.clip(result[y:y_end, x:x_end] * 0.9, 0, 255)
    
    return Image.fromarray(result)


def add_screen_lines(img: Image.Image, intensity: float = 0.15) -> Image.Image:
    """Add horizontal scan lines (screen capture artifact)."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    
    # Add horizontal lines at regular intervals
    line_spacing = random.randint(2, 4)
    for y in range(0, h, line_spacing):
        factor = random.uniform(1 - intensity, 1 + intensity * 0.5)
        arr[y, :, :] = np.clip(arr[y, :, :] * factor, 0, 255)
    
    return Image.fromarray(arr.astype(np.uint8))


def add_color_shift(img: Image.Image) -> Image.Image:
    """Add color channel shift (screen/camera mismatch)."""
    arr = np.array(img, dtype=np.float32)
    
    # Random per-channel adjustments
    r_factor = random.uniform(0.9, 1.1)
    g_factor = random.uniform(0.9, 1.1)
    b_factor = random.uniform(0.9, 1.1)
    
    arr[:, :, 0] = np.clip(arr[:, :, 0] * r_factor, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * g_factor, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * b_factor, 0, 255)
    
    return Image.fromarray(arr.astype(np.uint8))


def reduce_texture(img: Image.Image, strength: float = 0.5) -> Image.Image:
    """Reduce skin texture (flat look of printed photos)."""
    # Apply bilateral-like smoothing
    blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
    
    # Blend with original
    arr = np.array(img, dtype=np.float32)
    blurred_arr = np.array(blurred, dtype=np.float32)
    
    result = arr * (1 - strength) + blurred_arr * strength
    return Image.fromarray(result.astype(np.uint8))


def add_paper_texture(img: Image.Image) -> Image.Image:
    """Add paper grain/texture overlay."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    
    # Generate paper grain noise
    noise = np.random.randn(h, w) * 8  # Subtle paper texture
    
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] + noise, 0, 255)
    
    return Image.fromarray(arr.astype(np.uint8))


def add_reflection(img: Image.Image, intensity: float = 0.2) -> Image.Image:
    """Add specular reflection (glossy print or screen)."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    
    # Create gradient hotspot
    cx, cy = w * random.uniform(0.2, 0.8), h * random.uniform(0.2, 0.8)
    radius = min(h, w) * random.uniform(0.3, 0.6)
    
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    # Gaussian falloff
    reflection = np.exp(-dist ** 2 / (2 * (radius / 2) ** 2)) * intensity * 100
    
    arr[:, :, :] = np.clip(arr[:, :, :] + reflection[:, :, np.newaxis], 0, 255)
    
    return Image.fromarray(arr.astype(np.uint8))


def add_jpeg_artifacts(img: Image.Image, quality: int = 30) -> Image.Image:
    """Add JPEG compression artifacts (common in captured/printed images)."""
    import io
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def add_boundary_artifact(img: Image.Image) -> Image.Image:
    """Add border/boundary artifacts (paper edges, mask boundaries)."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    
    # Darken edges slightly (paper shadow)
    edge_width = random.randint(3, 8)
    darken = 0.85
    
    arr[:edge_width, :, :] *= darken
    arr[-edge_width:, :, :] *= darken
    arr[:, :edge_width, :] *= darken
    arr[:, -edge_width:, :] *= darken
    
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def create_print_attack(img: Image.Image) -> Image.Image:
    """Simulate print attack (photo on paper)."""
    result = img.copy()
    
    # Apply multiple effects
    effects = [
        lambda x: reduce_texture(x, random.uniform(0.3, 0.6)),  # Flat look
        lambda x: add_halftone_effect(x) if random.random() > 0.5 else x,
        lambda x: add_paper_texture(x),
        lambda x: add_reflection(x, random.uniform(0.1, 0.25)) if random.random() > 0.5 else x,
        lambda x: add_jpeg_artifacts(x, random.randint(25, 50)),
        lambda x: add_boundary_artifact(x) if random.random() > 0.5 else x,
    ]
    
    for effect in effects:
        result = effect(result)
    
    # Adjust overall appearance
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(random.uniform(0.85, 1.1))
    
    return result


def create_replay_attack(img: Image.Image) -> Image.Image:
    """Simulate replay attack (photo/video on screen)."""
    result = img.copy()
    
    effects = [
        lambda x: add_moire_pattern(x, random.uniform(0.15, 0.35)),
        lambda x: add_screen_lines(x, random.uniform(0.1, 0.2)),
        lambda x: add_color_shift(x),
        lambda x: add_reflection(x, random.uniform(0.15, 0.3)) if random.random() > 0.5 else x,
        lambda x: add_jpeg_artifacts(x, random.randint(40, 70)),
    ]
    
    for effect in effects:
        result = effect(result)
    
    # Screens often have slightly warmer/cooler tones
    enhancer = ImageEnhance.Color(result)
    result = enhancer.enhance(random.uniform(0.9, 1.15))
    
    return result


def create_spoof_image(img: Image.Image) -> Image.Image:
    """Create a spoof image using random attack type."""
    attack_type = random.choice(['print', 'replay'])
    
    if attack_type == 'print':
        return create_print_attack(img)
    else:
        return create_replay_attack(img)


def process_dataset(split: str = 'train', max_samples: int = None):
    """Process emotion dataset to create liveness pairs."""
    input_dir = EMOTION_DIR / split
    output_live_dir = OUTPUT_DIR / split / 'live'
    output_spoof_dir = OUTPUT_DIR / split / 'spoof'
    
    output_live_dir.mkdir(parents=True, exist_ok=True)
    output_spoof_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all images
    all_images = []
    for class_dir in input_dir.iterdir():
        if class_dir.is_dir():
            for img_file in class_dir.glob('*.jpg'):
                all_images.append(img_file)
            for img_file in class_dir.glob('*.png'):
                all_images.append(img_file)
    
    random.shuffle(all_images)
    
    if max_samples:
        all_images = all_images[:max_samples]
    
    print(f"Processing {len(all_images)} images for {split} split...")
    
    live_count = 0
    spoof_count = 0
    
    for idx, img_path in enumerate(tqdm(all_images)):
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Resize to consistent size
            img = img.resize((112, 112), Image.Resampling.LANCZOS)
            
            # Decide if this becomes live or spoof (50/50 split)
            if idx % 2 == 0:
                # Save as live (with minimal augmentation)
                output_path = output_live_dir / f'live_{idx:05d}.jpg'
                img.save(output_path, quality=95)
                live_count += 1
            else:
                # Create and save spoof
                spoof_img = create_spoof_image(img)
                output_path = output_spoof_dir / f'spoof_{idx:05d}.jpg'
                spoof_img.save(output_path, quality=90)
                spoof_count += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"  Created {live_count} live, {spoof_count} spoof images")
    return live_count, spoof_count


def main():
    print("=" * 60)
    print("Creating Improved Synthetic Liveness Dataset")
    print("=" * 60)
    print(f"Input: {EMOTION_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Create train and test splits
    train_live, train_spoof = process_dataset('train')
    test_live, test_spoof = process_dataset('test')
    
    # Save metadata
    import json
    from datetime import datetime
    
    metadata = {
        'dataset': 'improved_synthetic_liveness',
        'created': datetime.now().isoformat(),
        'source': 'RAF-DB emotion dataset',
        'improvements': [
            'Moiré patterns for replay attacks',
            'Halftone patterns for print attacks',
            'Screen scan lines',
            'Color channel shifts',
            'Reduced texture (flat look)',
            'Paper grain texture',
            'Specular reflections',
            'JPEG compression artifacts',
            'Boundary/edge artifacts'
        ],
        'train': {'live': train_live, 'spoof': train_spoof},
        'test': {'live': test_live, 'spoof': test_spoof}
    }
    
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("=" * 60)
    print("Dataset Creation Complete!")
    print("=" * 60)
    print(f"Train: {train_live} live + {train_spoof} spoof = {train_live + train_spoof}")
    print(f"Test: {test_live} live + {test_spoof} spoof = {test_live + test_spoof}")
    print(f"Metadata saved to: {OUTPUT_DIR / 'metadata.json'}")


if __name__ == '__main__':
    main()
