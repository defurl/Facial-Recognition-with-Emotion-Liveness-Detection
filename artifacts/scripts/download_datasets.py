#!/usr/bin/env python3
"""
Dataset Download Script for MTL Training
Downloads RAF-DB (emotion) and CelebA-Spoof (liveness) datasets from Kaggle.

Usage:
    python artifacts/scripts/download_datasets.py --emotion
    python artifacts/scripts/download_datasets.py --liveness
    python artifacts/scripts/download_datasets.py --all
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure we can import from src
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

DATASET_DIR = ROOT_DIR / "dataset"
EMOTION_DIR = DATASET_DIR / "emotion"
LIVENESS_DIR = DATASET_DIR / "liveness"


def download_rafdb():
    """Download RAF-DB emotion dataset (~39MB)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    print("=" * 60)
    print("DOWNLOADING RAF-DB EMOTION DATASET")
    print("=" * 60)
    print(f"Target directory: {EMOTION_DIR}")
    print("Dataset: shuvoalok/raf-db-dataset (~39MB)")
    print("-" * 60)
    
    EMOTION_DIR.mkdir(parents=True, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()
    
    print("Starting download...")
    api.dataset_download_files(
        dataset="shuvoalok/raf-db-dataset",
        path=str(EMOTION_DIR),
        unzip=True,
        quiet=False
    )
    
    print("\n✓ RAF-DB download complete!")
    print(f"  Location: {EMOTION_DIR}")
    
    # List downloaded contents
    contents = list(EMOTION_DIR.iterdir())
    print(f"  Contents: {len(contents)} items")
    for item in contents[:10]:
        print(f"    - {item.name}")
    if len(contents) > 10:
        print(f"    ... and {len(contents) - 10} more")


def download_celeba_spoof():
    """Download CelebA-Spoof liveness dataset (~78GB)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    print("=" * 60)
    print("DOWNLOADING CELEBA-SPOOF LIVENESS DATASET")
    print("=" * 60)
    print(f"Target directory: {LIVENESS_DIR}")
    print("Dataset: attentionlayer241/celeba-spoof-for-face-antispoofing (~78GB)")
    print("-" * 60)
    print("⚠️  WARNING: This is a LARGE dataset (~78GB)!")
    print("   Estimated download time: 1-4 hours depending on connection")
    print("-" * 60)
    
    LIVENESS_DIR.mkdir(parents=True, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()
    
    print("Starting download...")
    api.dataset_download_files(
        dataset="attentionlayer241/celeba-spoof-for-face-antispoofing",
        path=str(LIVENESS_DIR),
        unzip=True,
        quiet=False
    )
    
    print("\n✓ CelebA-Spoof download complete!")
    print(f"  Location: {LIVENESS_DIR}")
    
    # List downloaded contents
    contents = list(LIVENESS_DIR.iterdir())
    print(f"  Contents: {len(contents)} items")
    for item in contents[:10]:
        print(f"    - {item.name}")
    if len(contents) > 10:
        print(f"    ... and {len(contents) - 10} more")


def check_disk_space():
    """Check if there's enough disk space for downloads."""
    import shutil
    
    total, used, free = shutil.disk_usage(DATASET_DIR.parent)
    free_gb = free / (1024 ** 3)
    
    print(f"Available disk space: {free_gb:.1f} GB")
    
    if free_gb < 100:
        print("⚠️  WARNING: Less than 100GB free space!")
        print("   CelebA-Spoof requires ~78GB for download + extraction")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)
    
    return free_gb


def main():
    parser = argparse.ArgumentParser(description="Download datasets for MTL training")
    parser.add_argument("--emotion", action="store_true", help="Download RAF-DB emotion dataset")
    parser.add_argument("--liveness", action="store_true", help="Download CelebA-Spoof liveness dataset")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--skip-space-check", action="store_true", help="Skip disk space check")
    
    args = parser.parse_args()
    
    if not any([args.emotion, args.liveness, args.all]):
        parser.print_help()
        print("\nError: Please specify at least one dataset to download.")
        sys.exit(1)
    
    print("=" * 60)
    print("MTL DATASET DOWNLOADER")
    print("=" * 60)
    
    if not args.skip_space_check:
        check_disk_space()
    
    if args.emotion or args.all:
        download_rafdb()
        print()
    
    if args.liveness or args.all:
        download_celeba_spoof()
        print()
    
    print("=" * 60)
    print("ALL DOWNLOADS COMPLETE")
    print("=" * 60)
    print(f"Emotion dataset: {EMOTION_DIR}")
    print(f"Liveness dataset: {LIVENESS_DIR}")


if __name__ == "__main__":
    main()
