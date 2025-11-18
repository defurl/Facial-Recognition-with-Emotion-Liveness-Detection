"""
Setup script for downloading dataset and configuring environment
"""

import sys
import os
import subprocess
import zipfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import DATASET_DIR, BASE_DIR


def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    print("=" * 70)
    print("KAGGLE API SETUP")
    print("=" * 70)
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("\nKaggle API credentials not found!")
        print("Please do one of the following:")
        print("1. Download kaggle.json from https://www.kaggle.com/account")
        print(f"2. Place it in: {kaggle_dir}")
        print("3. Or enter credentials manually below")
        
        response = input("\nEnter credentials manually? (y/n): ")
        if response.lower() == 'y':
            username = input("Kaggle username: ")
            key = input("Kaggle API key: ")
            
            import json
            credentials = {"username": username, "key": key}
            with open(kaggle_json, 'w') as f:
                json.dump(credentials, f)
            
            os.chmod(kaggle_json, 0o600)
            print(f"\n✓ Credentials saved to {kaggle_json}")
        else:
            print("\n✗ Setup cancelled. Please configure Kaggle API manually.")
            return False
    else:
        print(f"✓ Kaggle credentials found: {kaggle_json}")
    
    # Verify credentials
    try:
        import kaggle
        print("✓ Kaggle API authentication successful")
        return True
    except Exception as e:
        print(f"✗ Kaggle authentication failed: {e}")
        return False


def download_dataset():
    """Download and extract dataset from Kaggle"""
    print("\n" + "=" * 70)
    print("DATASET DOWNLOAD")
    print("=" * 70)
    
    # Create dataset directory
    DATASET_DIR.mkdir(exist_ok=True)
    
    # Check if already exists
    classification_dir = DATASET_DIR / "classification_data"
    if classification_dir.exists() and any(classification_dir.iterdir()):
        print(f"\n✓ Dataset already exists at: {DATASET_DIR}")
        return True
    
    print(f"\nDownloading dataset to: {DATASET_DIR}")
    print("This may take several minutes (~1.5GB)...")
    
    try:
        # Download from Kaggle
        result = subprocess.run([
            'kaggle', 'competitions', 'download',
            '-c', '11-785-fall-20-homework-2-part-2',
            '-p', str(DATASET_DIR)
        ], check=True, capture_output=True, text=True)
        
        print("✓ Dataset downloaded successfully")
        
        # Extract zip
        zip_file = DATASET_DIR / "11-785-fall-20-homework-2-part-2.zip"
        if zip_file.exists():
            print("\nExtracting dataset...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(DATASET_DIR)
            
            # Remove zip to save space
            zip_file.unlink()
            print("✓ Dataset extracted and zip removed")
        
        print(f"\n✓ Dataset ready at: {DATASET_DIR}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e}")
        print("\nPlease ensure you have:")
        print("1. Joined the competition: https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2")
        print("2. Configured Kaggle API correctly")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "=" * 70)
    print("CHECKING DEPENDENCIES")
    print("=" * 70)
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 'matplotlib',
        'seaborn', 'PIL', 'cv2', 'sklearn', 'tqdm'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'cv2':
                __import__('cv2')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages installed")
        return True


def create_directories():
    """Create necessary directories"""
    print("\n" + "=" * 70)
    print("CREATING DIRECTORIES")
    print("=" * 70)
    
    directories = [
        BASE_DIR / "outputs",
        BASE_DIR / "dataset",
        BASE_DIR / "src",
        BASE_DIR / "scripts"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"✓ {directory}")
    
    print("\n✓ All directories created")


def main():
    print("=" * 70)
    print("FACE RECOGNITION SYSTEM - SETUP")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Check dependencies")
    print("2. Create necessary directories")
    print("3. Setup Kaggle API")
    print("4. Download and extract dataset")
    print("\n" + "=" * 70)
    
    input("\nPress Enter to continue...")
    
    # Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n⚠ Please install missing dependencies first.")
        return
    
    # Create directories
    create_directories()
    
    # Setup Kaggle
    kaggle_ok = setup_kaggle_api()
    if not kaggle_ok:
        print("\n⚠ Kaggle setup failed. Dataset download skipped.")
        print("You can still use pre-trained models or download manually.")
        return
    
    # Download dataset
    dataset_ok = download_dataset()
    
    # Summary
    print("\n" + "=" * 70)
    print("SETUP SUMMARY")
    print("=" * 70)
    print(f"Dependencies: {'✓ OK' if deps_ok else '✗ MISSING'}")
    print(f"Kaggle API: {'✓ OK' if kaggle_ok else '✗ FAILED'}")
    print(f"Dataset: {'✓ OK' if dataset_ok else '✗ FAILED'}")
    print("=" * 70)
    
    if deps_ok and kaggle_ok and dataset_ok:
        print("\n✓ Setup complete! You can now:")
        print("  1. Train models: python scripts/train_softmax.py")
        print("  2. Train models: python scripts/train_metric.py")
        print("  3. Evaluate: python scripts/evaluate.py")
        print("  4. Run GUI: python app.py")
    else:
        print("\n⚠ Setup incomplete. Please resolve issues above.")


if __name__ == "__main__":
    main()
