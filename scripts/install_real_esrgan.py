#!/usr/bin/env python3
"""
Install and setup real Real-ESRGAN with model downloads
"""

import os
import sys
import subprocess
import urllib.request
import hashlib
from pathlib import Path

def download_file(url, dest_path, expected_hash=None):
    """Download file with progress and hash verification"""
    print(f"Downloading {os.path.basename(dest_path)}...")
    
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        sys.stdout.write(f'\rProgress: {percent:.1f}%')
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest_path, reporthook=download_progress)
    print()  # New line after progress
    
    if expected_hash:
        print("Verifying file integrity...")
        with open(dest_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash != expected_hash:
            os.remove(dest_path)
            raise ValueError(f"Hash mismatch! Expected {expected_hash}, got {file_hash}")
    
    print(f"✅ Downloaded {os.path.basename(dest_path)}")

def main():
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("🚀 Setting up Real-ESRGAN with actual AI models...\n")
    
    # Install real Real-ESRGAN package
    print("📦 Installing Real-ESRGAN package...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "realesrgan", "basicsr", "facexlib", "gfpgan",
        "--upgrade"
    ])
    
    # Model URLs and hashes
    models = {
        "RealESRGAN_x4plus.pth": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "hash": "aa00f09ad753d88576b21ed977e97d634976377031b178acc3ea9a9d43b05c5b"
        },
        "RealESRGAN_x4plus_anime_6B.pth": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "hash": "f872d837d3c7f5c403d5e0f74e1e3d676d89b6b8f30b7e0b90a9a330a533da2f"
        },
        "RealESRGAN_x2plus.pth": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "hash": "49fafd45f8fd7aa8d31ab2a22d14a91b536c7e3b9b8989ca2b9c9cd3d0d3f6c5"
        }
    }
    
    # Download models
    print("\n📥 Downloading AI models...")
    for model_name, model_info in models.items():
        model_path = models_dir / model_name
        if model_path.exists():
            print(f"✓ {model_name} already exists")
        else:
            try:
                download_file(model_info["url"], str(model_path), model_info["hash"])
            except Exception as e:
                print(f"❌ Failed to download {model_name}: {e}")
                print("Continuing with other models...")
    
    print("\n✅ Real-ESRGAN setup complete!")
    print(f"Models installed in: {models_dir.absolute()}")
    
    # Test import
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        print("\n✅ Real-ESRGAN imports successful!")
    except ImportError as e:
        print(f"\n❌ Import test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()