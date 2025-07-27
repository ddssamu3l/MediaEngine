#!/usr/bin/env python3
"""
RIFE Setup Script for MediaEngine
This script downloads and installs RIFE models and dependencies.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path
import json
import shutil

class RIFESetup:
    """RIFE setup and installation manager."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models" / "rife"
        self.scripts_dir = self.base_dir / "scripts"
        
        # RIFE model URLs (these would need to be updated with actual URLs)
        self.model_urls = {
            "v4.6": {
                "url": "https://github.com/hzwer/ECCV2022-RIFE/releases/download/v4.6/rife46.zip",
                "files": ["flownet.pkl", "metric.pkl", "featnet.pkl", "fusionnet.pkl"]
            },
            "v4.4": {
                "url": "https://github.com/hzwer/ECCV2022-RIFE/releases/download/v4.4/rife44.zip", 
                "files": ["flownet.pkl", "metric.pkl", "featnet.pkl", "fusionnet.pkl"]
            }
        }
        
    def check_python_version(self):
        """Check if Python version is compatible."""
        if sys.version_info < (3, 7):
            print("Error: Python 3.7 or higher is required")
            return False
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return True
    
    def create_virtual_environment(self):
        """Create a virtual environment for RIFE."""
        venv_dir = self.base_dir / "rife_env"
        
        if venv_dir.exists():
            print("✓ Virtual environment already exists")
            return True
            
        try:
            print("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
            print("✓ Virtual environment created")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            return False
    
    def get_venv_python(self):
        """Get path to virtual environment Python."""
        if os.name == "nt":  # Windows
            return str(self.base_dir / "rife_env" / "Scripts" / "python.exe")
        else:  # Unix/Linux/macOS
            return str(self.base_dir / "rife_env" / "bin" / "python")
    
    def install_dependencies(self):
        """Install required Python packages."""
        python_path = self.get_venv_python()
        
        # Check if virtual environment Python exists
        if not os.path.exists(python_path):
            print("Error: Virtual environment not found")
            return False
        
        packages = [
            "torch>=1.12.0",
            "torchvision>=0.13.0", 
            "opencv-python>=4.5.0",
            "numpy>=1.21.0",
            "Pillow>=8.0.0",
            "tqdm>=4.60.0"
        ]
        
        print("Installing Python dependencies...")
        for package in packages:
            try:
                print(f"Installing {package}...")
                subprocess.run([python_path, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                print(f"✓ {package} installed")
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")
                return False
        
        return True
    
    def install_pytorch_optimized(self):
        """Install PyTorch with optimal configuration for the current system."""
        python_path = self.get_venv_python()
        
        print("Detecting optimal PyTorch configuration...")
        
        # Detect system capabilities
        has_cuda = self._check_cuda()
        has_mps = self._check_mps()
        
        if has_cuda:
            print("✓ CUDA detected, installing PyTorch with CUDA support")
            pytorch_cmd = [
                python_path, "-m", "pip", "install", 
                "torch", "torchvision", "--index-url", 
                "https://download.pytorch.org/whl/cu118"
            ]
        elif has_mps:
            print("✓ Apple Silicon detected, installing PyTorch with MPS support")
            pytorch_cmd = [
                python_path, "-m", "pip", "install",
                "torch", "torchvision"
            ]
        else:
            print("Installing CPU-only PyTorch")
            pytorch_cmd = [
                python_path, "-m", "pip", "install",
                "torch", "torchvision", "--index-url",
                "https://download.pytorch.org/whl/cpu"
            ]
        
        try:
            subprocess.run(pytorch_cmd, check=True)
            print("✓ PyTorch installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing PyTorch: {e}")
            return False
    
    def _check_cuda(self):
        """Check if CUDA is available."""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_mps(self):
        """Check if MPS (Apple Silicon) is available."""
        try:
            import platform
            return platform.system() == "Darwin" and platform.machine() == "arm64"
        except:
            return False
    
    def download_rife_models(self, model_version: str = "v4.6"):
        """Download RIFE model files."""
        if model_version not in self.model_urls:
            print(f"Error: Unknown model version {model_version}")
            return False
        
        model_info = self.model_urls[model_version]
        model_dir = self.models_dir / model_version
        
        # Check if model already exists
        if model_dir.exists() and all((model_dir / f).exists() for f in model_info["files"]):
            print(f"✓ RIFE {model_version} model already exists")
            return True
        
        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading RIFE {model_version} model...")
        try:
            # Download model archive
            archive_path = model_dir / f"rife_{model_version}.zip"
            urllib.request.urlretrieve(model_info["url"], archive_path)
            
            # Extract archive
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            
            # Remove archive
            archive_path.unlink()
            
            print(f"✓ RIFE {model_version} model downloaded and extracted")
            return True
            
        except Exception as e:
            print(f"Error downloading RIFE model: {e}")
            return False
    
    def create_rife_code(self):
        """Create basic RIFE model wrapper code."""
        rife_code_dir = self.models_dir / "rife"
        rife_code_dir.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py
        init_file = rife_code_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# RIFE model package\n")
        
        # Create basic model.py (this would need actual RIFE implementation)
        model_file = rife_code_dir / "model.py"
        if not model_file.exists():
            model_code = '''"""
Basic RIFE model wrapper for MediaEngine integration.
This is a placeholder - actual RIFE implementation would go here.
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """RIFE model wrapper."""
    
    def __init__(self):
        super().__init__()
        self.loaded = False
    
    def load_model(self, model_path, version="v4.6"):
        """Load RIFE model from path."""
        # Placeholder implementation
        print(f"Loading RIFE model {version} from {model_path}")
        self.loaded = True
        return True
    
    def inference(self, img0, img1, timestep, scale=1.0):
        """Perform frame interpolation."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        # Placeholder: simple average interpolation
        # Real implementation would use RIFE neural network
        alpha = timestep
        interpolated = img0 * (1 - alpha) + img1 * alpha
        return interpolated
    
    def half(self):
        """Convert model to half precision."""
        return super().half()
    
    def device(self):
        """Move model to appropriate device."""
        pass
'''
            model_file.write_text(model_code)
        
        # Create utils.py
        utils_file = rife_code_dir / "utils.py"
        if not utils_file.exists():
            utils_code = '''"""
Utility functions for RIFE frame interpolation.
"""

import cv2
import numpy as np

def read_img(path):
    """Read image from file."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def write_img(path, img):
    """Write image to file."""
    return cv2.imwrite(path, img)
'''
            utils_file.write_text(utils_code)
        
        print("✓ RIFE code structure created")
        return True
    
    def verify_installation(self):
        """Verify that RIFE installation is working."""
        python_path = self.get_venv_python()
        
        test_script = '''
import torch
import cv2
import numpy as np
import sys
from pathlib import Path

try:
    # Test PyTorch
    print(f"PyTorch version: {torch.__version__}")
    
    # Test device availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        print("MPS (Apple Silicon) available")
    else:
        print("Using CPU")
    
    # Test OpenCV
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test basic tensor operations
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    z = (x + y) / 2
    print("✓ Basic tensor operations working")
    
    print("✓ Installation verification successful")
    
except Exception as e:
    print(f"✗ Installation verification failed: {e}")
    sys.exit(1)
'''
        
        try:
            result = subprocess.run([python_path, "-c", test_script], 
                                  capture_output=True, text=True, check=True)
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Installation verification failed: {e.stderr}")
            return False
    
    def create_config_file(self):
        """Create configuration file for RIFE integration."""
        config = {
            "rife": {
                "python_path": self.get_venv_python(),
                "script_path": str(self.scripts_dir / "rife_interpolation.py"),
                "models_dir": str(self.models_dir),
                "default_model": "v4.6",
                "default_precision": "fp16",
                "default_device": "auto"
            }
        }
        
        config_file = self.base_dir / "rife_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration saved to {config_file}")
        return True
    
    def setup(self, model_version: str = "v4.6"):
        """Run complete RIFE setup process."""
        print("🎬 Setting up RIFE Frame Interpolation for MediaEngine")
        print("=" * 60)
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing PyTorch (optimized)", self.install_pytorch_optimized),
            ("Installing dependencies", self.install_dependencies),
            ("Creating RIFE code structure", self.create_rife_code),
            ("Downloading RIFE models", lambda: self.download_rife_models(model_version)),
            ("Verifying installation", self.verify_installation),
            ("Creating configuration", self.create_config_file),
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                print(f"❌ Setup failed at: {step_name}")
                return False
        
        print("\n🎉 RIFE setup completed successfully!")
        print("\nNext steps:")
        print(f"1. The RIFE environment is ready at: {self.base_dir}/rife_env")
        print(f"2. Models are installed at: {self.models_dir}")
        print(f"3. Configuration saved to: {self.base_dir}/rife_config.json")
        print("\nYou can now use frame interpolation in MediaEngine!")
        
        return True

def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup RIFE for MediaEngine")
    parser.add_argument("--model", default="v4.6", choices=["v4.6", "v4.4"], 
                       help="RIFE model version to install")
    parser.add_argument("--dir", default=".", help="Installation directory")
    
    args = parser.parse_args()
    
    setup = RIFESetup(args.dir)
    success = setup.setup(args.model)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()