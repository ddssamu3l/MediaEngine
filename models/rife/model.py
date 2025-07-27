"""
RIFE model implementation for MediaEngine frame interpolation.
This implementation uses the real RIFE v4.6 architecture with proper model weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path

# Import the real RIFE architecture
try:
    from .rife_architecture import RIFEModel
except ImportError:
    from rife_architecture import RIFEModel


class Model(nn.Module):
    """RIFE model wrapper for MediaEngine compatibility."""
    
    def __init__(self):
        super().__init__()
        
        # Use the real RIFE model
        self.rife_model = RIFEModel()
        self.loaded = False
        
    def load_model(self, model_path, version="v4.6"):
        """Load real RIFE model weights."""
        model_path = Path(model_path)
        
        # Try to load the real model weights
        success = self.rife_model.load_model(model_path)
        
        if success:
            self.loaded = True
            print(f"RIFE model {version} loaded successfully with real weights")
            return True
        else:
            print(f"Failed to load real weights from {model_path}")
            return False
    
    def inference(self, img0, img1, timestep, scale=1.0):
        """Perform frame interpolation inference using real RIFE model."""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use the real RIFE model for inference
        return self.rife_model.inference(img0, img1, timestep, scale)
    
    def half(self):
        """Convert model to half precision."""
        self.rife_model = self.rife_model.half()
        return self
    
    def to(self, device):
        """Move model to device."""
        self.rife_model = self.rife_model.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        self.rife_model.eval()
        return self