#!/usr/bin/env python3
"""
Debug the model output to see why it's producing blank frames.
"""

import torch
import numpy as np
import cv2
import sys
sys.path.append('/Users/dengjingxi/Documents/code/DataAnnotation/MediaEngine/models/rife')
from model import Model

def debug_interpolation():
    print("=== DEBUGGING INTERPOLATION OUTPUT ===")
    
    # Create model
    model = Model()
    model_path = "/Users/dengjingxi/Documents/code/DataAnnotation/MediaEngine/models/rife"
    model.load_model(model_path)
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create simple test frames
    height, width = 64, 64
    
    # Frame 0: white square on black background
    frame0 = torch.zeros(1, 3, height, width, device=device)
    frame0[:, :, 16:48, 16:48] = 1.0  # White square
    
    # Frame 1: white square moved to the right
    frame1 = torch.zeros(1, 3, height, width, device=device)
    frame1[:, :, 16:48, 32:48] = 1.0  # Moved white square
    
    print(f"Frame0 stats: mean={frame0.mean():.4f}, std={frame0.std():.4f}")
    print(f"Frame1 stats: mean={frame1.mean():.4f}, std={frame1.std():.4f}")
    
    with torch.no_grad():
        # Get internal forward pass results
        x = torch.cat([frame0, frame1], dim=1)  # 6 channels
        flow_list, mask, merged, _, _, _ = model.rife_model.forward(x, timestep=0.5)
        
        print(f"\nForward pass results:")
        print(f"Flow list length: {len(flow_list)}")
        for i, flow in enumerate(flow_list):
            print(f"  Flow[{i}] shape: {flow.shape}, mean: {flow.mean():.6f}, std: {flow.std():.6f}")
        
        print(f"Mask shape: {mask.shape}, mean: {mask.mean():.6f}, std: {mask.std():.6f}")
        
        print(f"Merged list length: {len(merged)}")
        for i, merge in enumerate(merged):
            print(f"  Merged[{i}] shape: {merge.shape}, mean: {merge.mean():.6f}, std: {merge.std():.6f}")
        
        # Test inference method
        result = model.inference(frame0, frame1, timestep=0.5)
        print(f"\nInference result:")
        print(f"  Shape: {result.shape}")
        print(f"  Mean: {result.mean():.6f}")
        print(f"  Std: {result.std():.6f}")
        print(f"  Min: {result.min():.6f}")
        print(f"  Max: {result.max():.6f}")
        
        # Check if any pixels are non-zero
        non_zero_pixels = (result > 1e-6).sum().item()
        print(f"  Non-zero pixels: {non_zero_pixels} / {result.numel()}")

if __name__ == "__main__":
    debug_interpolation()