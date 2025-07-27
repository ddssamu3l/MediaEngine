#!/usr/bin/env python3
"""
Test script to verify RIFE model loads real weights and produces proper interpolation.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import sys

# Add the current directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

from model import Model


def create_test_frames():
    """Create simple test frames for interpolation."""
    # Create a simple checkerboard pattern that moves
    height, width = 256, 256
    
    # Frame 0: Checkerboard at position 0
    frame0 = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(0, height, 32):
        for j in range(0, width, 32):
            if (i // 32 + j // 32) % 2 == 0:
                frame0[i:i+32, j:j+32] = [255, 255, 255]
    
    # Frame 1: Checkerboard shifted by 16 pixels
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(0, height, 32):
        for j in range(0, width, 32):
            if (i // 32 + j // 32) % 2 == 0:
                # Shift by 16 pixels
                start_j = min(j + 16, width - 32)
                frame1[i:i+32, start_j:start_j+32] = [255, 255, 255]
    
    return frame0, frame1


def frames_to_tensor(frame):
    """Convert numpy frame to PyTorch tensor."""
    # Convert from BGR to RGB and normalize to [0, 1]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).float() / 255.0
    # Change from HWC to CHW and add batch dimension
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_frame(tensor):
    """Convert PyTorch tensor back to numpy frame."""
    # Remove batch dimension and convert from CHW to HWC
    frame = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Convert from [0, 1] to [0, 255] and from RGB to BGR
    frame = (frame * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame_bgr


def test_model_loading():
    """Test that the model loads with real weights."""
    print("Testing RIFE model loading...")
    
    model = Model()
    model_path = Path(__file__).parent
    
    # Test loading
    success = model.load_model(model_path)
    
    if success and model.loaded:
        print("✓ Model loaded successfully with real weights")
        
        # Check model size (real weights should be much larger)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {total_params:,} parameters")
        
        if total_params > 1000000:  # Should have > 1M parameters
            print("✓ Model size indicates real weights (>1M parameters)")
            return True
        else:
            print("✗ Model size too small, may still be using placeholder weights")
            return False
    else:
        print("✗ Failed to load model")
        return False


def test_interpolation():
    """Test that the model produces meaningful interpolation."""
    print("\nTesting frame interpolation...")
    
    # Create model
    model = Model()
    model_path = Path(__file__).parent
    
    if not model.load_model(model_path):
        print("✗ Could not load model for interpolation test")
        return False
    
    # Set device and move model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create test frames
    frame0_np, frame1_np = create_test_frames()
    
    # Convert to tensors and move to device
    img0 = frames_to_tensor(frame0_np).to(device)
    img1 = frames_to_tensor(frame1_np).to(device)
    
    print(f"Input frames shape: {img0.shape}")
    
    try:
        # Perform interpolation at timestep 0.5
        with torch.no_grad():
            interpolated = model.inference(img0, img1, timestep=0.5)
        
        print(f"Interpolated frame shape: {interpolated.shape}")
        
        # Convert back to numpy
        result_frame = tensor_to_frame(interpolated)
        
        # Check if the result is not just grey/blank
        mean_intensity = np.mean(result_frame)
        std_intensity = np.std(result_frame)
        
        print(f"Result mean intensity: {mean_intensity:.2f}")
        print(f"Result std intensity: {std_intensity:.2f}")
        
        # Check if the result has meaningful variation (not just grey)
        if std_intensity > 10:  # Should have some variation
            print("✓ Interpolated frame has meaningful content (not blank/grey)")
            
            # Save test results for visual inspection
            output_dir = Path(__file__).parent / "test_output"
            output_dir.mkdir(exist_ok=True)
            
            cv2.imwrite(str(output_dir / "frame0.png"), frame0_np)
            cv2.imwrite(str(output_dir / "frame1.png"), frame1_np)
            cv2.imwrite(str(output_dir / "interpolated.png"), result_frame)
            
            print(f"✓ Test frames saved to {output_dir}")
            return True
        else:
            print("✗ Interpolated frame appears to be blank or grey")
            return False
            
    except Exception as e:
        print(f"✗ Error during interpolation: {e}")
        return False


def main():
    """Run all tests."""
    print("RIFE Real Weights Test")
    print("=" * 50)
    
    # Test 1: Model loading
    loading_ok = test_model_loading()
    
    # Test 2: Interpolation
    interpolation_ok = test_interpolation()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Model Loading: {'PASS' if loading_ok else 'FAIL'}")
    print(f"Interpolation: {'PASS' if interpolation_ok else 'FAIL'}")
    
    if loading_ok and interpolation_ok:
        print("\n🎉 All tests passed! RIFE is working with real weights.")
        return True
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)