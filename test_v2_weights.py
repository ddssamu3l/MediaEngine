#!/usr/bin/env python3
"""
Test the v2 architecture against actual weights.
"""

import torch
import sys
sys.path.append('/Users/dengjingxi/Documents/code/DataAnnotation/MediaEngine/models/rife')
from rife_architecture_v2 import RIFEModel

def test_weight_loading():
    print("=== TESTING V2 ARCHITECTURE ===")
    
    # Create model
    model = RIFEModel()
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load weights
    model_path = "/Users/dengjingxi/Documents/code/DataAnnotation/MediaEngine/models/rife"
    success = model.load_model(model_path)
    
    if success:
        print("✓ Model loaded successfully!")
        
        # Test a simple forward pass
        print("\nTesting forward pass...")
        model.eval()
        
        # Create dummy input: 2 RGB images + 1 dummy GT = 9 channels
        dummy_input = torch.randn(1, 9, 64, 64)
        
        try:
            with torch.no_grad():
                output = model.forward(dummy_input)
            print("✓ Forward pass successful!")
            print(f"Output structure: {len(output)} elements")
            if len(output) >= 3:
                print(f"  flow_list: {len(output[0])} flows")
                print(f"  mask shape: {output[1].shape}")
                print(f"  merged: {len(output[2])} merged frames")
            
            return True
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("✗ Model loading failed")
        return False

if __name__ == "__main__":
    test_weight_loading()