#!/usr/bin/env python3
"""
Analyze the structure of RIFE model weights to understand architecture mismatches.
"""

import torch
from pathlib import Path

def analyze_weights():
    weights_path = "/Users/dengjingxi/Documents/code/DataAnnotation/MediaEngine/models/rife/rife46.pth"
    
    print("=== ANALYZING RIFE WEIGHTS ===")
    print(f"Loading: {weights_path}")
    
    # Load the weights file
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    print("\n=== CHECKPOINT STRUCTURE ===")
    print(f"Top-level keys: {list(checkpoint.keys())}")
    
    # Get the actual state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Using 'model' key from checkpoint")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict'] 
        print("Using 'state_dict' key from checkpoint")
    else:
        state_dict = checkpoint
        print("Using checkpoint directly as state_dict")
    
    print(f"\nState dict has {len(state_dict)} keys")
    
    # Analyze the key structure
    print("\n=== WEIGHT KEYS ANALYSIS ===")
    all_keys = list(state_dict.keys())
    
    # Group keys by their structure
    flownet_keys = [k for k in all_keys if 'flownet' in k]
    block_keys = [k for k in all_keys if any(x in k for x in ['block0', 'block1', 'block2', 'block_tea'])]
    conv_keys = [k for k in all_keys if 'conv' in k]
    other_keys = [k for k in all_keys if not any(x in k for x in ['flownet', 'block', 'conv'])]
    
    print(f"Keys containing 'flownet': {len(flownet_keys)}")
    if flownet_keys:
        print("  Examples:")
        for key in flownet_keys[:5]:
            shape = state_dict[key].shape
            print(f"    {key}: {shape}")
        if len(flownet_keys) > 5:
            print(f"    ... and {len(flownet_keys) - 5} more")
    
    print(f"\nKeys containing block names: {len(block_keys)}")
    if block_keys:
        print("  Examples:")
        for key in block_keys[:10]:
            shape = state_dict[key].shape
            print(f"    {key}: {shape}")
        if len(block_keys) > 10:
            print(f"    ... and {len(block_keys) - 10} more")
    
    print(f"\nKeys containing 'conv': {len(conv_keys)}")
    if conv_keys:
        print("  Examples:")
        for key in conv_keys[:10]:
            shape = state_dict[key].shape
            print(f"    {key}: {shape}")
        if len(conv_keys) > 10:
            print(f"    ... and {len(conv_keys) - 10} more")
    
    print(f"\nOther keys: {len(other_keys)}")
    if other_keys:
        for key in other_keys:
            shape = state_dict[key].shape
            print(f"    {key}: {shape}")
    
    # Show all keys for detailed analysis
    print(f"\n=== ALL KEYS ({len(all_keys)} total) ===")
    for i, key in enumerate(all_keys):
        shape = state_dict[key].shape
        dtype = state_dict[key].dtype
        print(f"{i+1:3d}. {key}: {shape} ({dtype})")
    
    print("\n=== KEY PATTERNS ===")
    # Analyze common patterns
    patterns = {}
    for key in all_keys:
        parts = key.split('.')
        if len(parts) >= 2:
            pattern = '.'.join(parts[:2])  # First two levels
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(key)
    
    print("Common key patterns:")
    for pattern, keys in sorted(patterns.items()):
        print(f"  {pattern}: {len(keys)} keys")
        if len(keys) <= 3:
            for key in keys:
                print(f"    - {key}")
        else:
            for key in keys[:2]:
                print(f"    - {key}")
            print(f"    - ... and {len(keys) - 2} more")

if __name__ == "__main__":
    analyze_weights()