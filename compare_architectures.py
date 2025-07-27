#!/usr/bin/env python3
"""
Compare RIFE model architectures - actual weights vs our implementation.
"""

import torch
import sys
sys.path.append('/Users/dengjingxi/Documents/code/DataAnnotation/MediaEngine/models/rife')
from rife_architecture import RIFEModel

def compare_architectures():
    print("=== ARCHITECTURE COMPARISON ===")
    
    # Load the actual weights
    weights_path = "/Users/dengjingxi/Documents/code/DataAnnotation/MediaEngine/models/rife/rife46.pth"
    actual_weights = torch.load(weights_path, map_location='cpu')
    
    # Create our model and get its state dict
    our_model = RIFEModel()
    our_state_dict = our_model.state_dict()
    
    print(f"Actual weights keys: {len(actual_weights)}")
    print(f"Our model keys: {len(our_state_dict)}")
    
    print(f"\n=== ACTUAL WEIGHTS STRUCTURE ===")
    actual_keys = list(actual_weights.keys())
    
    # The actual weights don't have flownet prefix!
    print("Key structure analysis:")
    print("- All keys start directly with 'block' (block0, block1, block2, block3)")
    print("- There's a block3 (which we call block_tea)")
    print("- No 'flownet.' prefix")
    print("- Uses '.beta' parameters (likely for PReLU)")
    print("- Different channel sizes:")
    
    # Analyze channel sizes for each block
    for block_name in ['block0', 'block1', 'block2', 'block3']:
        # Find the first conv layer to determine input channels
        conv_key = f"{block_name}.conv0.0.0.weight"
        if conv_key in actual_weights:
            shape = actual_weights[conv_key].shape
            print(f"  {block_name}: input_channels={shape[1]}, output_channels={shape[0]}")
            
        # Find the convblock to determine the main channels
        convblock_key = f"{block_name}.convblock.0.conv.weight"
        if convblock_key in actual_weights:
            shape = actual_weights[convblock_key].shape
            print(f"    convblock channels: {shape[0]}")
    
    print(f"\n=== OUR MODEL STRUCTURE ===")
    our_keys = list(our_state_dict.keys())
    print("Key structure:")
    print("- All keys start with 'flownet.' prefix")
    print("- Uses 'block0', 'block1', 'block2', 'block_tea'")
    print("- Our channel configurations:")
    
    # Show a few key examples from our model
    for key in our_keys[:10]:
        shape = our_state_dict[key].shape
        print(f"  {key}: {shape}")
    
    print(f"\n=== KEY MAPPING ANALYSIS ===")
    
    # Try to map keys
    mapped_keys = 0
    unmapped_actual = []
    unmapped_ours = []
    
    for actual_key in actual_keys:
        # Try to find corresponding key in our model
        # Remove flownet prefix and map block3 to block_tea
        our_key = actual_key
        if actual_key.startswith('block3.'):
            our_key = actual_key.replace('block3.', 'block_tea.')
        our_key = 'flownet.' + our_key
        
        if our_key in our_state_dict:
            actual_shape = actual_weights[actual_key].shape
            our_shape = our_state_dict[our_key].shape
            if actual_shape == our_shape:
                mapped_keys += 1
            else:
                print(f"SHAPE MISMATCH: {actual_key} {actual_shape} vs {our_key} {our_shape}")
        else:
            # Check for potential naming differences
            # The actual weights use .beta for PReLU, we might use different naming
            if '.beta' in actual_key:
                # Our PReLU might be stored differently
                prelu_key = actual_key.replace('.beta', '.weight')
                if actual_key.startswith('block3.'):
                    prelu_key = prelu_key.replace('block3.', 'block_tea.')
                prelu_key = 'flownet.' + prelu_key
                if prelu_key in our_state_dict:
                    print(f"PReLU mapping: {actual_key} -> {prelu_key}")
                    mapped_keys += 1
                else:
                    unmapped_actual.append(actual_key)
            else:
                unmapped_actual.append(actual_key)
    
    # Check for keys in our model that don't exist in actual weights
    for our_key in our_keys:
        # Remove flownet prefix and map block_tea to block3
        actual_key = our_key.replace('flownet.', '')
        if actual_key.startswith('block_tea.'):
            actual_key = actual_key.replace('block_tea.', 'block3.')
        
        if actual_key not in actual_keys:
            unmapped_ours.append(our_key)
    
    print(f"\nMapped keys: {mapped_keys}")
    print(f"Unmapped actual keys: {len(unmapped_actual)}")
    print(f"Unmapped our keys: {len(unmapped_ours)}")
    
    if unmapped_actual:
        print(f"\nFirst 10 unmapped actual keys:")
        for key in unmapped_actual[:10]:
            print(f"  {key}: {actual_weights[key].shape}")
    
    if unmapped_ours:
        print(f"\nFirst 10 unmapped our keys:")
        for key in unmapped_ours[:10]:
            print(f"  {key}: {our_state_dict[key].shape}")

if __name__ == "__main__":
    compare_architectures()