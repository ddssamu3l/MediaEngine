#!/usr/bin/env python3
"""
Check the exact channel configurations in the actual weights.
"""

import torch

def check_channels():
    weights_path = "/Users/dengjingxi/Documents/code/DataAnnotation/MediaEngine/models/rife/rife46.pth"
    weights = torch.load(weights_path, map_location='cpu')
    
    print("=== CHANNEL ANALYSIS ===")
    
    for block_name in ['block0', 'block1', 'block2', 'block3']:
        print(f"\n{block_name.upper()}:")
        
        # conv0.0.0 (first layer of conv0)
        key = f"{block_name}.conv0.0.0.weight"
        if key in weights:
            shape = weights[key].shape  # [out_channels, in_channels, H, W]
            print(f"  conv0.0.0: {shape[1]} -> {shape[0]} (input->output)")
            
        # conv0.1.0 (second layer of conv0)  
        key = f"{block_name}.conv0.1.0.weight"
        if key in weights:
            shape = weights[key].shape
            print(f"  conv0.1.0: {shape[1]} -> {shape[0]} (input->output)")
            
        # convblock.0.conv (first convblock layer)
        key = f"{block_name}.convblock.0.conv.weight"
        if key in weights:
            shape = weights[key].shape
            print(f"  convblock: {shape[1]} -> {shape[0]} (same throughout)")
            
        # lastconv.0 (output layer)
        key = f"{block_name}.lastconv.0.weight"
        if key in weights:
            shape = weights[key].shape
            print(f"  lastconv: {shape[1]} -> {shape[0]} (ConvTranspose2d)")
            print(f"  Expected output: 5 channels? Actual: {shape[0]}")
            
        print(f"  Block summary:")
        conv0_0_key = f"{block_name}.conv0.0.0.weight"
        conv0_1_key = f"{block_name}.conv0.1.0.weight"
        convblock_key = f"{block_name}.convblock.0.conv.weight"
        
        if all(k in weights for k in [conv0_0_key, conv0_1_key, convblock_key]):
            input_ch = weights[conv0_0_key].shape[1]
            c_half = weights[conv0_0_key].shape[0]  
            c = weights[conv0_1_key].shape[0]
            convblock_ch = weights[convblock_key].shape[0]
            
            print(f"    Input channels: {input_ch}")
            print(f"    c//2 = {c_half}")
            print(f"    c = {c}")
            print(f"    convblock channels: {convblock_ch}")
            print(f"    Relationship: c = {c}, convblock = {convblock_ch}")

if __name__ == "__main__":
    check_channels()