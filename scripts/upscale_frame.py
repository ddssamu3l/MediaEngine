#!/usr/bin/env python3
"""
Real-ESRGAN Frame Upscaler
Proof of concept for upscaling single video frames
"""

import argparse
import cv2
import os
import sys
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np

class FrameUpscaler:
    def __init__(self, model_name='RealESRGAN_x4plus', scale=4, gpu_id=None):
        """
        Initialize the upscaler
        
        Args:
            model_name: Model to use ('RealESRGAN_x4plus', 'RealESRGAN_x2plus', 'RealESRGAN_x4plus_anime_6B')
            scale: Upscaling factor (2, 3, or 4)
            gpu_id: GPU ID to use (None for CPU)
        """
        self.model_name = model_name
        self.scale = scale
        self.gpu_id = gpu_id
        self.upsampler = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Real-ESRGAN model"""
        try:
            # Model configurations
            model_configs = {
                'RealESRGAN_x4plus': {
                    'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                    'netscale': 4,
                    'num_block': 23,
                    'num_feat': 64
                },
                'RealESRGAN_x2plus': {
                    'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRGAN_x2plus.pth',
                    'netscale': 2,
                    'num_block': 23,
                    'num_feat': 64
                },
                'RealESRGAN_x4plus_anime_6B': {
                    'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
                    'netscale': 4,
                    'num_block': 6,
                    'num_feat': 64
                }
            }
            
            config = model_configs.get(self.model_name)
            if not config:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            # Create model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=config['num_feat'],
                num_block=config['num_block'],
                num_grow_ch=32,
                scale=config['netscale']
            )
            
            # Initialize upsampler
            self.upsampler = RealESRGANer(
                scale=config['netscale'],
                model_path=config['model_path'],
                model=model,
                tile=0,  # No tiling for single frames
                tile_pad=10,
                pre_pad=0,
                half=False,  # Use FP32 for better quality
                gpu_id=self.gpu_id
            )
            
            print(f"✅ Initialized {self.model_name} (scale: {config['netscale']}x)")
            
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            sys.exit(1)
    
    def upscale_frame(self, input_path, output_path):
        """
        Upscale a single frame
        
        Args:
            input_path: Path to input image
            output_path: Path to save upscaled image
        """
        try:
            # Read image
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not read image: {input_path}")
            
            print(f"📥 Input: {img.shape[1]}x{img.shape[0]} pixels")
            
            # Upscale
            print("🔄 Upscaling...")
            output, _ = self.upsampler.enhance(img, outscale=self.scale)
            
            print(f"📤 Output: {output.shape[1]}x{output.shape[0]} pixels")
            
            # Save result
            cv2.imwrite(output_path, output)
            print(f"✅ Saved upscaled frame to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Upscaling failed: {e}")
            return False
    
    def get_info(self):
        """Get model information"""
        return {
            'model_name': self.model_name,
            'scale': self.scale,
            'gpu_enabled': self.gpu_id is not None
        }

def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN Frame Upscaler')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--model', default='RealESRGAN_x4plus', 
                       choices=['RealESRGAN_x4plus', 'RealESRGAN_x2plus', 'RealESRGAN_x4plus_anime_6B'],
                       help='Model to use')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 3, 4],
                       help='Upscaling factor')
    parser.add_argument('--gpu', type=int, help='GPU ID to use (default: CPU)')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize upscaler
    upscaler = FrameUpscaler(
        model_name=args.model,
        scale=args.scale,
        gpu_id=args.gpu
    )
    
    # Print info
    info = upscaler.get_info()
    print(f"🎯 Model: {info['model_name']}")
    print(f"📈 Scale: {info['scale']}x")
    print(f"🖥️  GPU: {'Enabled' if info['gpu_enabled'] else 'CPU only'}")
    
    # Upscale frame
    success = upscaler.upscale_frame(args.input, args.output)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 