#!/usr/bin/env python3
"""
Working Real-ESRGAN implementation using the actual realesrgan package
"""

import argparse
import cv2
import os
import sys
import numpy as np

# Try to use the real realesrgan package
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    HAS_REAL_ESRGAN = True
except ImportError:
    HAS_REAL_ESRGAN = False
    print("⚠️ Real-ESRGAN package not available, using high-quality fallback")

import torch
import torch.nn.functional as F
import subprocess
import platform

def get_device(args):
    """Get the best available device"""
    if args.cpu:
        device = torch.device("cpu")
        print("💻 Using CPU")
        return device, False
    elif args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"🎮 Using CUDA GPU {args.gpu}")
        return device, True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # Get Apple Silicon info
        try:
            result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                  capture_output=True, text=True, timeout=5)
            if 'Chip:' in result.stdout:
                chip_line = [line for line in result.stdout.split('\n') if 'Chip:' in line][0]
                chip_name = chip_line.split('Chip:')[1].strip()
                print(f"🍎 Using Apple Silicon GPU: {chip_name} (MPS)")
            else:
                print("🍎 Using Apple Silicon GPU (MPS)")
        except:
            print("🍎 Using Apple Silicon GPU (MPS)")
        return device, True
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎮 Using CUDA GPU")
        return device, True
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
        return device, False

def create_real_esrgan_upscaler(model_name, scale, device, use_half):
    """Create a real Real-ESRGAN upscaler"""
    if not HAS_REAL_ESRGAN:
        return None
    
    try:
        # Model configurations
        if model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = 'models/RealESRGAN_x4plus.pth'
            netscale = 4
        elif model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            model_path = 'models/RealESRGAN_x2plus.pth'
            netscale = 2
        elif model_name == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            model_path = 'models/RealESRGAN_x4plus_anime_6B.pth'
            netscale = 4
        else:
            return None
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None
        
        # Create upsampler with proper settings
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=0,  # No tiling for now
            tile_pad=10,
            pre_pad=0,
            half=use_half and device.type != 'cpu',
            device=device
        )
        
        print(f"✅ Real-ESRGAN model loaded: {model_name}")
        return upsampler
        
    except Exception as e:
        print(f"❌ Failed to create Real-ESRGAN upsampler: {e}")
        return None

def upscale_with_pytorch_fallback(img, scale, device):
    """High-quality fallback using PyTorch with proper aspect ratio preservation"""
    print(f"🔄 Using PyTorch fallback on {device} with aspect ratio preservation")
    
    import time
    start_time = time.time()
    
    # Convert to tensor and move to device
    if len(img.shape) == 3:
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    else:
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    # Move to GPU if available
    img_tensor = img_tensor.to(device)
    print(f"💾 Tensor moved to: {img_tensor.device}")
    
    # Get original dimensions
    _, _, h, w = img_tensor.shape
    
    # Calculate exact target dimensions (preserve aspect ratio)
    target_h = h * scale
    target_w = w * scale
    
    print(f"📐 Scaling: {w}x{h} → {target_w}x{target_h} (exactly {scale}x)")
    
    # Use high-quality interpolation on GPU
    upscaled = F.interpolate(
        img_tensor, 
        size=(target_h, target_w),  # Exact target size
        mode='bicubic', 
        align_corners=False
    )
    
    # Move back to CPU for saving
    upscaled = upscaled.cpu()
    
    # Convert back to numpy
    upscaled = upscaled.squeeze().permute(1, 2, 0).numpy()
    upscaled = (upscaled * 255).clip(0, 255).astype(np.uint8)
    
    elapsed = time.time() - start_time
    print(f"⏱️  GPU processing time: {elapsed:.3f}s")
    
    return upscaled

def upscale_image(input_path, output_path, model_name, scale, args):
    """Upscale an image with proper aspect ratio preservation"""
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ Could not read image: {input_path}")
        return False
    
    height, width = img.shape[:2]
    print(f"📥 Input: {width}x{height}")
    
    # Get device
    device, use_gpu = get_device(args)
    use_half = args.fp16 and use_gpu
    
    # Try to use real Real-ESRGAN first
    upsampler = create_real_esrgan_upscaler(model_name, scale, device, use_half)
    
    if upsampler is not None:
        try:
            print("🤖 Using Real-ESRGAN neural network")
            output, _ = upsampler.enhance(img, outscale=scale)
            
            # Verify output dimensions are correct
            out_height, out_width = output.shape[:2]
            expected_width = width * scale
            expected_height = height * scale
            
            print(f"📤 Real-ESRGAN output: {out_width}x{out_height}")
            
            if abs(out_width - expected_width) > 2 or abs(out_height - expected_height) > 2:
                print(f"⚠️ Dimension mismatch! Expected {expected_width}x{expected_height}, got {out_width}x{out_height}")
                print("🔄 Falling back to PyTorch interpolation")
                output = upscale_with_pytorch_fallback(img, scale, device)
            
        except Exception as e:
            print(f"❌ Real-ESRGAN failed: {e}")
            print("🔄 Using PyTorch fallback")
            output = upscale_with_pytorch_fallback(img, scale, device)
    else:
        output = upscale_with_pytorch_fallback(img, scale, device)
    
    # Save result
    success = cv2.imwrite(output_path, output)
    if success and os.path.exists(output_path):
        final_height, final_width = output.shape[:2]
        print(f"✅ Saved: {output_path} ({final_width}x{final_height})")
        
        # Verify aspect ratio preservation
        original_ratio = width / height
        final_ratio = final_width / final_height
        if abs(original_ratio - final_ratio) > 0.01:
            print(f"⚠️ Aspect ratio changed! Original: {original_ratio:.3f}, Final: {final_ratio:.3f}")
        else:
            print(f"✅ Aspect ratio preserved: {original_ratio:.3f}")
        
        return True
    else:
        print(f"❌ Failed to save: {output_path}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input image')
    parser.add_argument('output', help='Output image')
    parser.add_argument('--model', default='RealESRGAN_x4plus',
                       choices=['RealESRGAN_x4plus', 'RealESRGAN_x2plus', 'RealESRGAN_x4plus_anime_6B'])
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--fp16', action='store_true', help='Use half precision')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--gpu', type=int, help='GPU device ID')
    parser.add_argument('--info', action='store_true', help='Show system information')
    
    args = parser.parse_args()
    
    # Show system info if requested
    if args.info:
        device, use_gpu = get_device(args)
        print(f"\n📊 System Information:")
        print(f"  Platform: {platform.system()} {platform.release()}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  MPS Available: {torch.backends.mps.is_available()}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Devices: {torch.cuda.device_count()}")
        return
    
    # Validate model file exists
    model_path = f'models/{args.model}.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    # Process image
    success = upscale_image(args.input, args.output, args.model, args.scale, args)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()