#!/usr/bin/env python3
"""
Real-ESRGAN Frame Upscaler - Production Version with GPU Support
Uses actual Real-ESRGAN neural networks for true AI upscaling
"""

import argparse
import cv2
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Real Real-ESRGAN imports with compatibility fixes
try:
    # Fix for torchvision compatibility issue
    import torchvision.transforms.functional as F
    if not hasattr(F, 'resize'):
        # Monkey patch for compatibility
        import torchvision.transforms._functional_tensor as F_t
        F.resize = F_t.resize
    
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except ImportError as e:
    print(f"❌ Missing Real-ESRGAN: {e}")
    print("Please install: pip install realesrgan basicsr")
    sys.exit(1)


class RealFrameUpscaler:
    """Production Real-ESRGAN upscaler with GPU support"""
    
    # Model configurations
    MODEL_CONFIGS = {
        'RealESRGAN_x4plus': {
            'model_path': 'models/RealESRGAN_x4plus.pth',
            'netscale': 4,
            'model_arch': 'RRDBNet',
            'num_block': 6,
            'num_grow_ch': 32,
            'scale': 4
        },
        'RealESRGAN_x2plus': {
            'model_path': 'models/RealESRGAN_x2plus.pth',
            'netscale': 2,
            'model_arch': 'RRDBNet',
            'num_block': 6,
            'num_grow_ch': 32,
            'scale': 2
        },
        'RealESRGAN_x4plus_anime_6B': {
            'model_path': 'models/RealESRGAN_x4plus_anime_6B.pth',
            'netscale': 4,
            'model_arch': 'RRDBNet',
            'num_block': 6,
            'num_grow_ch': 32,
            'scale': 4
        }
    }
    
    def __init__(self, model_name: str = 'RealESRGAN_x4plus', scale: int = 4, 
                 use_gpu: bool = True, gpu_id: Optional[int] = None,
                 fp16: bool = False, tile: int = 0, tile_pad: int = 10):
        """
        Initialize Real-ESRGAN with proper GPU support
        
        Args:
            model_name: Model to use
            scale: Output scale
            use_gpu: Whether to use GPU
            gpu_id: GPU device ID (None for auto-select)
            fp16: Use half precision (faster, less memory)
            tile: Tile size for processing large images (0 = auto)
            tile_pad: Padding for tiles
        """
        self.model_name = model_name
        self.scale = scale
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.fp16 = fp16
        self.tile = tile
        self.tile_pad = tile_pad
        
        # Detect device
        self.device = self._get_device()
        self.upsampler = None
        self._initialize_model()
    
    def _get_device(self):
        """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)"""
        if not self.use_gpu:
            return 'cpu'
        
        # Check for Apple Silicon MPS
        if torch.backends.mps.is_available():
            print("🍎 Using Apple Silicon GPU (MPS)")
            return 'mps'
        
        # Check for CUDA
        if torch.cuda.is_available():
            if self.gpu_id is not None:
                print(f"🎮 Using CUDA GPU {self.gpu_id}")
                return f'cuda:{self.gpu_id}'
            else:
                print("🎮 Using CUDA GPU 0")
                return 'cuda:0'
        
        print("💻 Using CPU (no GPU available)")
        return 'cpu'
    
    def _initialize_model(self):
        """Initialize the real Real-ESRGAN model with proper architecture"""
        try:
            config = self.MODEL_CONFIGS[self.model_name]
            
            # Check if model file exists
            model_path = Path(config['model_path'])
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    f"Please run: python scripts/install_real_esrgan.py"
                )
            
            print(f"🚀 Loading Real-ESRGAN {self.model_name}")
            print(f"   Model: {model_path}")
            print(f"   Device: {self.device}")
            print(f"   FP16: {self.fp16 and self.device != 'cpu'}")
            
            # Create model architecture
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=config['num_block'],
                num_grow_ch=config['num_grow_ch'],
                scale=config['scale']
            )
            
            # Initialize upsampler
            self.upsampler = RealESRGANer(
                scale=config['netscale'],
                model_path=str(model_path),
                model=model,
                tile=self.tile,
                tile_pad=self.tile_pad,
                pre_pad=0,
                half=self.fp16 and self.device != 'cpu',
                device=self.device
            )
            
            print("✅ Real-ESRGAN model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            raise
    
    def upscale_frame(self, input_path: str, output_path: str, 
                      denoise_strength: float = 0.5) -> bool:
        """
        Upscale a frame using real AI upscaling
        
        Args:
            input_path: Input image path
            output_path: Output image path
            denoise_strength: Denoising strength (0-1)
            
        Returns:
            bool: Success status
        """
        try:
            # Read image
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"❌ Could not read image: {input_path}")
                return False
            
            height, width = img.shape[:2]
            print(f"📥 Input: {width}x{height}")
            
            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Perform real AI upscaling
            print("🤖 Performing AI upscaling...")
            
            # For large images, use tiling to manage memory
            if self.tile == 0:  # Auto tile size
                # Estimate based on resolution and device
                if self.device == 'cpu':
                    auto_tile = 400
                elif self.device == 'mps':
                    auto_tile = 800 if not self.fp16 else 1024
                else:  # CUDA
                    auto_tile = 1024 if not self.fp16 else 2048
                
                if width > auto_tile or height > auto_tile:
                    self.upsampler.tile = auto_tile
                    print(f"   Using tile size: {auto_tile} (auto)")
            
            # Enhance with Real-ESRGAN
            output, _ = self.upsampler.enhance(
                img, 
                outscale=self.scale,
                alpha_upsampler='realesrgan'
            )
            
            # Save result
            success = cv2.imwrite(output_path, output)
            if not success:
                print(f"❌ Failed to save: {output_path}")
                return False
            
            # Verify and report
            if os.path.exists(output_path):
                out_height, out_width = output.shape[:2]
                out_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"📤 Output: {out_width}x{out_height} ({out_size:.1f}MB)")
                print(f"✅ AI upscaling completed!")
                return True
            
            return False
            
        except torch.cuda.OutOfMemoryError:
            print("❌ GPU out of memory! Try:")
            print("   • Using smaller tile size (--tile 512)")
            print("   • Enabling FP16 mode (--fp16)")
            print("   • Using CPU mode (--cpu)")
            return False
        except Exception as e:
            print(f"❌ Upscaling failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def benchmark(self, input_path: str) -> Dict[str, Any]:
        """Benchmark upscaling performance"""
        import time
        
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not read image"}
        
        height, width = img.shape[:2]
        
        # Warm-up run
        print("🔥 Warming up...")
        _ = self.upsampler.enhance(img, outscale=self.scale)
        
        # Benchmark runs
        print("⏱️ Benchmarking (5 runs)...")
        times = []
        for i in range(5):
            start = time.time()
            _ = self.upsampler.enhance(img, outscale=self.scale)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"   Run {i+1}: {elapsed:.2f}s")
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        return {
            "input_size": f"{width}x{height}",
            "output_size": f"{width*self.scale}x{height*self.scale}",
            "device": str(self.device),
            "model": self.model_name,
            "scale": self.scale,
            "fp16": self.fp16,
            "tile_size": self.upsampler.tile,
            "avg_time": f"{avg_time:.2f}s",
            "fps": f"{fps:.2f}",
            "times": times
        }


def main():
    parser = argparse.ArgumentParser(
        description='Real-ESRGAN Frame Upscaler - Production Version'
    )
    
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--model', default='RealESRGAN_x4plus',
                       choices=list(RealFrameUpscaler.MODEL_CONFIGS.keys()),
                       help='Model to use')
    parser.add_argument('--scale', type=int, default=4,
                       help='Output scale (must match model scale)')
    parser.add_argument('--denoise', type=float, default=0.5,
                       help='Denoise strength (0-1)')
    parser.add_argument('--tile', type=int, default=0,
                       help='Tile size for large images (0=auto)')
    parser.add_argument('--tile-pad', type=int, default=10,
                       help='Tile padding')
    parser.add_argument('--fp16', action='store_true',
                       help='Use half precision (faster, less memory)')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--gpu', type=int, metavar='ID',
                       help='GPU device ID')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Initialize upscaler
    upscaler = RealFrameUpscaler(
        model_name=args.model,
        scale=args.scale,
        use_gpu=not args.cpu,
        gpu_id=args.gpu,
        fp16=args.fp16,
        tile=args.tile,
        tile_pad=args.tile_pad
    )
    
    if args.benchmark:
        print("\n📊 Running benchmark...")
        results = upscaler.benchmark(args.input)
        print("\n📈 Benchmark Results:")
        for key, value in results.items():
            print(f"   {key}: {value}")
    else:
        success = upscaler.upscale_frame(
            args.input, 
            args.output,
            denoise_strength=args.denoise
        )
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()