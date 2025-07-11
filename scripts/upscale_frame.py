#!/usr/bin/env python3
"""
Real-ESRGAN Frame Upscaler - Enhanced Production Version
Proof of concept for upscaling single video frames with comprehensive validation
"""

import argparse
import cv2
import os
import sys
import tempfile
import psutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import torch
    import numpy as np
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please run the setup script: ./scripts/setup_upscaling.sh")
    sys.exit(1)


class SystemValidator:
    """Validates system requirements and estimates resource usage"""
    
    @staticmethod
    def check_available_memory() -> int:
        """Returns available memory in GB"""
        return psutil.virtual_memory().available // (1024**3)
    
    @staticmethod
    def estimate_memory_usage(width: int, height: int, scale: int, dtype_size: int = 4) -> float:
        """Estimates memory usage in GB for upscaling"""
        # Input image memory
        input_memory = width * height * 3 * dtype_size
        # Output image memory  
        output_memory = (width * scale) * (height * scale) * 3 * dtype_size
        # Model memory (approximate)
        model_memory = 2 * (1024**3)  # ~2GB for model
        # Buffer and processing overhead
        overhead = 1.5
        
        total_memory = (input_memory + output_memory + model_memory) * overhead
        return total_memory / (1024**3)  # Convert to GB
    
    @staticmethod
    def validate_dimensions(width: int, height: int, scale: int) -> Tuple[bool, str]:
        """Validates if dimensions are reasonable for upscaling"""
        # Check maximum output dimensions
        max_dimension = 8192  # 8K maximum
        if width * scale > max_dimension or height * scale > max_dimension:
            return False, f"Output dimensions ({width*scale}x{height*scale}) exceed maximum supported size ({max_dimension}x{max_dimension})"
        
        # Check minimum dimensions
        if width < 64 or height < 64:
            return False, f"Input dimensions ({width}x{height}) are too small (minimum 64x64)"
        
        # Estimate memory usage
        estimated_memory = SystemValidator.estimate_memory_usage(width, height, scale)
        available_memory = SystemValidator.check_available_memory()
        
        if estimated_memory > available_memory * 0.8:  # Use max 80% of available memory
            return False, f"Estimated memory usage ({estimated_memory:.1f}GB) exceeds available memory ({available_memory}GB)"
        
        return True, ""


class FrameUpscaler:
    """Enhanced frame upscaler with comprehensive validation and error handling"""
    
    # Model configurations with validation
    MODEL_CONFIGS = {
        'RealESRGAN_x4plus': {
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'netscale': 4,
            'num_block': 23,
            'num_feat': 64,
            'description': 'General purpose 4x upscaling (best for photos/real content)'
        },
        'RealESRGAN_x2plus': {
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRGAN_x2plus.pth',
            'netscale': 2,
            'num_block': 23,
            'num_feat': 64,
            'description': 'General purpose 2x upscaling (faster, good quality)'
        },
        'RealESRGAN_x4plus_anime_6B': {
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
            'netscale': 4,
            'num_block': 6,
            'num_feat': 64,
            'description': 'Anime/cartoon 4x upscaling (optimized for animated content)'
        }
    }
    
    def __init__(self, model_name: str = 'RealESRGAN_x4plus', scale: int = 4, gpu_id: Optional[int] = None):
        """
        Initialize the upscaler with validation
        
        Args:
            model_name: Model to use
            scale: Upscaling factor (2, 3, or 4)
            gpu_id: GPU ID to use (None for CPU)
        """
        self.model_name = model_name
        self.scale = scale
        self.gpu_id = gpu_id
        self.upsampler = None
        self._validate_configuration()
        self._initialize_model()
    
    def _validate_configuration(self):
        """Validate configuration parameters"""
        if self.model_name not in self.MODEL_CONFIGS:
            available_models = list(self.MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown model: {self.model_name}. Available models: {available_models}")
        
        if self.scale not in [2, 3, 4]:
            raise ValueError(f"Invalid scale: {self.scale}. Must be 2, 3, or 4")
        
        # Warn if scale doesn't match model's native scale
        model_config = self.MODEL_CONFIGS[self.model_name]
        if self.scale != model_config['netscale']:
            print(f"⚠️  Warning: Scale {self.scale} differs from model's native scale {model_config['netscale']}")
    
    def _initialize_model(self):
        """Initialize the Real-ESRGAN model with comprehensive error handling"""
        try:
            config = self.MODEL_CONFIGS[self.model_name]
            
            print(f"🎯 Initializing {self.model_name}")
            print(f"   Description: {config['description']}")
            print(f"   Scale: {self.scale}x")
            print(f"   Device: {'GPU' if self.gpu_id is not None else 'CPU'}")
            
            # Check GPU availability if requested
            if self.gpu_id is not None:
                if not torch.cuda.is_available():
                    print("⚠️  CUDA not available, falling back to CPU")
                    self.gpu_id = None
                elif self.gpu_id >= torch.cuda.device_count():
                    print(f"⚠️  GPU {self.gpu_id} not available, using GPU 0")
                    self.gpu_id = 0
            
            # Create model architecture
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=config['num_feat'],
                num_block=config['num_block'],
                num_grow_ch=32,
                scale=config['netscale']
            )
            
            # Initialize upsampler with optimized settings
            tile_size = 400 if self.gpu_id is not None else 200  # Smaller tiles for CPU
            
            self.upsampler = RealESRGANer(
                scale=config['netscale'],
                model_path=config['model_path'],
                model=model,
                tile=tile_size,
                tile_pad=10,
                pre_pad=0,
                half=self.gpu_id is not None,  # Use half precision for GPU
                gpu_id=self.gpu_id
            )
            
            print(f"✅ Model initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            print("\n🔧 Troubleshooting:")
            print("  • Ensure all dependencies are installed")
            print("  • Check internet connection for model download")
            print("  • Try CPU mode if GPU initialization fails")
            print("  • Verify CUDA installation if using GPU")
            sys.exit(1)
    
    def validate_input_image(self, input_path: str) -> Tuple[bool, str, Optional[Tuple[int, int]]]:
        """Validate input image and return dimensions"""
        try:
            # Check file existence
            if not os.path.exists(input_path):
                return False, f"Input file not found: {input_path}", None
            
            # Check file size (max 100MB for safety)
            file_size = os.path.getsize(input_path) / (1024 * 1024)
            if file_size > 100:
                return False, f"Input file too large: {file_size:.1f}MB (max 100MB)", None
            
            # Try to read image
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                return False, f"Could not read image (unsupported format or corrupted): {input_path}", None
            
            height, width = img.shape[:2]
            
            # Validate dimensions for upscaling
            is_valid, error_msg = SystemValidator.validate_dimensions(width, height, self.scale)
            if not is_valid:
                return False, error_msg, (width, height)
            
            return True, "", (width, height)
            
        except Exception as e:
            return False, f"Error validating input: {e}", None
    
    def upscale_frame(self, input_path: str, output_path: str) -> bool:
        """
        Upscale a single frame with comprehensive validation
        
        Args:
            input_path: Path to input image
            output_path: Path to save upscaled image
            
        Returns:
            bool: Success status
        """
        try:
            # Validate input
            is_valid, error_msg, dimensions = self.validate_input_image(input_path)
            if not is_valid:
                print(f"❌ {error_msg}")
                return False
            
            width, height = dimensions
            print(f"📥 Input: {width}x{height} pixels ({os.path.getsize(input_path)/(1024*1024):.1f}MB)")
            
            # Estimate and display resource usage
            estimated_memory = SystemValidator.estimate_memory_usage(width, height, self.scale)
            available_memory = SystemValidator.check_available_memory()
            print(f"💾 Estimated memory usage: {estimated_memory:.1f}GB (Available: {available_memory}GB)")
            
            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"📁 Created output directory: {output_dir}")
            
            # Read image
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            
            # Perform upscaling with progress indication
            print("🔄 Upscaling in progress...")
            output, _ = self.upsampler.enhance(img, outscale=self.scale)
            
            # Validate output dimensions
            expected_width, expected_height = width * self.scale, height * self.scale
            actual_height, actual_width = output.shape[:2]
            
            if actual_width != expected_width or actual_height != expected_height:
                print(f"⚠️  Warning: Output dimensions ({actual_width}x{actual_height}) don't match expected ({expected_width}x{expected_height})")
            
            # Save result with error handling
            success = cv2.imwrite(output_path, output)
            if not success:
                print(f"❌ Failed to save image to: {output_path}")
                return False
            
            # Verify output file was created and get stats
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"📤 Output: {actual_width}x{actual_height} pixels ({output_size:.1f}MB)")
                print(f"✅ Upscaling completed successfully!")
                print(f"   Saved to: {output_path}")
                return True
            else:
                print(f"❌ Output file was not created: {output_path}")
                return False
                
        except Exception as e:
            print(f"❌ Upscaling failed: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        config = self.MODEL_CONFIGS[self.model_name]
        return {
            'model_name': self.model_name,
            'description': config['description'],
            'scale': self.scale,
            'native_scale': config['netscale'],
            'gpu_enabled': self.gpu_id is not None,
            'gpu_id': self.gpu_id,
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }


def main():
    """Main function with comprehensive argument parsing and validation"""
    parser = argparse.ArgumentParser(
        description='Real-ESRGAN Frame Upscaler - Enhanced Production Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.jpg output.jpg
  %(prog)s input.jpg output.jpg --model RealESRGAN_x2plus --scale 2
  %(prog)s input.jpg output.jpg --model RealESRGAN_x4plus_anime_6B --gpu 0

Available Models:
  RealESRGAN_x4plus         - General purpose 4x (best for photos)
  RealESRGAN_x2plus         - General purpose 2x (faster)
  RealESRGAN_x4plus_anime_6B - Anime/cartoon 4x (for animated content)
        """)
    
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--model', default='RealESRGAN_x4plus',
                       choices=list(FrameUpscaler.MODEL_CONFIGS.keys()),
                       help='Model to use (default: RealESRGAN_x4plus)')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 3, 4],
                       help='Upscaling factor (default: 4)')
    parser.add_argument('--gpu', type=int, metavar='ID',
                       help='GPU device ID to use (default: CPU)')
    parser.add_argument('--info', action='store_true',
                       help='Show system information and exit')
    
    args = parser.parse_args()
    
    # Show system information if requested
    if args.info:
        print("📋 System Information:")
        print(f"  Python: {sys.version}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name} ({props.total_memory // 1024**2}MB)")
        print(f"  Available RAM: {SystemValidator.check_available_memory()}GB")
        print(f"  CPU cores: {psutil.cpu_count()}")
        return
    
    # Validate input arguments
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Initialize upscaler
    try:
        upscaler = FrameUpscaler(
            model_name=args.model,
            scale=args.scale,
            gpu_id=args.gpu
        )
    except Exception as e:
        print(f"❌ Failed to initialize upscaler: {e}")
        sys.exit(1)
    
    # Show configuration
    info = upscaler.get_info()
    print(f"\n📊 Configuration:")
    print(f"  Model: {info['model_name']}")
    print(f"  Description: {info['description']}")
    print(f"  Scale: {info['scale']}x (native: {info['native_scale']}x)")
    print(f"  Device: {'GPU ' + str(info['gpu_id']) if info['gpu_enabled'] else 'CPU'}")
    print()
    
    # Perform upscaling
    success = upscaler.upscale_frame(args.input, args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 