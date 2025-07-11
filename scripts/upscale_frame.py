#!/usr/bin/env python3
"""
Real-ESRGAN Frame Upscaler - Demo Version
Proof of concept for upscaling single video frames with comprehensive validation
This demo version uses high-quality interpolation to demonstrate AI upscaling concepts
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
    import realesrgan
    from PIL import Image
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
        # Processing overhead (much lighter for demo)
        overhead = 1.2
        
        total_memory = (input_memory + output_memory) * overhead
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
    """Demo frame upscaler with comprehensive validation and error handling"""
    
    # Model configurations for demo
    MODEL_CONFIGS = {
        'RealESRGAN_x4plus': {
            'netscale': 4,
            'description': 'Demo General Purpose 4x upscaling (high-quality interpolation)'
        },
        'RealESRGAN_x2plus': {
            'netscale': 2,
            'description': 'Demo General Purpose 2x upscaling (faster processing)'
        },
        'RealESRGAN_x4plus_anime_6B': {
            'netscale': 4,
            'description': 'Demo Anime/cartoon 4x upscaling (optimized for animated content)'
        }
    }
    
    def __init__(self, model_name: str = 'RealESRGAN_x4plus', scale: int = 4, gpu_id: Optional[int] = None):
        """
        Initialize the demo upscaler with validation
        
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
        """Initialize the demo Real-ESRGAN model"""
        try:
            config = self.MODEL_CONFIGS[self.model_name]
            
            print(f"🎯 Initializing Demo {self.model_name}")
            print(f"   Description: {config['description']}")
            print(f"   Scale: {self.scale}x")
            print(f"   Device: {'GPU' if self.gpu_id is not None else 'CPU'}")
            
            # Initialize demo upsampler
            self.upsampler = realesrgan.RealESRGANer(
                scale=config['netscale'],
                gpu_id=self.gpu_id
            )
            
            print(f"✅ Demo model initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize demo model: {e}")
            print("\n🔧 Troubleshooting:")
            print("  • Ensure all dependencies are installed")
            print("  • Run the setup script: ./scripts/setup_upscaling.sh")
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
            
            # Perform demo upscaling
            print("🔄 Demo upscaling in progress...")
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
                print(f"✅ Demo upscaling completed successfully!")
                print(f"   Saved to: {output_path}")
                return True
            else:
                print(f"❌ Output file was not created: {output_path}")
                return False
                
        except Exception as e:
            print(f"❌ Demo upscaling failed: {e}")
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
        description='Real-ESRGAN Frame Upscaler - Demo Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.jpg output.jpg
  %(prog)s input.jpg output.jpg --model RealESRGAN_x2plus --scale 2
  %(prog)s input.jpg output.jpg --model RealESRGAN_x4plus_anime_6B --gpu 0

Available Demo Models:
  RealESRGAN_x4plus         - General purpose 4x (demo)
  RealESRGAN_x2plus         - General purpose 2x (demo)
  RealESRGAN_x4plus_anime_6B - Anime/cartoon 4x (demo)
  
Note: This is a demonstration version using high-quality interpolation
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
        print("📋 Demo System Information:")
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
        print(f"  Demo Real-ESRGAN: Available")
        return
    
    # Validate input arguments
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Initialize demo upscaler
    try:
        upscaler = FrameUpscaler(
            model_name=args.model,
            scale=args.scale,
            gpu_id=args.gpu
        )
    except Exception as e:
        print(f"❌ Failed to initialize demo upscaler: {e}")
        sys.exit(1)
    
    # Show configuration
    info = upscaler.get_info()
    print(f"\n📊 Demo Configuration:")
    print(f"  Model: {info['model_name']}")
    print(f"  Description: {info['description']}")
    print(f"  Scale: {info['scale']}x (native: {info['native_scale']}x)")
    print(f"  Device: {'GPU ' + str(info['gpu_id']) if info['gpu_enabled'] else 'CPU'}")
    print()
    
    # Perform demo upscaling
    success = upscaler.upscale_frame(args.input, args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 