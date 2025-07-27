#!/usr/bin/env python3
"""
RIFE Frame Interpolation Script for MediaEngine
This script provides GPU-accelerated frame interpolation using RIFE models.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any

# Add the project root directory to Python path for RIFE imports
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent  # Go up one level from scripts/ to project root
sys.path.insert(0, str(project_root))

try:
    # Import RIFE modules (assuming RIFE is installed in models/rife/)
    from models.rife.model import Model
    from models.rife.utils import read_img, write_img
except ImportError as e:
    print(f"Error: Failed to import RIFE modules: {e}")
    print("Please ensure RIFE is properly installed in the models/rife/ directory")
    sys.exit(1)

class RIFEInterpolator:
    """RIFE-based frame interpolation engine with GPU acceleration."""
    
    def __init__(self, model_path: str = "models/rife", device: str = "auto", precision: str = "fp16"):
        """
        Initialize RIFE interpolator.
        
        Args:
            model_path: Path to RIFE model files
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            precision: Precision mode ('fp16', 'fp32')
        """
        self.model_path = Path(model_path)
        self.precision = precision
        self.device = self._detect_device(device)
        self.model = None
        self.scale = 1.0
        self.interpolation_times = []
        
        print(f"Initializing RIFE on {self.device} with {precision} precision")
        
    def _detect_device(self, device: str) -> torch.device:
        """Detect optimal device for processing."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def load_model(self, model_version: str = "v4.6") -> bool:
        """Load RIFE model."""
        try:
            self.model = Model()
            self.model.load_model(self.model_path, model_version)
            self.model.eval()
            self.model.device()
            
            # Enable appropriate precision
            if self.precision == "fp16" and self.device.type in ["cuda", "mps"]:
                self.model.half()
                
            print(f"Successfully loaded RIFE model {model_version}")
            return True
            
        except Exception as e:
            print(f"Error loading RIFE model: {e}")
            return False
    
    def interpolate_frame_pair(self, img0: np.ndarray, img1: np.ndarray, timestep: float = 0.5) -> np.ndarray:
        """
        Interpolate between two frames.
        
        Args:
            img0: First frame (numpy array)
            img1: Second frame (numpy array)
            timestep: Interpolation timestep (0.0 to 1.0)
            
        Returns:
            Interpolated frame as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Convert images to tensors
        img0_tensor = torch.from_numpy(img0.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        
        # Move to device
        img0_tensor = img0_tensor.to(self.device)
        img1_tensor = img1_tensor.to(self.device)
        
        # Apply precision
        if self.precision == "fp16" and self.device.type in ["cuda", "mps"]:
            img0_tensor = img0_tensor.half()
            img1_tensor = img1_tensor.half()
        
        # Perform interpolation
        with torch.no_grad():
            timestep_tensor = torch.tensor([timestep]).float().to(self.device)
            if self.precision == "fp16" and self.device.type in ["cuda", "mps"]:
                timestep_tensor = timestep_tensor.half()
                
            output = self.model.inference(img0_tensor, img1_tensor, timestep_tensor, self.scale)
        
        # Convert back to numpy
        output_np = output[0].cpu().float().numpy().transpose(1, 2, 0) * 255.0
        output_np = np.clip(output_np, 0, 255).astype(np.uint8)
        
        # Track performance
        interpolation_time = time.time() - start_time
        self.interpolation_times.append(interpolation_time)
        
        return output_np
    
    def interpolate_sequence(self, input_frames: List[str], output_dir: str, multiplier: int = 2,
                           batch_size: int = 8, progress_callback=None) -> Dict[str, Any]:
        """
        Interpolate a sequence of frames.
        
        Args:
            input_frames: List of input frame file paths
            output_dir: Output directory for interpolated frames
            multiplier: Frame rate multiplier (2x, 3x, 4x)
            batch_size: Number of frame pairs to process in batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with interpolation results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        total_pairs = len(input_frames) - 1
        interpolated_count = 0
        total_interpolated = total_pairs * (multiplier - 1)
        
        start_time = time.time()
        
        # Process frame pairs
        for i in range(total_pairs):
            if progress_callback:
                progress = int((i / total_pairs) * 100)
                progress_callback(progress, f"Processing frame pair {i+1}/{total_pairs}")
            
            # Load frame pair
            img0_path = input_frames[i]
            img1_path = input_frames[i + 1]
            
            img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
            img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
            
            if img0 is None or img1 is None:
                print(f"Warning: Could not load frame pair {i}, {i+1}")
                continue
            
            # Copy original frames to output
            frame0_output = os.path.join(output_dir, f"frame_{i*multiplier:06d}.png")
            cv2.imwrite(frame0_output, img0)
            
            # Generate interpolated frames
            for j in range(1, multiplier):
                timestep = j / multiplier
                
                try:
                    interpolated = self.interpolate_frame_pair(img0, img1, timestep)
                    
                    output_path = os.path.join(output_dir, f"frame_{i*multiplier + j:06d}.png")
                    cv2.imwrite(output_path, interpolated)
                    interpolated_count += 1
                    
                except Exception as e:
                    print(f"Error interpolating frame {i}-{j}: {e}")
                    continue
        
        # Copy final frame
        if input_frames:
            final_img = cv2.imread(input_frames[-1], cv2.IMREAD_COLOR)
            if final_img is not None:
                final_output = os.path.join(output_dir, f"frame_{total_pairs*multiplier:06d}.png")
                cv2.imwrite(final_output, final_img)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_interpolation_time = np.mean(self.interpolation_times) if self.interpolation_times else 0
        fps = len(self.interpolation_times) / total_time if total_time > 0 else 0
        
        result = {
            "success": True,
            "interpolated_frames": interpolated_count,
            "total_output_frames": len(input_frames) + interpolated_count,
            "processing_time": total_time,
            "average_frame_time": avg_interpolation_time,
            "fps": fps,
            "multiplier": multiplier,
            "device": str(self.device),
            "precision": self.precision
        }
        
        if progress_callback:
            progress_callback(100, f"Completed: {interpolated_count} frames interpolated")
        
        return result
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        memory_info = {}
        
        if self.device.type == "cuda" and torch.cuda.is_available():
            memory_info["allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            memory_info["cached_gb"] = torch.cuda.memory_reserved() / 1024**3
            memory_info["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        elif self.device.type == "mps" and torch.backends.mps.is_available():
            # MPS doesn't have detailed memory stats, estimate based on model size
            memory_info["estimated_gb"] = 2.0  # Rough estimate for RIFE model
        else:
            memory_info["cpu_memory_gb"] = 0.0  # CPU memory tracking not implemented
        
        return memory_info

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RIFE Frame Interpolation")
    
    parser.add_argument("--input", required=True, help="Input directory containing frames")
    parser.add_argument("--output", required=True, help="Output directory for interpolated frames")
    parser.add_argument("--model", default="v4.6", help="RIFE model version (v4.6, v4.4, v4.0)")
    parser.add_argument("--multiplier", type=int, default=2, choices=[2, 3, 4], 
                       help="Frame rate multiplier")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for large frames")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16", 
                       help="Precision mode")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                       help="Device to use for processing")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (for CUDA)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU processing")
    parser.add_argument("--output-json", help="Output JSON file for results")
    parser.add_argument("--progress-file", help="File to write progress updates")
    
    return parser.parse_args()

def progress_callback_factory(progress_file: Optional[str]):
    """Create a progress callback that writes to a file."""
    def callback(progress: int, message: str):
        if progress_file:
            try:
                with open(progress_file, "w") as f:
                    json.dump({"progress": progress, "message": message}, f)
            except:
                pass  # Ignore errors in progress reporting
        print(f"Progress: {progress}% - {message}")
    
    return callback

def main():
    """Main function."""
    args = parse_arguments()
    
    # Override device if CPU is forced
    device = "cpu" if args.cpu else args.device
    
    # Set CUDA device if specified
    if args.gpu > 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    try:
        # Initialize interpolator
        interpolator = RIFEInterpolator(device=device, precision=args.precision)
        
        # Load model
        if not interpolator.load_model(args.model):
            print("Failed to load RIFE model")
            sys.exit(1)
        
        # Get input frames
        input_dir = Path(args.input)
        frame_files = sorted(input_dir.glob("frame_*.png"))
        
        if len(frame_files) < 2:
            print(f"Error: Need at least 2 frames, found {len(frame_files)}")
            sys.exit(1)
        
        frame_paths = [str(f) for f in frame_files]
        
        print(f"Found {len(frame_paths)} input frames")
        print(f"Interpolating with {args.multiplier}x multiplier")
        
        # Create progress callback
        progress_callback = progress_callback_factory(args.progress_file)
        
        # Perform interpolation
        result = interpolator.interpolate_sequence(
            frame_paths, args.output, args.multiplier, 
            args.batch_size, progress_callback
        )
        
        # Add memory usage to result
        result["memory_usage"] = interpolator.get_memory_usage()
        
        # Output results
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2)
        
        print("\nInterpolation completed successfully!")
        print(f"Interpolated frames: {result['interpolated_frames']}")
        print(f"Total output frames: {result['total_output_frames']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Average FPS: {result['fps']:.2f}")
        print(f"Device: {result['device']}")
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "device": device,
            "precision": args.precision
        }
        
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(error_result, f, indent=2)
        
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()