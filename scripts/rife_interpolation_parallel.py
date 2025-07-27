#!/usr/bin/env python3
"""
High-Performance Parallel RIFE Frame Interpolation Script for MediaEngine
Implements aggressive GPU utilization with parallel batch processing and async CPU/GPU overlap.
"""

import argparse
import os
import sys
import time
import json
import asyncio
import threading
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import gc

# Add the project root directory to Python path for RIFE imports
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent  # Go up one level from scripts/ to project root
sys.path.insert(0, str(project_root))

try:
    # Import RIFE modules
    from models.rife.model import Model
    from models.rife.utils import read_img, write_img
except ImportError as e:
    print(f"Error: Failed to import RIFE modules: {e}")
    print("Please ensure RIFE is properly installed in the models/rife/ directory")
    sys.exit(1)

class FramePairDataset(Dataset):
    """Dataset for loading frame pairs efficiently."""
    
    def __init__(self, frame_pairs: List[Tuple[str, str]], multiplier: int):
        self.frame_pairs = frame_pairs
        self.multiplier = multiplier
        self.total_interpolations = len(frame_pairs) * (multiplier - 1)
        
    def __len__(self):
        return self.total_interpolations
    
    def __getitem__(self, idx):
        pair_idx = idx // (self.multiplier - 1)
        timestep_idx = idx % (self.multiplier - 1) + 1
        timestep = timestep_idx / self.multiplier
        
        frame0_path, frame1_path = self.frame_pairs[pair_idx]
        
        return {
            'frame0_path': frame0_path,
            'frame1_path': frame1_path,
            'timestep': timestep,
            'output_idx': idx,
            'pair_idx': pair_idx,
            'timestep_idx': timestep_idx
        }

class AsyncFrameLoader:
    """Asynchronous frame loader with prefetching and caching."""
    
    def __init__(self, cache_size: int = 64):
        self.cache = {}
        self.cache_size = cache_size
        self.lock = threading.Lock()
        self.load_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _load_frame(self, path: str) -> np.ndarray:
        """Load a single frame from disk."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load frame: {path}")
        return img
    
    def preload_frames(self, paths: List[str]):
        """Preload frames asynchronously."""
        futures = []
        for path in paths:
            if path not in self.cache:
                future = self.executor.submit(self._load_frame, path)
                futures.append((path, future))
        
        for path, future in futures:
            try:
                img = future.result()
                with self.lock:
                    if len(self.cache) >= self.cache_size:
                        # Remove oldest entry
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                    self.cache[path] = img
            except Exception as e:
                print(f"Warning: Failed to preload {path}: {e}")
    
    def get_frame(self, path: str) -> np.ndarray:
        """Get frame with caching."""
        with self.lock:
            if path in self.cache:
                return self.cache[path].copy()
        
        # Load synchronously if not cached
        return self._load_frame(path)
    
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        with self.lock:
            self.cache.clear()

class ParallelRIFEInterpolator:
    """High-performance parallel RIFE interpolator with maximum GPU utilization."""
    
    def __init__(self, model_path: str = "models/rife", device: str = "auto", 
                 precision: str = "fp16", max_batch_size: int = 32):
        """
        Initialize parallel RIFE interpolator.
        
        Args:
            model_path: Path to RIFE model files
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            precision: Precision mode ('fp16', 'fp32')
            max_batch_size: Maximum batch size for GPU processing
        """
        self.model_path = Path(model_path)
        self.precision = precision
        self.device = self._detect_device(device)
        self.max_batch_size = max_batch_size
        self.model = None
        self.frame_loader = AsyncFrameLoader(cache_size=128)
        
        # Performance tracking
        self.total_interpolations = 0
        self.processing_times = []
        self.gpu_utilization_samples = []
        
        # Async processing
        self.input_queue = queue.Queue(maxsize=256)
        self.output_queue = queue.Queue(maxsize=256)
        self.processing_threads = []
        
        print(f"Initializing Parallel RIFE on {self.device} with {precision} precision")
        print(f"Maximum batch size: {max_batch_size}")
        
    def _detect_device(self, device: str) -> torch.device:
        """Detect optimal device for processing."""
        if device == "auto":
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name()
                print(f"CUDA GPU detected: {device_name}")
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                print("Apple Silicon MPS detected")
                return torch.device("mps")
            else:
                print("Using CPU (GPU not available)")
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _optimize_batch_size(self, frame_resolution=None) -> int:
        """Dynamically optimize batch size based on GPU memory and resolution."""
        if self.device.type == "cpu":
            return min(4, self.max_batch_size)
        
        try:
            if self.device.type == "cuda":
                # Get GPU memory info
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU Memory: {gpu_memory:.1f}GB")
                
                # Aggressive batch sizing based on GPU memory
                if gpu_memory >= 24:  # High-end GPUs (RTX 4090, A100, etc.)
                    optimal_batch = min(64, self.max_batch_size)
                elif gpu_memory >= 16:  # High-end GPUs (RTX 4080, 3090, etc.)
                    optimal_batch = min(48, self.max_batch_size)
                elif gpu_memory >= 12:  # Mid-high GPUs (RTX 4070 Ti, 3080, etc.)
                    optimal_batch = min(32, self.max_batch_size)
                elif gpu_memory >= 8:   # Mid-range GPUs (RTX 4060 Ti, 3070, etc.)
                    optimal_batch = min(24, self.max_batch_size)
                elif gpu_memory >= 6:   # Entry GPUs (RTX 4060, 3060, etc.)
                    optimal_batch = min(16, self.max_batch_size)
                else:
                    optimal_batch = min(8, self.max_batch_size)
                    
            elif self.device.type == "mps":
                # Apple Silicon - be more conservative with high resolution
                optimal_batch = min(16, self.max_batch_size)  # Reduced from 32
                
                # Further reduce for high resolution content
                if frame_resolution and frame_resolution[0] * frame_resolution[1] > 1920 * 1080:
                    optimal_batch = min(4, optimal_batch)  # Very conservative for 2K/4K
                    print(f"High resolution detected ({frame_resolution[0]}x{frame_resolution[1]}), using conservative batch size")
                
            else:
                optimal_batch = min(4, self.max_batch_size)
                
            print(f"Optimized batch size: {optimal_batch}")
            return optimal_batch
            
        except Exception as e:
            print(f"Warning: Failed to optimize batch size: {e}")
            return min(4, self.max_batch_size)  # More conservative fallback
    
    def load_model(self, model_version: str = "v4.6") -> bool:
        """Load RIFE model with optimizations."""
        try:
            self.model = Model()
            self.model.load_model(self.model_path, model_version)
            self.model.eval()
            self.model.to(self.device)
            
            # Enable appropriate precision and optimizations
            if self.precision == "fp16" and self.device.type in ["cuda", "mps"]:
                self.model.half()
            
            # Compile model for better performance (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.device.type == "cuda":
                try:
                    self.model = torch.compile(self.model, mode="max-autotune")
                    print("Model compiled with max-autotune for optimal performance")
                except Exception as e:
                    print(f"Model compilation failed (using uncompiled): {e}")
            
            # Optimize memory allocation
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = True
                print("CUDA optimizations enabled")
            
            print(f"Successfully loaded RIFE model {model_version}")
            return True
            
        except Exception as e:
            print(f"Error loading RIFE model: {e}")
            return False
    
    def interpolate_batch(self, batch_data: List[Dict]) -> List[np.ndarray]:
        """Interpolate a batch of frame pairs simultaneously."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        batch_size = len(batch_data)
        if batch_size == 0:
            return []
        
        start_time = time.time()
        
        # Load all frames in the batch
        frames0 = []
        frames1 = []
        timesteps = []
        
        for data in batch_data:
            frame0 = self.frame_loader.get_frame(data['frame0_path'])
            frame1 = self.frame_loader.get_frame(data['frame1_path'])
            
            frames0.append(frame0)
            frames1.append(frame1)
            timesteps.append(data['timestep'])
        
        # Convert to tensors
        batch_img0 = self._frames_to_batch_tensor(frames0)
        batch_img1 = self._frames_to_batch_tensor(frames1)
        batch_timesteps = torch.tensor(timesteps, dtype=torch.float32, device=self.device)
        
        if self.precision == "fp16" and self.device.type in ["cuda", "mps"]:
            batch_img0 = batch_img0.half()
            batch_img1 = batch_img1.half()
            batch_timesteps = batch_timesteps.half()
        
        # Perform batch interpolation
        with torch.no_grad():
            batch_output = self.model.inference(batch_img0, batch_img1, batch_timesteps, scale=1.0)
        
        # Convert back to numpy arrays
        results = []
        for i in range(batch_size):
            output_np = batch_output[i].cpu().float().numpy().transpose(1, 2, 0) * 255.0
            output_np = np.clip(output_np, 0, 255).astype(np.uint8)
            results.append(output_np)
        
        # Track performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return results
    
    def _frames_to_batch_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Convert list of frames to batch tensor."""
        batch_array = np.stack([frame.transpose(2, 0, 1) for frame in frames])
        batch_tensor = torch.from_numpy(batch_array).float() / 255.0
        return batch_tensor.to(self.device)
    
    async def async_interpolate_sequence(self, input_frames: List[str], output_dir: str, 
                                       multiplier: int = 2, progress_callback=None) -> Dict[str, Any]:
        """
        Asynchronously interpolate a sequence with maximum GPU utilization.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get frame resolution for optimization
        first_frame = cv2.imread(input_frames[0])
        frame_resolution = (first_frame.shape[1], first_frame.shape[0]) if first_frame is not None else None
        
        # Optimize batch size for this GPU and resolution
        optimal_batch_size = self._optimize_batch_size(frame_resolution)
        
        # Create frame pairs
        frame_pairs = [(input_frames[i], input_frames[i + 1]) 
                      for i in range(len(input_frames) - 1)]
        
        # Preload frames asynchronously
        all_frame_paths = list(set(input_frames))  # Remove duplicates
        self.frame_loader.preload_frames(all_frame_paths[:optimal_batch_size * 2])
        
        # Create dataset and dataloader for efficient batching
        dataset = FramePairDataset(frame_pairs, multiplier)
        total_interpolations = len(dataset)
        
        print(f"Processing {total_interpolations} interpolations in batches of {optimal_batch_size}")
        
        start_time = time.time()
        interpolated_count = 0
        
        # Process in parallel batches
        batch_futures = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit batches for processing
            for i in range(0, total_interpolations, optimal_batch_size):
                batch_end = min(i + optimal_batch_size, total_interpolations)
                batch_data = [dataset[j] for j in range(i, batch_end)]
                
                # Preload frames for next batch
                if batch_end < total_interpolations:
                    next_batch_end = min(batch_end + optimal_batch_size, total_interpolations)
                    next_batch_paths = []
                    for j in range(batch_end, next_batch_end):
                        data = dataset[j]
                        next_batch_paths.extend([data['frame0_path'], data['frame1_path']])
                    self.frame_loader.preload_frames(next_batch_paths)
                
                # Submit batch for processing
                future = executor.submit(self._process_batch_with_saving, batch_data, output_dir)
                batch_futures.append((future, len(batch_data)))
            
            # Collect results with progress updates
            for future, batch_size in batch_futures:
                try:
                    batch_results = await asyncio.get_event_loop().run_in_executor(None, future.result)
                    interpolated_count += batch_size
                    
                    if progress_callback:
                        progress = int((interpolated_count / total_interpolations) * 100)
                        progress_callback(progress, f"Processed {interpolated_count}/{total_interpolations} interpolations")
                    
                    # Force garbage collection periodically
                    if interpolated_count % (optimal_batch_size * 4) == 0:
                        gc.collect()
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    print(f"Batch processing error: {e}")
                    continue
        
        # Copy original frames to output
        await self._copy_original_frames(input_frames, output_dir, multiplier)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        fps = interpolated_count / total_time if total_time > 0 else 0
        
        result = {
            "success": True,
            "interpolated_frames": interpolated_count,
            "total_output_frames": len(input_frames) + interpolated_count,
            "processing_time": total_time,
            "average_batch_time": avg_processing_time,
            "effective_fps": fps,
            "multiplier": multiplier,
            "device": str(self.device),
            "precision": self.precision,
            "batch_size": optimal_batch_size,
            "gpu_utilization": self._estimate_gpu_utilization()
        }
        
        # Cleanup
        self.frame_loader.cleanup()
        
        if progress_callback:
            progress_callback(100, f"Completed: {interpolated_count} frames interpolated at {fps:.1f} FPS")
        
        return result
    
    def _process_batch_with_saving(self, batch_data: List[Dict], output_dir: str) -> List[str]:
        """Process a batch and save results."""
        results = self.interpolate_batch(batch_data)
        saved_files = []
        
        for i, (data, interpolated_frame) in enumerate(zip(batch_data, results)):
            output_path = os.path.join(output_dir, 
                f"frame_{data['pair_idx']:06d}_{data['timestep_idx']:02d}.png")
            
            cv2.imwrite(output_path, interpolated_frame)
            saved_files.append(output_path)
        
        return saved_files
    
    async def _copy_original_frames(self, input_frames: List[str], output_dir: str, multiplier: int):
        """Copy original frames to output directory."""
        def copy_frame(i, frame_path):
            frame = cv2.imread(frame_path)
            if frame is not None:
                output_path = os.path.join(output_dir, f"frame_{i*multiplier:06d}_00.png")
                cv2.imwrite(output_path, frame)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(copy_frame, i, frame_path) 
                      for i, frame_path in enumerate(input_frames)]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error copying original frame: {e}")
    
    def _estimate_gpu_utilization(self) -> float:
        """Estimate GPU utilization based on processing patterns."""
        if self.device.type == "cuda" and self.processing_times:
            # Rough estimation based on processing speed
            avg_time = np.mean(self.processing_times)
            batch_size = self._optimize_batch_size()  # No resolution needed for estimation
            
            # Assume optimal utilization if processing fast with large batches
            if avg_time < 0.1 and batch_size >= 16:
                return 95.0  # High utilization
            elif avg_time < 0.5 and batch_size >= 8:
                return 85.0  # Good utilization
            else:
                return 70.0  # Moderate utilization
        return 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        memory_info = {}
        
        if self.device.type == "cuda" and torch.cuda.is_available():
            memory_info["allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            memory_info["cached_gb"] = torch.cuda.memory_reserved() / 1024**3
            memory_info["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            memory_info["utilization_estimate"] = self._estimate_gpu_utilization()
        elif self.device.type == "mps":
            memory_info["estimated_gb"] = 3.0  # Rough estimate for RIFE + batching
            memory_info["utilization_estimate"] = 80.0 if len(self.processing_times) > 0 else 0.0
        else:
            memory_info["cpu_memory_gb"] = 0.0
            memory_info["utilization_estimate"] = 0.0
        
        return memory_info

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parallel RIFE Frame Interpolation")
    
    parser.add_argument("--input", required=True, help="Input directory containing frames")
    parser.add_argument("--output", required=True, help="Output directory for interpolated frames")
    parser.add_argument("--model", default="v4.6", help="RIFE model version (v4.6, v4.4, v4.0)")
    parser.add_argument("--multiplier", type=int, default=2, choices=[2, 3, 4], 
                       help="Frame rate multiplier")
    parser.add_argument("--max-batch-size", type=int, default=64, 
                       help="Maximum batch size (will be optimized based on GPU)")
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

async def main_async():
    """Main async function."""
    args = parse_arguments()
    
    # Override device if CPU is forced
    device = "cpu" if args.cpu else args.device
    
    # Set CUDA device if specified
    if args.gpu > 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    try:
        # Initialize parallel interpolator
        interpolator = ParallelRIFEInterpolator(
            device=device, 
            precision=args.precision,
            max_batch_size=args.max_batch_size
        )
        
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
        print(f"Interpolating with {args.multiplier}x multiplier using parallel processing")
        
        # Create progress callback
        progress_callback = progress_callback_factory(args.progress_file)
        
        # Perform parallel interpolation
        result = await interpolator.async_interpolate_sequence(
            frame_paths, args.output, args.multiplier, progress_callback
        )
        
        # Add memory usage to result
        result["memory_usage"] = interpolator.get_memory_usage()
        
        # Output results
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2)
        
        print("\nParallel interpolation completed successfully!")
        print(f"Interpolated frames: {result['interpolated_frames']}")
        print(f"Total output frames: {result['total_output_frames']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Effective FPS: {result['effective_fps']:.2f}")
        print(f"Batch size used: {result['batch_size']}")
        print(f"GPU utilization estimate: {result['gpu_utilization']:.1f}%")
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

def main():
    """Main function with async support."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()