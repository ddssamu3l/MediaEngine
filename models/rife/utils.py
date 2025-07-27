"""
Utility functions for RIFE frame interpolation.
"""

import cv2
import numpy as np
import torch
from pathlib import Path


def read_img(path):
    """Read image from file and return as numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def write_img(path, img):
    """Write image to file."""
    return cv2.imwrite(str(path), img)


def img_to_tensor(img):
    """Convert numpy image to tensor."""
    if isinstance(img, np.ndarray):
        # Convert BGR to RGB and normalize to [0, 1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).float() / 255.0
        
        # Add batch dimension and move channels to front
        if len(tensor.shape) == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        return tensor
    return img


def tensor_to_img(tensor):
    """Convert tensor to numpy image."""
    if isinstance(tensor, torch.Tensor):
        # Move to CPU and remove batch dimension if present
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # Move channels to last dimension
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy and scale to [0, 255]
        img = tensor.cpu().float().numpy() * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img_bgr
    
    return tensor


def prepare_frame_pair(frame0_path, frame1_path, device="cpu"):
    """Load and prepare a pair of frames for interpolation."""
    img0 = read_img(frame0_path)
    img1 = read_img(frame1_path)
    
    # Convert to tensors
    tensor0 = img_to_tensor(img0).to(device)
    tensor1 = img_to_tensor(img1).to(device)
    
    return tensor0, tensor1


def save_interpolated_frame(tensor, output_path):
    """Save interpolated frame tensor to file."""
    img = tensor_to_img(tensor)
    return write_img(output_path, img)


def batch_frames_to_tensors(frames, device="cpu"):
    """Convert list of frame arrays to batch tensor."""
    tensors = []
    for frame in frames:
        if isinstance(frame, str) or isinstance(frame, Path):
            frame = read_img(frame)
        tensor = img_to_tensor(frame)
        tensors.append(tensor.squeeze(0))  # Remove batch dimension
    
    # Stack into batch
    batch_tensor = torch.stack(tensors, dim=0).to(device)
    return batch_tensor


def resize_frame(frame, target_size):
    """Resize frame to target size."""
    if isinstance(frame, torch.Tensor):
        # Use PyTorch interpolation
        return torch.nn.functional.interpolate(
            frame, size=target_size, mode='bilinear', align_corners=False
        )
    else:
        # Use OpenCV for numpy arrays
        return cv2.resize(frame, target_size)


def pad_frame_to_multiple(frame, multiple=8):
    """Pad frame dimensions to be multiples of specified value."""
    if isinstance(frame, torch.Tensor):
        _, _, h, w = frame.shape
        new_h = ((h + multiple - 1) // multiple) * multiple
        new_w = ((w + multiple - 1) // multiple) * multiple
        
        if new_h != h or new_w != w:
            pad_h = new_h - h
            pad_w = new_w - w
            frame = torch.nn.functional.pad(
                frame, (0, pad_w, 0, pad_h), mode='reflect'
            )
    else:
        h, w = frame.shape[:2]
        new_h = ((h + multiple - 1) // multiple) * multiple
        new_w = ((w + multiple - 1) // multiple) * multiple
        
        if new_h != h or new_w != w:
            pad_h = new_h - h
            pad_w = new_w - w
            frame = cv2.copyMakeBorder(
                frame, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
            )
    
    return frame


def validate_frame_compatibility(frame0, frame1):
    """Check if two frames are compatible for interpolation."""
    if isinstance(frame0, torch.Tensor) and isinstance(frame1, torch.Tensor):
        if frame0.shape != frame1.shape:
            raise ValueError(f"Frame shapes don't match: {frame0.shape} vs {frame1.shape}")
        return True
    elif isinstance(frame0, np.ndarray) and isinstance(frame1, np.ndarray):
        if frame0.shape != frame1.shape:
            raise ValueError(f"Frame shapes don't match: {frame0.shape} vs {frame1.shape}")
        return True
    else:
        raise ValueError("Frames must be both tensors or both numpy arrays")


def get_frame_info(frame_path):
    """Get basic information about a frame file."""
    frame = read_img(frame_path)
    height, width, channels = frame.shape
    
    return {
        "path": str(frame_path),
        "width": width,
        "height": height,
        "channels": channels,
        "size": (width, height)
    }


def estimate_memory_usage(width, height, batch_size=1, precision="fp32"):
    """Estimate GPU memory usage for frame interpolation."""
    # Base calculation: 2 input frames + 1 output frame + intermediate tensors
    pixels_per_frame = width * height * 3  # 3 channels
    frames_memory = pixels_per_frame * 3  # Input + output frames
    
    # Add overhead for intermediate computations (flows, warped frames, etc.)
    intermediate_overhead = pixels_per_frame * 4  # Rough estimate
    
    total_pixels = frames_memory + intermediate_overhead
    
    # Convert to bytes based on precision
    if precision == "fp16":
        bytes_per_pixel = 2
    else:  # fp32
        bytes_per_pixel = 4
    
    memory_bytes = total_pixels * batch_size * bytes_per_pixel
    memory_gb = memory_bytes / (1024 ** 3)
    
    return memory_gb