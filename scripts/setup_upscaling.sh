#!/bin/bash
# Real-ESRGAN Setup Script for Universal Media Engine
# Supports cross-platform installation with comprehensive validation

set -e  # Exit on any error

echo "🚀 Setting up AI Upscaling (Real-ESRGAN) for Universal Media Engine..."
echo

# Function to detect Python command
detect_python() {
    local python_cmd=""
    
    if command -v python3 &> /dev/null; then
        python_cmd="python3"
    elif command -v python &> /dev/null; then
        # Verify it's Python 3
        if python -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)" 2>/dev/null; then
            python_cmd="python"
        fi
    elif command -v py &> /dev/null; then
        # Windows Python Launcher
        if py -3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)" 2>/dev/null; then
            python_cmd="py -3"
        fi
    fi
    
    echo "$python_cmd"
}

# Function to get system info
get_system_info() {
    echo "📋 System Information:"
    echo "  OS: $(uname -s 2>/dev/null || echo "Unknown")"
    echo "  Architecture: $(uname -m 2>/dev/null || echo "Unknown")"
    
    # Check available memory
    if command -v free &> /dev/null; then
        local mem_gb=$(free -g | awk '/^Mem:/{print $2}')
        echo "  Available Memory: ${mem_gb}GB"
        if [ "$mem_gb" -lt 8 ]; then
            echo "  ⚠️  Warning: Less than 8GB RAM detected. AI upscaling may be limited."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        local mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
        local mem_gb=$((mem_bytes / 1024 / 1024 / 1024))
        echo "  Available Memory: ${mem_gb}GB"
        if [ "$mem_gb" -lt 8 ]; then
            echo "  ⚠️  Warning: Less than 8GB RAM detected. AI upscaling may be limited."
        fi
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "  GPU: NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1 | while IFS=, read name memory; do
            echo "    Name: $name"
            echo "    VRAM: ${memory}MB"
        done
    else
        echo "  GPU: No NVIDIA GPU detected (CPU-only mode)"
    fi
    echo
}

# Detect Python
PYTHON_CMD=$(detect_python)

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Python 3.7+ is required but not found"
    echo "Please install Python 3.7 or later:"
    echo "  • macOS: brew install python3"
    echo "  • Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  • Windows: Download from python.org"
    exit 1
fi

echo "✅ Found Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Show system information
get_system_info

# Create virtual environment
ENV_NAME="realesrgan_env"
echo "📦 Creating Python virtual environment ($ENV_NAME)..."

if [ -d "$ENV_NAME" ]; then
    echo "⚠️  Environment $ENV_NAME already exists. Removing..."
    rm -rf "$ENV_NAME"
fi

$PYTHON_CMD -m venv "$ENV_NAME"

# Activate environment (cross-platform)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source "$ENV_NAME/Scripts/activate"
else
    source "$ENV_NAME/bin/activate"
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with smart GPU detection
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected - installing CUDA version..."
    
    # Check CUDA version for compatibility
    if nvidia-smi | grep -q "CUDA Version: 12"; then
        echo "  Installing PyTorch with CUDA 12.1 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif nvidia-smi | grep -q "CUDA Version: 11"; then
        echo "  Installing PyTorch with CUDA 11.8 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  Installing PyTorch with latest CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
else
    echo "💻 Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install Real-ESRGAN and dependencies
echo "🎯 Installing Real-ESRGAN and dependencies..."
pip install realesrgan
pip install opencv-python-headless  # Use headless version to avoid GUI dependencies
pip install pillow numpy

# Create scripts directory if it doesn't exist
mkdir -p scripts

# Test installation with comprehensive validation
echo "🧪 Testing installation..."

# Test Python imports
echo "  Testing Python package imports..."
python -c "
import sys
print(f'  Python version: {sys.version}')

import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA devices: {torch.cuda.device_count()}')
    print(f'  Current device: {torch.cuda.get_device_name(0)}')

import cv2
print(f'  OpenCV version: {cv2.__version__}')

from PIL import Image
print(f'  Pillow version: {Image.__version__}')

import realesrgan
print(f'  Real-ESRGAN: Available')

print('✅ All packages imported successfully')
"

# Test Real-ESRGAN functionality if we have a test image
if [ -f "../demo.mp4" ] || [ -f "demo.mp4" ]; then
    echo "  Testing Real-ESRGAN with sample video frame..."
    
    # Extract a test frame using FFmpeg (if available)
    if command -v ffmpeg &> /dev/null; then
        echo "    Extracting test frame..."
        ffmpeg -i ../demo.mp4 -vframes 1 -y test_frame.jpg 2>/dev/null || \
        ffmpeg -i demo.mp4 -vframes 1 -y test_frame.jpg 2>/dev/null || \
        echo "    Could not extract test frame (FFmpeg issue)"
        
        if [ -f "test_frame.jpg" ]; then
            echo "    Testing upscaling..."
            python scripts/upscale_frame.py test_frame.jpg test_upscaled.jpg --model RealESRGAN_x2plus --scale 2 2>/dev/null && \
            echo "    ✅ Real-ESRGAN test successful" || \
            echo "    ⚠️  Real-ESRGAN test failed (models will download on first use)"
            
            # Cleanup test files
            rm -f test_frame.jpg test_upscaled.jpg
        fi
    fi
fi

echo
echo "✅ Real-ESRGAN setup complete!"
echo
echo "📝 Usage Instructions:"
echo "1. Activate the environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   realesrgan_env\\Scripts\\activate"
else
    echo "   source realesrgan_env/bin/activate"
fi
echo "2. Run your media engine with AI upscaling enabled"
echo "3. Models will be automatically downloaded on first use"
echo
echo "💡 Tips:"
echo "  • Use 2x scaling for faster processing"
echo "  • Use 4x scaling for maximum quality"
echo "  • Anime model works best for animated content"
echo "  • Ensure sufficient RAM for large videos (4GB+ recommended)"
echo
echo "🔧 Troubleshooting:"
echo "  • If CUDA errors occur, try CPU mode first"
echo "  • For large videos, use shorter clips or lower resolution"
echo "  • Check that FFmpeg is installed for video processing" 