#!/bin/bash

# Universal Media Engine Launcher
# This script sets up the Python environment and runs the media engine

echo "🎬 Universal Media Engine"
echo "Setting up environment..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.7+ from https://www.python.org/downloads/"
    echo "macOS: brew install python3"
    echo "Ubuntu: sudo apt install python3 python3-pip"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "realesrgan_env" ]; then
    echo "🔧 Creating Python virtual environment..."
    python3 -m venv realesrgan_env
    
    # Activate virtual environment
    source realesrgan_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    echo "📦 Installing required Python packages..."
    pip install torch torchvision opencv-python numpy
    
    # Try to install Real-ESRGAN (optional)
    echo "🤖 Installing Real-ESRGAN (optional, for best quality)..."
    pip install realesrgan basicsr || echo "⚠️  Real-ESRGAN installation failed, using fallback mode"
    
    echo "✅ Python environment setup complete!"
else
    echo "✅ Python environment already exists"
fi

# Activate virtual environment
source realesrgan_env/bin/activate

# Check if FFmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ FFmpeg is required but not installed."
    echo "Please install FFmpeg:"
    echo "macOS: brew install ffmpeg"
    echo "Ubuntu: sudo apt install ffmpeg"
    echo "Windows: Download from https://ffmpeg.org/download.html"
    exit 1
fi

# Run the media engine
echo "🚀 Starting Universal Media Engine..."
echo ""
./mediaengine