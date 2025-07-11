#!/bin/bash
# Setup script for Real-ESRGAN with actual AI models

set -e

echo "🚀 Setting up Real-ESRGAN with actual AI models..."
echo

# Check if virtual environment exists
if [ ! -d "realesrgan_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv realesrgan_env"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source realesrgan_env/bin/activate

# Install real Real-ESRGAN
echo "📥 Installing Real-ESRGAN packages..."
pip install --upgrade pip
pip install realesrgan basicsr facexlib gfpgan
pip install torch torchvision # Ensure PyTorch is installed

# Run the model download script
echo "🤖 Downloading AI models..."
python scripts/install_real_esrgan.py

# Make scripts executable
chmod +x scripts/upscale_frame_real.py

echo
echo "✅ Real-ESRGAN setup complete!"
echo
echo "To test the installation:"
echo "  ./realesrgan_env/bin/python scripts/upscale_frame_real.py test.jpg output.jpg --benchmark"
echo
echo "Note: First run will be slower as models are loaded into memory."