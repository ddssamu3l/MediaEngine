#!/bin/bash
# Setup script for Real-ESRGAN integration

echo "🚀 Setting up AI Upscaling (Real-ESRGAN)..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv realesrgan_env

# Activate environment
source realesrgan_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust based on your system)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected, installing CUDA version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 Installing CPU version..."
    pip install torch torchvision torchaudio
fi

# Install Real-ESRGAN
echo "🎯 Installing Real-ESRGAN..."
pip install realesrgan

# Install additional dependencies
pip install opencv-python pillow numpy

# Create scripts directory
mkdir -p scripts

# Test installation
echo "🧪 Testing installation..."
python -c "import realesrgan; print('✅ Real-ESRGAN installed successfully')"

echo "✅ Setup complete!"
echo ""
echo "To use AI upscaling:"
echo "1. Activate the environment: source realesrgan_env/bin/activate"
echo "2. Run your media engine with upscaling enabled"
echo ""
echo "Models will be automatically downloaded on first use." 