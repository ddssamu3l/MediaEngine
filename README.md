# Universal Media Engine

A terminal-based tool for converting videos to multiple formats (GIF, APNG, WebP, AVIF, MP4, WebM) with real-time progress tracking.

## Prerequisites

### 1. Go (version 1.21 or later)
```bash
# Check if Go is installed
go version

# Install Go if needed:
# macOS: brew install go
# Ubuntu: sudo apt install golang-go
# Windows: Download from https://golang.org/download.html
```

### 2. FFmpeg and FFprobe
Both tools are required for media processing and validation:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

Verify installation:
```bash
ffmpeg -version
ffprobe -version
```

### 3. Python (for AI Upscaling - Optional)
AI upscaling requires Python 3.7+ with PyTorch:

```bash
# Create virtual environment (recommended)
python3 -m venv realesrgan_env
source realesrgan_env/bin/activate  # Linux/macOS
# or
realesrgan_env\Scripts\activate     # Windows

# Install required packages
pip install torch torchvision opencv-python numpy

# Optional: Install Real-ESRGAN for best quality
pip install realesrgan basicsr
```

## Installation

1. **Install Go dependencies:**
```bash
go mod tidy
```

2. **Set up AI models (optional):**
If you want to use AI upscaling, create a `models` directory and download the model files:
```bash
mkdir models
cd models

# Download Real-ESRGAN models (optional, for best quality)
# General purpose 4x model (~64MB)
curl -L -o RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# General purpose 2x model (~64MB)  
curl -L -o RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

# Anime/cartoon optimized model (~17MB)
curl -L -o RealESRGAN_x4plus_anime_6B.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth

cd ..
```

3. **Build the application:**
```bash
go build .
```

## Usage

### Quick Start (Recommended)
For the easiest experience, use the launcher scripts that handle environment setup automatically:

**macOS/Linux:**
```bash
# Install FFmpeg first
brew install ffmpeg              # macOS
sudo apt install ffmpeg         # Ubuntu/Debian

# Run the media engine
./run_mediaengine.sh
```

**Windows:**
```cmd
# Install FFmpeg from https://ffmpeg.org/download.html
# Or with chocolatey: choco install ffmpeg

# Run the media engine
run_mediaengine.bat
```

The launcher script will automatically:
1. Create a Python virtual environment
2. Install required packages (PyTorch, OpenCV, etc.)
3. Set up AI models for upscaling
4. Launch the media engine

### Manual Run (For Developers):
```bash
# Option 1: Run directly
go run .

# Option 2: Use built binary
./mediaengine
```

### Interactive workflow:
1. Enter path to your video file
2. Select output format (GIF, APNG, WebP, AVIF, MP4, WebM)
3. Choose quality profile or set custom quality
4. Set time range (start/end times)
5. Choose frame rate
6. **Select AI scaling option:**
   - Keep Original Resolution (no AI processing)
   - Upscale - Enhance quality (2x, 4x with neural networks)
   - Downscale - Reduce quality (1/2x, 1/4x, 1/8x)
7. Specify output path
8. Watch real-time conversion progress with GPU acceleration

### Supported formats:
- **Input**: MP4, MKV, MOV, AVI, WebM, FLV, WMV
- **Output**: GIF, APNG, WebP, AVIF, MP4, WebM

### AI Upscaling Features:
- **GPU Acceleration**: Supports Apple Silicon (MPS), NVIDIA CUDA, and CPU fallback
- **Multiple Models**: General purpose (2x, 4x) and anime-optimized (4x) models
- **Quality Preservation**: Maintains exact aspect ratios during scaling
- **Downscaling**: Intelligent quality reduction for smaller file sizes
- **Progress Tracking**: Real-time GPU utilization and processing status

## Dependencies

The project uses these Go modules:
- `github.com/charmbracelet/lipgloss` - Terminal styling
- `github.com/manifoldco/promptui` - Interactive prompts  
- `github.com/schollz/progressbar/v3` - Progress tracking

## Troubleshooting

### Common issues:

**FFmpeg not found:**
```
❌ FFmpeg or FFprobe is not installed or not in PATH
```
- Install FFmpeg and ensure both `ffmpeg` and `ffprobe` are in your PATH

**Permission denied:**
- Check read permissions on input file
- Check write permissions on output directory

**Codec not available:**
```
❌ The AVIF format encoder is not available
```
- Install FFmpeg with full codec support, or choose a different output format

**Build errors:**
- Ensure Go 1.21+ is installed: `go version`
- Run `go mod tidy` to update dependencies

**AI Upscaling not available:**
```
⚠️ AI Scaling not available (Real-ESRGAN not installed)
```
- Install Python 3.7+ and required packages (see Prerequisites section)
- Download model files to the `models/` directory (see Installation section)
- Ensure virtual environment is activated if using one

**GPU acceleration issues:**
- **Apple Silicon**: Install PyTorch with MPS support: `pip install torch torchvision`
- **NVIDIA GPU**: Install PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- **CPU fallback**: The system will automatically use CPU if GPU is unavailable

**Python environment issues:**
- Check Python version: `python3 --version` (must be 3.7+)
- Verify PyTorch installation: `python3 -c "import torch; print(torch.__version__)"`
- Check GPU availability: `python3 -c "import torch; print(torch.backends.mps.is_available())"`

## Development

Run tests:
```bash
go test ./...
```

Build for different platforms:
```bash
# Windows
GOOS=windows GOARCH=amd64 go build -o mediaengine.exe .

# macOS
GOOS=darwin GOARCH=amd64 go build -o mediaengine-macos .

# Linux
GOOS=linux GOARCH=amd64 go build -o mediaengine-linux .
```