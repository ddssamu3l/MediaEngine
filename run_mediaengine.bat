@echo off
REM Universal Media Engine Launcher for Windows
REM This script sets up the Python environment and runs the media engine

echo 🎬 Universal Media Engine
echo Setting up environment...

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 3 is required but not installed.
    echo Please install Python 3.7+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "realesrgan_env" (
    echo 🔧 Creating Python virtual environment...
    python -m venv realesrgan_env
    
    REM Activate virtual environment
    call realesrgan_env\Scripts\activate.bat
    
    REM Upgrade pip
    python -m pip install --upgrade pip
    
    echo 📦 Installing required Python packages...
    pip install torch torchvision opencv-python numpy
    
    REM Try to install Real-ESRGAN (optional)
    echo 🤖 Installing Real-ESRGAN (optional, for best quality)...
    pip install realesrgan basicsr || echo ⚠️  Real-ESRGAN installation failed, using fallback mode
    
    echo ✅ Python environment setup complete!
) else (
    echo ✅ Python environment already exists
)

REM Activate virtual environment
call realesrgan_env\Scripts\activate.bat

REM Check if FFmpeg is available
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ FFmpeg is required but not installed.
    echo Please install FFmpeg:
    echo Windows: Download from https://ffmpeg.org/download.html
    echo Or install with chocolatey: choco install ffmpeg
    pause
    exit /b 1
)

REM Run the media engine
echo 🚀 Starting Universal Media Engine...
echo.

REM Check if we have Windows executable, otherwise use the Unix one
if exist "mediaengine.exe" (
    mediaengine.exe
) else (
    mediaengine
)

pause