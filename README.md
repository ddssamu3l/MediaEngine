# VideoToGIF

A robust, terminal-based CLI tool for converting MP4 videos to GIF with comprehensive validation, security features, and real-time progress tracking.

## Features

✨ **Comprehensive Input Validation**
- MP4 file format verification
- File size limits (500MB default)
- Path security validation
- Permission checks

🎯 **Smart User Interface**
- Arrow-key navigable resolution selection
- Real-time FFmpeg progress tracking
- Styled terminal UI with clear error messages
- Conversion summary before processing

🔒 **Security & Safety**
- Directory traversal protection
- System directory write protection
- File permission validation
- Path sanitization

⚡ **Optimized Conversion**
- High-quality GIF output with Lanczos scaling
- Customizable frame rates (1-30 fps)
- Multiple resolution presets
- Efficient FFmpeg integration

## Prerequisites

### 1. Go Installation (Version 1.21 or later)

**macOS:**
```bash
# Using Homebrew
brew install go

# Or download from official site
curl -L https://golang.org/dl/go1.21.0.darwin-amd64.tar.gz -o go.tar.gz
sudo tar -C /usr/local -xzf go.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.zshrc
source ~/.zshrc
```

**Ubuntu/Debian:**
```bash
# Using package manager
sudo apt update
sudo apt install golang-go

# Or install latest version manually
wget https://golang.org/dl/go1.21.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
```

**Windows:**
1. Download installer from https://golang.org/dl/
2. Run the MSI installer
3. Go will be automatically added to PATH

### 2. FFmpeg Installation

**macOS:**
```bash
# Using Homebrew (recommended)
brew install ffmpeg

# Using MacPorts
sudo port install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**CentOS/RHEL/Rocky Linux:**
```bash
# Enable EPEL repository first
sudo dnf install epel-release
sudo dnf install ffmpeg ffmpeg-devel
```

**Windows:**
1. Download from https://ffmpeg.org/download.html#build-windows
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH:
   - Open System Properties → Advanced → Environment Variables
   - Edit the PATH variable and add `C:\ffmpeg\bin`
   - Restart terminal/command prompt

**Arch Linux:**
```bash
sudo pacman -S ffmpeg
```

## Installation

### Method 1: Build from Source

1. **Clone the repository:**
```bash
git clone <repository-url>
cd videotogif
```

2. **Install dependencies:**
```bash
go mod tidy
```

3. **Build the application:**
```bash
go build -o videotogif .
```

4. **Test the installation:**
```bash
./videotogif
```

### Method 2: System-wide Installation

After building the application, install it system-wide:

**macOS/Linux:**
```bash
# Build the binary
go build -o videotogif .

# Install to system location
sudo cp videotogif /usr/local/bin/

# Make sure /usr/local/bin is in your PATH
echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc  # or ~/.zshrc
source ~/.bashrc  # or source ~/.zshrc

# Verify installation
which videotogif
videotogif --version
```

**Windows:**
```cmd
# Build the binary
go build -o videotogif.exe .

# Create a tools directory (if it doesn't exist)
mkdir C:\tools

# Copy the executable
copy videotogif.exe C:\tools\

# Add C:\tools to your PATH:
# 1. Open System Properties → Advanced → Environment Variables
# 2. Edit the PATH variable and add C:\tools
# 3. Open a new command prompt and test:
videotogif
```

### Method 3: Go Install (Direct)

```bash
# Install directly from source
go install github.com/youruser/videotogif@latest

# The binary will be installed to $GOPATH/bin or $HOME/go/bin
# Make sure this directory is in your PATH
```

## Cross-Platform Builds

Build for different platforms:

```bash
# Windows 64-bit
GOOS=windows GOARCH=amd64 go build -o videotogif-windows.exe .

# macOS (Intel)
GOOS=darwin GOARCH=amd64 go build -o videotogif-macos-intel .

# macOS (Apple Silicon)
GOOS=darwin GOARCH=arm64 go build -o videotogif-macos-arm .

# Linux 64-bit
GOOS=linux GOARCH=amd64 go build -o videotogif-linux .

# Linux ARM64 (Raspberry Pi)
GOOS=linux GOARCH=arm64 go build -o videotogif-linux-arm64 .
```

## Usage

### Basic Usage

1. **Run the tool:**
```bash
videotogif
```

2. **Follow the interactive prompts:**
   - Enter path to your MP4 video file
   - Review the displayed video information
   - Set start time, end time, and frame rate
   - Select output resolution using arrow keys
   - Specify output path for the GIF
   - Confirm conversion settings
   - Watch real-time progress

### Command Line Examples

The tool currently operates in interactive mode. Future versions may include command-line arguments.

## Features in Detail

### Input Validation
- Validates MP4 file format and integrity
- Checks file permissions and accessibility
- Enforces file size limits (500MB default)
- Prevents directory traversal attacks

### Security Features
- Path sanitization and cleaning
- System directory protection
- Write permission verification
- Input validation against malicious paths

### Progress Tracking
- Real-time FFmpeg progress parsing
- Frame-accurate progress reporting
- Time estimation and elapsed time display
- Visual progress bar with completion percentage

### Output Quality
- Lanczos scaling for high-quality resizing
- Optimized GIF encoding parameters
- Customizable frame rates (1-30 fps)
- Multiple resolution presets

## Troubleshooting

### Common Issues

**"FFmpeg not found" error:**
- Verify FFmpeg installation: `ffmpeg -version`
- Check PATH configuration
- Reinstall FFmpeg if necessary

**Permission denied errors:**
- Check file read permissions for input video
- Verify write permissions for output directory
- Run with appropriate user privileges

**File format errors:**
- Ensure input file is a valid MP4
- Check if file is corrupted
- Try a different input file

**Memory issues with large files:**
- Use files smaller than 500MB
- Consider reducing resolution or frame rate
- Close other applications to free memory

### Getting Help

If you encounter issues:
1. Check the error message for specific guidance
2. Verify all prerequisites are installed
3. Test with a small, simple MP4 file
4. Check file and directory permissions

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here]