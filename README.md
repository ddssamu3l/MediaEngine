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

## Installation

1. **Install Go dependencies:**
```bash
go mod tidy
```

2. **Build the application:**
```bash
go build .
```

## Usage

### Run the application:
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
5. Choose frame rate and resolution
6. Specify output path
7. Watch real-time conversion progress

### Supported formats:
- **Input**: MP4, MKV, MOV, AVI, WebM, FLV, WMV
- **Output**: GIF, APNG, WebP, AVIF, MP4, WebM

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
- Use the sqlite_fts5 build tag: `go build -tags sqlite_fts5 .`

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