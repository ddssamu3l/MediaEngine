# VideoToGIF

Convert MP4 videos to GIF with ease using this terminal-based CLI tool built with Go and FFmpeg.

## Project Structure

```
videotogif/
├── main.go
├── go.mod
├── internal/
│   ├── ffmpeg/
│   │   └── ffmpeg.go
│   ├── ui/
│   │   └── ui.go
│   └── video/
│       └── info.go
└── README.md
```

## Setup Instructions

### Prerequisites

1. **Install Go** (version 1.21 or later):
   - Download from https://golang.org/dl/
   - Follow installation instructions for your OS

2. **Install FFmpeg**:
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
   - **Windows**: Download from https://ffmpeg.org/download.html and add to PATH
   - **CentOS/RHEL**: `sudo yum install ffmpeg` or `sudo dnf install ffmpeg`

### Building and Running

1. **Clone/Create the project**:
```bash
mkdir videotogif
cd videotogif
```

2. **Initialize Go module**:
```bash
go mod init videotogif
```

3. **Create the directory structure and files** as shown above.

4. **Install dependencies**:
```bash
go mod tidy
```

5. **Build the application**:
```bash
go build -o videotogif .
```

6. **Run the application**:
```bash
./videotogif
```

### Installation (Optional)

To install the tool system-wide:

1. **Build for your system**:
```bash
go build -o videotogif .
```

2. **Move to a directory in your PATH**:
```bash
# On macOS/Linux
sudo mv videotogif /usr/local/bin/

# On Windows, move videotogif.exe to a directory in your PATH
```

3. **Now you can run it from anywhere**:
```bash
videotogif
```

### Cross-compilation (Optional)

To build for different platforms:

```bash
# For Windows
GOOS=windows GOARCH=amd64 go build -o videotogif.exe .

# For macOS
GOOS=darwin GOARCH=amd64 go build -o videotogif-mac .

# For Linux
GOOS=linux GOARCH=amd64 go build -o videotogif-linux .
```

## Usage

1. Run the tool: `./videotogif`
2. Enter the path to your MP4 video file
3. Review the displayed video information
4. Set the start time, end time, and frame rate
5. Select the desired resolution from the dropdown
6. Specify the output path for the GIF
7. Watch the progress bar as your video converts to GIF!

The tool supports both absolute and relative file paths and provides a beautiful terminal UI with styled output and progress tracking.