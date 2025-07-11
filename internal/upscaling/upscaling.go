// internal/upscaling/upscaling.go
package upscaling

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// UpscalingConfig holds configuration for AI scaling (both upscaling and downscaling)
type UpscalingConfig struct {
	Enabled      bool
	Model        string // RealESRGAN_x4plus, RealESRGAN_x2plus, RealESRGAN_x4plus_anime_6B
	Scale        int    // Positive for upscaling (2, 3, 4), negative for downscaling (-2, -4, -8)
	UseGPU       bool
	GPUDevice    int
	PythonPath   string
	ScriptPath   string
	TempDir      string
	IsDownscale  bool   // True if scale is negative
}

// VideoUpscalingResult contains the results of video upscaling
type VideoUpscalingResult struct {
	InputPath       string
	OutputPath      string
	OriginalSize    VideoSize
	UpscaledSize    VideoSize
	ProcessingTime  time.Duration
	FramesProcessed int
	Success         bool
	ErrorMessage    string
}

// VideoSize represents video dimensions
type VideoSize struct {
	Width  int
	Height int
}

// MemoryEstimate represents memory usage estimation
type MemoryEstimate struct {
	EstimatedGB     float64
	AvailableGB     float64
	RecommendedSafe bool
}

// ProgressCallback is called during video processing to report progress
type ProgressCallback func(current, total int, message string)

// UpscalingModels defines available models
var UpscalingModels = map[string]string{
	"general_4x": "RealESRGAN_x4plus",
	"general_2x": "RealESRGAN_x2plus",
	"anime_4x":   "RealESRGAN_x4plus_anime_6B",
}

// UpscalingModelDescriptions provides user-friendly descriptions
var UpscalingModelDescriptions = map[string]string{
	"general_4x": "General Purpose 4x (Best for photos/real content)",
	"general_2x": "General Purpose 2x (Faster, good quality)",
	"anime_4x":   "Anime/Cartoon 4x (Optimized for animated content)",
}

// Upscaler handles AI upscaling operations
type Upscaler struct {
	config *UpscalingConfig
}

// NewUpscaler creates a new upscaler instance
func NewUpscaler(config *UpscalingConfig) *Upscaler {
	return &Upscaler{config: config}
}

// IsAvailable checks if Real-ESRGAN is available
func (u *Upscaler) IsAvailable() bool {
	if u.config.PythonPath == "" {
		return false
	}

	// Check if Python environment exists
	if _, err := os.Stat(u.config.PythonPath); os.IsNotExist(err) {
		return false
	}

	// Check if script exists
	if _, err := os.Stat(u.config.ScriptPath); os.IsNotExist(err) {
		return false
	}

	// Test if our direct implementation dependencies are available
	cmd := exec.Command(u.config.PythonPath, "-c", "import torch; import cv2; import numpy; print('OK')")
	output, err := cmd.Output()
	if err != nil {
		return false
	}

	// Check if models directory exists
	if _, err := os.Stat("models"); os.IsNotExist(err) {
		return false
	}

	// Check if at least one model exists
	modelFiles := []string{"models/RealESRGAN_x4plus.pth", "models/RealESRGAN_x2plus.pth", "models/RealESRGAN_x4plus_anime_6B.pth"}
	for _, modelPath := range modelFiles {
		if _, err := os.Stat(modelPath); err == nil {
			return strings.Contains(string(output), "OK")
		}
	}

	return false
}

// EstimateMemoryUsage estimates memory requirements for video upscaling
func (u *Upscaler) EstimateMemoryUsage(width, height int) (*MemoryEstimate, error) {
	// Constants for memory estimation
	const (
		dtypeSize      = 4   // float32 size in bytes
		modelMemoryGB  = 2.0 // Approximate model memory in GB
		overheadFactor = 1.5 // Processing overhead multiplier
		safetyMargin   = 0.8 // Use max 80% of available memory
	)

	// Calculate image memory usage
	inputMemory := float64(width * height * 3 * dtypeSize)
	
	var outputMemory float64
	if u.config.Scale > 0 {
		// Upscaling
		outputMemory = float64((width * u.config.Scale) * (height * u.config.Scale) * 3 * dtypeSize)
	} else {
		// Downscaling
		divisor := -u.config.Scale
		outputMemory = float64((width / divisor) * (height / divisor) * 3 * dtypeSize)
	}

	// Total memory estimation
	totalMemory := (inputMemory + outputMemory + modelMemoryGB*1024*1024*1024) * overheadFactor
	estimatedGB := totalMemory / (1024 * 1024 * 1024)

	// Get available system memory
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Get total system memory (approximation)
	availableGB := float64(m.Sys) / (1024 * 1024 * 1024)
	if availableGB < 1 {
		// Fallback estimation if memory stats are unreliable
		availableGB = 8.0 // Assume 8GB minimum
	}

	estimate := &MemoryEstimate{
		EstimatedGB:     estimatedGB,
		AvailableGB:     availableGB,
		RecommendedSafe: estimatedGB <= availableGB*safetyMargin,
	}

	return estimate, nil
}

// GetVideoInfo extracts video information using ffprobe
func (u *Upscaler) GetVideoInfo(videoPath string) (*VideoSize, int, error) {
	// Use ffprobe to get video information
	cmd := exec.Command("ffprobe", "-v", "quiet", "-select_streams", "v:0",
		"-show_entries", "stream=width,height,nb_frames", "-of", "csv=s=x:p=0", videoPath)

	output, err := cmd.Output()
	if err != nil {
		return nil, 0, fmt.Errorf("failed to get video info: %v", err)
	}

	// Parse output (format: widthxheightxframes)
	parts := strings.Split(strings.TrimSpace(string(output)), "x")
	if len(parts) < 2 {
		return nil, 0, fmt.Errorf("invalid video info format")
	}

	width, err := strconv.Atoi(parts[0])
	if err != nil {
		return nil, 0, fmt.Errorf("invalid width: %v", err)
	}

	height, err := strconv.Atoi(parts[1])
	if err != nil {
		return nil, 0, fmt.Errorf("invalid height: %v", err)
	}

	// Extract frame count (may not always be available)
	frameCount := 0
	if len(parts) >= 3 && parts[2] != "N/A" && parts[2] != "" {
		frameCount, _ = strconv.Atoi(parts[2])
	}

	return &VideoSize{Width: width, Height: height}, frameCount, nil
}

// UpscaleVideo processes an entire video with AI upscaling
func (u *Upscaler) UpscaleVideo(inputPath, outputPath string, startTime, endTime float64, frameRate int, progressCallback ProgressCallback) (*VideoUpscalingResult, error) {
	processingStartTime := time.Now()
	result := &VideoUpscalingResult{
		InputPath:  inputPath,
		OutputPath: outputPath,
		Success:    false,
	}

	if !u.config.Enabled {
		result.ErrorMessage = "upscaling is disabled"
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if !u.IsAvailable() {
		result.ErrorMessage = "Real-ESRGAN is not available"
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Get video information
	videoSize, _, err := u.GetVideoInfo(inputPath)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to analyze video: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	result.OriginalSize = *videoSize
	
	// Calculate output size based on scaling mode
	if u.config.Scale > 0 {
		// Upscaling
		result.UpscaledSize = VideoSize{
			Width:  videoSize.Width * u.config.Scale,
			Height: videoSize.Height * u.config.Scale,
		}
	} else {
		// Downscaling (scale is negative)
		divisor := -u.config.Scale
		result.UpscaledSize = VideoSize{
			Width:  videoSize.Width / divisor,
			Height: videoSize.Height / divisor,
		}
	}

	// Estimate memory usage
	memEstimate, err := u.EstimateMemoryUsage(videoSize.Width, videoSize.Height)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to estimate memory usage: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if !memEstimate.RecommendedSafe {
		result.ErrorMessage = fmt.Sprintf("insufficient memory: estimated %.1fGB, available %.1fGB",
			memEstimate.EstimatedGB, memEstimate.AvailableGB)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Create temporary directory for processing
	tempDir, err := os.MkdirTemp(u.config.TempDir, "upscaling_*")
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to create temp directory: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}
	defer func() {
		// Clean up temporary files
		if removeErr := os.RemoveAll(tempDir); removeErr != nil {
			fmt.Printf("Warning: failed to clean up temp directory %s: %v\n", tempDir, removeErr)
		}
	}()

	if progressCallback != nil {
		progressCallback(0, 100, "Initializing AI upscaling...")
	}

	// Step 1: Extract frames
	framesDir := filepath.Join(tempDir, "frames")
	if err := os.MkdirAll(framesDir, 0755); err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to create frames directory: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if progressCallback != nil {
		progressCallback(10, 100, "Extracting video frames...")
	}

	framePattern := filepath.Join(framesDir, "frame_%06d.png")
	
	// Build ffmpeg command with time range and frame rate constraints
	extractCmd := exec.Command("ffmpeg", 
		"-i", inputPath,
		"-ss", fmt.Sprintf("%.3f", startTime), // Start time
		"-t", fmt.Sprintf("%.3f", endTime-startTime), // Duration
		"-r", fmt.Sprintf("%d", frameRate), // Frame rate
		"-y", framePattern)
	
	if err := extractCmd.Run(); err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to extract frames: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Count extracted frames
	frameFiles, err := filepath.Glob(filepath.Join(framesDir, "frame_*.png"))
	if err != nil || len(frameFiles) == 0 {
		result.ErrorMessage = "no frames extracted from video"
		return result, fmt.Errorf(result.ErrorMessage)
	}

	actualFrameCount := len(frameFiles)
	result.FramesProcessed = actualFrameCount

	if progressCallback != nil {
		scalingType := "upscaling"
		if u.config.Scale < 0 {
			scalingType = "downscaling"
		}
		progressCallback(20, 100, fmt.Sprintf("Processing %d frames with AI %s...", actualFrameCount, scalingType))
	}

	// Step 2: Upscale frames
	upscaledDir := filepath.Join(tempDir, "upscaled")
	if err := os.MkdirAll(upscaledDir, 0755); err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to create upscaled directory: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Process frames in batches to manage memory
	batchSize := 10                                  // Process 10 frames at a time
	progressStep := 60.0 / float64(actualFrameCount) // 60% of progress for upscaling

	for i, frameFile := range frameFiles {
		frameName := filepath.Base(frameFile)
		upscaledPath := filepath.Join(upscaledDir, frameName)

		if err := u.UpscaleFrame(frameFile, upscaledPath); err != nil {
			result.ErrorMessage = fmt.Sprintf("failed to upscale frame %s: %v", frameName, err)
			return result, fmt.Errorf(result.ErrorMessage)
		}

		// Update progress
		if progressCallback != nil {
			currentProgress := 20 + int(float64(i+1)*progressStep)
			scaledVerb := "Upscaled"
			if u.config.Scale < 0 {
				scaledVerb = "Downscaled"
			}
			progressCallback(currentProgress, 100, fmt.Sprintf("%s frame %d/%d", scaledVerb, i+1, actualFrameCount))
		}

		// Memory management: force garbage collection every batch
		if (i+1)%batchSize == 0 {
			runtime.GC()
		}
	}

	if progressCallback != nil {
		scaledType := "upscaled"
		if u.config.Scale < 0 {
			scaledType = "downscaled"
		}
		progressCallback(80, 100, fmt.Sprintf("Reassembling video with %s frames...", scaledType))
	}

	// Step 3: Extract audio from original video
	audioPath := filepath.Join(tempDir, "audio.aac")
	audioCmd := exec.Command("ffmpeg", "-i", inputPath, "-vn", "-acodec", "copy", "-y", audioPath)
	audioCmd.Run() // Don't fail if no audio track exists

	// Step 4: Get original video frame rate for reassembly
	originalFrameRate := "15" // Default fallback (matches the app's default)
	if frameRateCmd := exec.Command("ffprobe", "-v", "quiet", "-select_streams", "v:0",
		"-show_entries", "stream=r_frame_rate", "-of", "csv=s=x:p=0", inputPath); frameRateCmd != nil {
		if output, err := frameRateCmd.Output(); err == nil {
			frameRateStr := strings.TrimSpace(string(output))
			if frameRateStr != "" && frameRateStr != "0/0" && frameRateStr != "N/A" {
				// Handle fractional frame rates like "30000/1001"
				if strings.Contains(frameRateStr, "/") {
					parts := strings.Split(frameRateStr, "/")
					if len(parts) == 2 {
						if num, err1 := strconv.ParseFloat(parts[0], 64); err1 == nil {
							if den, err2 := strconv.ParseFloat(parts[1], 64); err2 == nil && den != 0 {
								fps := num / den
								originalFrameRate = fmt.Sprintf("%.3f", fps)
							}
						}
					}
				} else {
					originalFrameRate = frameRateStr
				}
			}
		}
	}

	// Step 5: Reassemble video with improved FFmpeg parameters
	upscaledPattern := filepath.Join(upscaledDir, "frame_%06d.png")
	var assembleCmd *exec.Cmd

	// Check if audio file was created
	if _, err := os.Stat(audioPath); err == nil {
		// Reassemble with audio
		assembleCmd = exec.Command("ffmpeg",
			"-framerate", originalFrameRate,
			"-i", upscaledPattern,
			"-i", audioPath,
			"-c:v", "libx264",
			"-preset", "medium",
			"-crf", "18",
			"-pix_fmt", "yuv420p", // Ensure compatibility
			"-c:a", "aac",
			"-movflags", "+faststart", // Web optimization
			"-y", outputPath)
	} else {
		// Reassemble without audio
		assembleCmd = exec.Command("ffmpeg",
			"-framerate", originalFrameRate,
			"-i", upscaledPattern,
			"-c:v", "libx264",
			"-preset", "medium",
			"-crf", "18",
			"-pix_fmt", "yuv420p", // Ensure compatibility
			"-movflags", "+faststart", // Web optimization
			"-y", outputPath)
	}

	// Run command and capture output for debugging
	cmdOutput, err := assembleCmd.CombinedOutput()
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to reassemble video: %v\nFFmpeg command: %v\nFFmpeg output: %s",
			err, assembleCmd.Args, string(cmdOutput))
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if progressCallback != nil {
		scaledType := "upscaled"
		if u.config.Scale < 0 {
			scaledType = "downscaled"
		}
		progressCallback(95, 100, fmt.Sprintf("Finalizing %s video...", scaledType))
	}

	// Verify output file was created
	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		result.ErrorMessage = "video processing completed but output file was not created"
		return result, fmt.Errorf(result.ErrorMessage)
	}

	result.ProcessingTime = time.Since(processingStartTime)
	result.Success = true

	if progressCallback != nil {
		scalingType := "upscaling"
		if u.config.Scale < 0 {
			scalingType = "downscaling"
		}
		progressCallback(100, 100, fmt.Sprintf("AI %s completed successfully!", scalingType))
	}

	return result, nil
}

// UpscaleFrame processes a single frame using Real-ESRGAN (upscaling or downscaling)
func (u *Upscaler) UpscaleFrame(inputPath, outputPath string) error {
	if !u.config.Enabled {
		return fmt.Errorf("scaling is disabled")
	}

	if !u.IsAvailable() {
		return fmt.Errorf("Real-ESRGAN is not available")
	}

	// For downscaling, we need to handle it differently
	if u.config.Scale < 0 {
		return u.downscaleFrame(inputPath, outputPath)
	}

	// Get the Real-ESRGAN model name from the mapping
	realModelName, exists := UpscalingModels[u.config.Model]
	if !exists {
		return fmt.Errorf("unknown model key: %s", u.config.Model)
	}

	// Build command arguments for upscaling
	args := []string{
		u.config.ScriptPath,
		inputPath,
		outputPath,
		"--model", realModelName, // Use the Real-ESRGAN model name
		"--scale", strconv.Itoa(u.config.Scale),
	}

	// Add GPU/CPU flag
	if !u.config.UseGPU {
		args = append(args, "--cpu")
	} else if u.config.GPUDevice > 0 {
		args = append(args, "--gpu", strconv.Itoa(u.config.GPUDevice))
	}
	
	// Add FP16 flag for better GPU performance
	if u.config.UseGPU {
		args = append(args, "--fp16")
	}

	// Execute upscaling
	cmd := exec.Command(u.config.PythonPath, args...)
	output, err := cmd.CombinedOutput()

	if err != nil {
		return fmt.Errorf("upscaling failed: %v\nOutput: %s", err, string(output))
	}

	// Verify output file was created
	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		return fmt.Errorf("upscaling completed but output file was not created")
	}

	return nil
}

// downscaleFrame downscales a single frame using FFmpeg
func (u *Upscaler) downscaleFrame(inputPath, outputPath string) error {
	// Calculate the scale factor for FFmpeg
	divisor := -u.config.Scale
	scaleFilter := fmt.Sprintf("scale=iw/%d:ih/%d", divisor, divisor)

	// Use FFmpeg to downscale while maintaining aspect ratio and quality
	cmd := exec.Command("ffmpeg",
		"-i", inputPath,
		"-vf", scaleFilter,
		"-c:v", "png",        // Use PNG for lossless intermediate frames
		"-pix_fmt", "rgb24",  // Maintain color accuracy
		"-y", outputPath)

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("downscaling failed: %v\nOutput: %s", err, string(output))
	}

	// Verify output file was created
	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		return fmt.Errorf("downscaling completed but output file was not created")
	}

	return nil
}

// DetectPythonPath attempts to find a suitable Python executable
func DetectPythonPath() string {
	// Try different Python commands in order of preference
	candidates := []string{"python3", "python", "py"}

	// Add platform-specific paths
	if runtime.GOOS == "windows" {
		candidates = append(candidates, "py -3")
	}

	for _, candidate := range candidates {
		var cmd *exec.Cmd
		if strings.Contains(candidate, " ") {
			// Handle commands with arguments (like "py -3")
			parts := strings.Fields(candidate)
			cmd = exec.Command(parts[0], append(parts[1:], "--version")...)
		} else {
			cmd = exec.Command(candidate, "--version")
		}

		if output, err := cmd.Output(); err == nil {
			// Verify it's Python 3.7+
			versionRegex := regexp.MustCompile(`Python 3\.(\d+)`)
			if match := versionRegex.FindStringSubmatch(string(output)); len(match) > 1 {
				if minorVersion, err := strconv.Atoi(match[1]); err == nil && minorVersion >= 7 {
					return candidate
				}
			}
		}
	}

	return ""
}

// GetModelInfo returns information about available models
func GetModelInfo() map[string]string {
	return UpscalingModelDescriptions
}

// ValidateConfig validates upscaling configuration
func ValidateConfig(config *UpscalingConfig) error {
	if !config.Enabled {
		return nil
	}

	// Check Python path
	if config.PythonPath == "" {
		return fmt.Errorf("Python path is required for upscaling")
	}

	// For compound commands like "py -3", just check the base command
	baseCmd := strings.Fields(config.PythonPath)[0]
	if _, err := exec.LookPath(baseCmd); err != nil {
		return fmt.Errorf("Python executable not found: %s", config.PythonPath)
	}

	// Check script path
	if config.ScriptPath == "" {
		return fmt.Errorf("upscaling script path is required")
	}

	if _, err := os.Stat(config.ScriptPath); os.IsNotExist(err) {
		return fmt.Errorf("upscaling script not found: %s", config.ScriptPath)
	}

	// Validate model
	if _, exists := UpscalingModels[config.Model]; !exists {
		return fmt.Errorf("invalid upscaling model: %s", config.Model)
	}

	// Validate scale
	if config.Scale == 0 {
		return nil // No scaling
	}
	
	if config.Scale > 0 {
		// Upscaling
		if config.Scale < 2 || config.Scale > 4 {
			return fmt.Errorf("upscaling scale must be 2, 3, or 4")
		}
	} else {
		// Downscaling (negative values)
		allowedDownscales := map[int]bool{-2: true, -4: true, -8: true}
		if !allowedDownscales[config.Scale] {
			return fmt.Errorf("downscaling scale must be -2, -4, or -8")
		}
	}

	// Validate GPU settings
	if config.UseGPU && config.GPUDevice < 0 {
		return fmt.Errorf("invalid GPU device ID: %d", config.GPUDevice)
	}

	return nil
}

// GetGPUInfo returns information about available GPU acceleration
func GetGPUInfo(pythonPath string) string {
	cmd := exec.Command(pythonPath, "-c", `
import torch
import subprocess
import platform

info = []
info.append(f"Platform: {platform.system()}")

if torch.backends.mps.is_available():
    try:
        result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                              capture_output=True, text=True, timeout=5)
        if 'Chip:' in result.stdout:
            chip_line = [line for line in result.stdout.split('\n') if 'Chip:' in line][0]
            chip_name = chip_line.split('Chip:')[1].strip()
            info.append(f"GPU: {chip_name} (MPS)")
        else:
            info.append("GPU: Apple Silicon (MPS)")
    except:
        info.append("GPU: Apple Silicon (MPS)")
elif torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    info.append(f"GPU: {gpu_name} ({gpu_memory:.1f}GB CUDA)")
else:
    info.append("GPU: None (CPU only)")

print(" | ".join(info))
`)

	output, err := cmd.Output()
	if err != nil {
		return "GPU: Unknown"
	}

	return strings.TrimSpace(string(output))
}

// GetDefaultConfig returns default upscaling configuration
func GetDefaultConfig() *UpscalingConfig {
	// Check for virtual environment first
	venvPython := "./realesrgan_env/bin/python"
	if runtime.GOOS == "windows" {
		venvPython = "./realesrgan_env/Scripts/python.exe"
	}

	var pythonPath string
	if _, err := os.Stat(venvPython); err == nil {
		pythonPath = venvPython
	} else {
		// Fallback to auto-detect system Python
		pythonPath = DetectPythonPath()
		if pythonPath == "" {
			pythonPath = "python3" // Final fallback
		}
	}

	return &UpscalingConfig{
		Enabled:    false,
		Model:      "general_4x",
		Scale:      4,
		UseGPU:     true,  // Enable GPU by default
		GPUDevice:  0,
		PythonPath: pythonPath,
		ScriptPath: "scripts/real_esrgan_working.py",  // Use working Real-ESRGAN implementation
		TempDir:    os.TempDir(),
	}
}
