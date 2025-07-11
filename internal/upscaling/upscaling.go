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

// UpscalingConfig holds configuration for AI upscaling
type UpscalingConfig struct {
	Enabled    bool
	Model      string // RealESRGAN_x4plus, RealESRGAN_x2plus, RealESRGAN_x4plus_anime_6B
	Scale      int    // 2, 3, or 4
	UseGPU     bool
	GPUDevice  int
	PythonPath string
	ScriptPath string
	TempDir    string
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

	// Test if realesrgan package is available
	cmd := exec.Command(u.config.PythonPath, "-c", "import realesrgan; print('OK')")
	output, err := cmd.Output()
	if err != nil {
		return false
	}

	return strings.TrimSpace(string(output)) == "OK"
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
	outputMemory := float64((width * u.config.Scale) * (height * u.config.Scale) * 3 * dtypeSize)

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
func (u *Upscaler) UpscaleVideo(inputPath, outputPath string, progressCallback ProgressCallback) (*VideoUpscalingResult, error) {
	startTime := time.Now()
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
	result.UpscaledSize = VideoSize{
		Width:  videoSize.Width * u.config.Scale,
		Height: videoSize.Height * u.config.Scale,
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
	extractCmd := exec.Command("ffmpeg", "-i", inputPath, "-y", framePattern)
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
		progressCallback(20, 100, fmt.Sprintf("Processing %d frames with AI upscaling...", actualFrameCount))
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
			progressCallback(currentProgress, 100, fmt.Sprintf("Upscaled frame %d/%d", i+1, actualFrameCount))
		}

		// Memory management: force garbage collection every batch
		if (i+1)%batchSize == 0 {
			runtime.GC()
		}
	}

	if progressCallback != nil {
		progressCallback(80, 100, "Reassembling video with upscaled frames...")
	}

	// Step 3: Extract audio from original video
	audioPath := filepath.Join(tempDir, "audio.aac")
	audioCmd := exec.Command("ffmpeg", "-i", inputPath, "-vn", "-acodec", "copy", "-y", audioPath)
	audioCmd.Run() // Don't fail if no audio track exists

	// Step 4: Reassemble video
	upscaledPattern := filepath.Join(upscaledDir, "frame_%06d.png")
	var assembleCmd *exec.Cmd

	// Check if audio file was created
	if _, err := os.Stat(audioPath); err == nil {
		// Reassemble with audio
		assembleCmd = exec.Command("ffmpeg",
			"-framerate", "30", // Use appropriate framerate
			"-i", upscaledPattern,
			"-i", audioPath,
			"-c:v", "libx264",
			"-preset", "medium",
			"-crf", "18", // High quality encoding
			"-c:a", "aac",
			"-strict", "experimental",
			"-y", outputPath)
	} else {
		// Reassemble without audio
		assembleCmd = exec.Command("ffmpeg",
			"-framerate", "30",
			"-i", upscaledPattern,
			"-c:v", "libx264",
			"-preset", "medium",
			"-crf", "18",
			"-y", outputPath)
	}

	if err := assembleCmd.Run(); err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to reassemble video: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if progressCallback != nil {
		progressCallback(95, 100, "Finalizing upscaled video...")
	}

	// Verify output file was created
	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		result.ErrorMessage = "video processing completed but output file was not created"
		return result, fmt.Errorf(result.ErrorMessage)
	}

	result.ProcessingTime = time.Since(startTime)
	result.Success = true

	if progressCallback != nil {
		progressCallback(100, 100, "AI upscaling completed successfully!")
	}

	return result, nil
}

// UpscaleFrame upscales a single frame using Real-ESRGAN
func (u *Upscaler) UpscaleFrame(inputPath, outputPath string) error {
	if !u.config.Enabled {
		return fmt.Errorf("upscaling is disabled")
	}

	if !u.IsAvailable() {
		return fmt.Errorf("Real-ESRGAN is not available")
	}

	// Build command arguments
	args := []string{
		u.config.ScriptPath,
		inputPath,
		outputPath,
		"--model", u.config.Model,
		"--scale", strconv.Itoa(u.config.Scale),
	}

	// Add GPU flag if enabled
	if u.config.UseGPU {
		args = append(args, "--gpu", strconv.Itoa(u.config.GPUDevice))
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
	if config.Scale < 2 || config.Scale > 4 {
		return fmt.Errorf("upscaling scale must be 2, 3, or 4")
	}

	// Validate GPU settings
	if config.UseGPU && config.GPUDevice < 0 {
		return fmt.Errorf("invalid GPU device ID: %d", config.GPUDevice)
	}

	return nil
}

// GetDefaultConfig returns default upscaling configuration
func GetDefaultConfig() *UpscalingConfig {
	// Auto-detect Python path
	pythonPath := DetectPythonPath()
	if pythonPath == "" {
		pythonPath = "python3" // Fallback
	}

	return &UpscalingConfig{
		Enabled:    false,
		Model:      "general_4x",
		Scale:      4,
		UseGPU:     false,
		GPUDevice:  0,
		PythonPath: pythonPath,
		ScriptPath: "scripts/upscale_frame.py",
		TempDir:    os.TempDir(),
	}
}
