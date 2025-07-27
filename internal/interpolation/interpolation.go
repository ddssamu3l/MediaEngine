// internal/interpolation/interpolation.go
package interpolation

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"mediaengine/internal/upscaling"
)

// InterpolationConfig holds configuration for AI frame interpolation
type InterpolationConfig struct {
	Enabled         bool
	Model           string // RIFE model version (v4.6, v4.4, etc.)
	TargetFPS       int    // Target frame rate (60, 120, etc.)
	OriginalFPS     int    // Original video frame rate
	Multiplier      int    // Frame rate multiplier (2x, 3x, 4x)
	UseGPU          bool
	GPUDevice       int
	PythonPath      string
	ScriptPath      string
	TempDir         string
	QualityLevel    string // "fast", "balanced", "high_quality"
	MaxBatchSize    int    // Maximum batch size (will be optimized based on GPU)
	UseParallel     bool   // Enable parallel processing
	AsyncProcessing bool   // Enable async CPU/GPU overlap
	GPUMemoryLimit  float64 // GPU memory limit in GB (0 = auto)
}

// InterpolationResult contains the results of frame interpolation
type InterpolationResult struct {
	InputPath              string
	OutputPath             string
	OriginalFPS            int
	InterpolatedFPS        int
	OriginalFrameCount     int
	InterpolatedFrameCount int
	ProcessingTime         time.Duration
	FramesProcessed        int
	Success                bool
	ErrorMessage           string
	GPUUtilization         float64
	EffectiveFPS           float64 // Actual processing FPS achieved
	BatchSizeUsed          int     // Actual batch size used
	MemoryUsage            map[string]any // GPU memory usage stats
	ParallelProcessing     bool    // Whether parallel processing was used
}

// ProgressCallback is called during interpolation processing to report progress
type ProgressCallback func(current, total int, message string)

// InterpolationModels defines available RIFE models
var InterpolationModels = map[string]string{
	"rife_v4.6":      "v4.6 (Highest Quality, Slower)",
	"rife_v4.4":      "v4.4 (Balanced Quality/Speed)",
	"rife_v4.0":      "v4.0 (Fastest, Lower Quality)",
	"rife_v4.6_lite": "v4.6 Lite (Optimized for Diffusion Videos)",
}

// QualityProfiles defines processing quality levels with aggressive GPU utilization
var QualityProfiles = map[string]QualityProfile{
	"fast": {
		Name:        "Maximum Speed",
		Description: "Maximum GPU utilization, optimized for speed",
		BatchSize:   64,  // Aggressive batch size
		TileSize:    256,
		Precision:   "fp16",
	},
	"balanced": {
		Name:        "Balanced Performance",
		Description: "High GPU utilization with good quality",
		BatchSize:   32,  // High batch size
		TileSize:    512,
		Precision:   "fp16",
	},
	"high_quality": {
		Name:        "Maximum Quality",
		Description: "Best quality with parallel processing",
		BatchSize:   16,  // Still high for parallel processing
		TileSize:    1024,
		Precision:   "fp32",
	},
}

// QualityProfile represents different processing quality levels
type QualityProfile struct {
	Name        string
	Description string
	BatchSize   int
	TileSize    int
	Precision   string
}

// Interpolator handles AI frame interpolation operations
type Interpolator struct {
	config *InterpolationConfig
}

// NewInterpolator creates a new interpolator instance
func NewInterpolator(config *InterpolationConfig) *Interpolator {
	return &Interpolator{config: config}
}

// IsAvailable checks if RIFE frame interpolation is available
func (i *Interpolator) IsAvailable() bool {
	if i.config.PythonPath == "" {
		return false
	}

	// Check if Python environment exists
	if _, err := os.Stat(i.config.PythonPath); os.IsNotExist(err) {
		return false
	}

	// Check if script exists
	if _, err := os.Stat(i.config.ScriptPath); os.IsNotExist(err) {
		return false
	}

	// Test if RIFE dependencies are available
	cmd := exec.Command(i.config.PythonPath, "-c", "import torch; import cv2; import numpy; print('OK')")
	output, err := cmd.Output()
	if err != nil {
		return false
	}

	// Check if RIFE models directory exists
	if _, err := os.Stat("models/rife"); os.IsNotExist(err) {
		return false
	}

	// Check if at least one RIFE model exists
	modelFiles := []string{
		"models/rife/flownet.pkl",
		"models/rife/metric.pkl", 
		"models/rife/featnet.pkl",
		"models/rife/fusionnet.pkl",
	}
	
	foundModel := false
	for _, modelPath := range modelFiles {
		if _, err := os.Stat(modelPath); err == nil {
			foundModel = true
			break
		}
	}

	return foundModel && strings.Contains(string(output), "OK")
}

// EstimateProcessingTime estimates how long parallel frame interpolation will take
func (i *Interpolator) EstimateProcessingTime(frameCount int, width, height int) (time.Duration, error) {
	// Get GPU info for accurate estimation
	gpuInfo, err := upscaling.GetDetailedGPUInfo(i.config.PythonPath)
	if err != nil {
		// Fallback estimation for CPU (very slow)
		interpolatedFrames := frameCount * i.config.Multiplier
		return time.Duration(interpolatedFrames * 2) * time.Second, nil
	}

	// Calculate optimal batch size based on GPU capabilities
	optimalBatchSize := i.calculateOptimalBatchSize(gpuInfo, width, height)
	
	// Base processing time per batch (aggressive parallel processing)
	var baseBatchTime time.Duration
	
	switch gpuInfo.DeviceType {
	case "CUDA":
		if gpuInfo.TotalMemoryGB >= 24 { // RTX 4090, A100, etc.
			baseBatchTime = 200 * time.Millisecond // Very fast with large batches
		} else if gpuInfo.TotalMemoryGB >= 16 { // RTX 4080, 3090, etc.
			baseBatchTime = 300 * time.Millisecond
		} else if gpuInfo.TotalMemoryGB >= 12 { // RTX 4070 Ti, 3080, etc.
			baseBatchTime = 400 * time.Millisecond
		} else if gpuInfo.TotalMemoryGB >= 8 { // RTX 4060 Ti, 3070, etc.
			baseBatchTime = 600 * time.Millisecond
		} else { // Lower-end GPUs
			baseBatchTime = 1000 * time.Millisecond
		}
	case "MPS": // Apple Silicon
		if strings.Contains(gpuInfo.GPUName, "M3") || strings.Contains(gpuInfo.GPUName, "M4") {
			baseBatchTime = 400 * time.Millisecond // M3/M4 with unified memory
		} else {
			baseBatchTime = 600 * time.Millisecond // M1/M2
		}
	default: // CPU
		baseBatchTime = 10 * time.Second // Very slow
		optimalBatchSize = 1 // No batching on CPU
	}

	// Adjust for resolution complexity
	resolutionFactor := float64(width*height) / (1920 * 1080)
	if resolutionFactor < 0.25 {
		resolutionFactor = 0.25
	}

	// Adjust for precision (parallel processing reduces impact)
	precisionFactor := 1.0
	if profile, exists := QualityProfiles[i.config.QualityLevel]; exists {
		switch profile.Precision {
		case "fp32":
			precisionFactor = 1.2 // Less impact with parallel processing
		case "fp16":
			precisionFactor = 1.0
		}
	}

	// Calculate total interpolations needed
	interpolatedFrames := frameCount * (i.config.Multiplier - 1)
	
	// Calculate number of batches needed
	numBatches := (interpolatedFrames + optimalBatchSize - 1) / optimalBatchSize
	
	// Total estimated time with parallel processing efficiency
	parallelEfficiency := 0.85 // 85% efficiency due to parallel overhead
	totalTime := time.Duration(float64(numBatches) * float64(baseBatchTime) * resolutionFactor * precisionFactor / parallelEfficiency)
	
	return totalTime, nil
}

// calculateOptimalBatchSize determines the best batch size for maximum GPU utilization
func (i *Interpolator) calculateOptimalBatchSize(gpuInfo *upscaling.GPUInfo, width, height int) int {
	if i.config.MaxBatchSize <= 0 {
		i.config.MaxBatchSize = 64 // Default maximum
	}
	
	// Memory per frame in GB (RGB24 + processing overhead)
	frameMemoryGB := float64(width * height * 3 * 4) / (1024 * 1024 * 1024) // 4 bytes per pixel for processing
	
	// Available memory for batching (leave 20% headroom)
	availableMemory := gpuInfo.AvailableMemoryGB * 0.8
	
	// Calculate memory-based batch size
	memoryBatchSize := int(availableMemory / (frameMemoryGB * 4)) // 4x overhead for interpolation
	
	// GPU-specific optimization
	var optimalBatchSize int
	switch gpuInfo.DeviceType {
	case "CUDA":
		if gpuInfo.TotalMemoryGB >= 24 {
			optimalBatchSize = min(64, memoryBatchSize) // High-end GPUs
		} else if gpuInfo.TotalMemoryGB >= 16 {
			optimalBatchSize = min(48, memoryBatchSize)
		} else if gpuInfo.TotalMemoryGB >= 12 {
			optimalBatchSize = min(32, memoryBatchSize)
		} else if gpuInfo.TotalMemoryGB >= 8 {
			optimalBatchSize = min(24, memoryBatchSize)
		} else {
			optimalBatchSize = min(16, memoryBatchSize)
		}
	case "MPS":
		// Apple Silicon unified memory allows aggressive batching
		optimalBatchSize = min(32, memoryBatchSize)
	default:
		optimalBatchSize = min(4, memoryBatchSize) // CPU limitation
	}
	
	// Apply user-defined maximum
	optimalBatchSize = min(optimalBatchSize, i.config.MaxBatchSize)
	
	// Ensure minimum batch size
	optimalBatchSize = max(optimalBatchSize, 1)
	
	return optimalBatchSize
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// InterpolateVideo processes an entire video with AI frame interpolation with comprehensive error handling
func (i *Interpolator) InterpolateVideo(inputPath, outputPath string, startTime, endTime float64, progressCallback ProgressCallback) (*InterpolationResult, error) {
	processingStartTime := time.Now()
	result := &InterpolationResult{
		InputPath:       inputPath,
		OutputPath:      outputPath,
		OriginalFPS:     i.config.OriginalFPS,
		InterpolatedFPS: i.config.TargetFPS,
		Success:         false,
	}

	if !i.config.Enabled {
		result.ErrorMessage = "frame interpolation is disabled"
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if !i.IsAvailable() {
		result.ErrorMessage = "RIFE frame interpolation is not available"
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Validate frame rate multiplier
	if i.config.TargetFPS <= i.config.OriginalFPS {
		result.ErrorMessage = fmt.Sprintf("target FPS (%d) must be higher than original FPS (%d)", 
			i.config.TargetFPS, i.config.OriginalFPS)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Calculate multiplier
	multiplier := float64(i.config.TargetFPS) / float64(i.config.OriginalFPS)
	if multiplier < 2.0 || multiplier > 4.0 {
		result.ErrorMessage = fmt.Sprintf("frame rate multiplier (%.1fx) must be between 2x and 4x", multiplier)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Get video information
	videoSize, originalFrameCount, err := i.getVideoInfo(inputPath, startTime, endTime)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to analyze video: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	result.OriginalFrameCount = originalFrameCount
	result.InterpolatedFrameCount = int(float64(originalFrameCount) * multiplier)

	// Check if GPU can handle the interpolation
	canHandle, message, err := upscaling.CanHandleInterpolation(
		i.config.PythonPath, videoSize.Width, videoSize.Height, i.config.TargetFPS, i.config.OriginalFPS)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to check GPU compatibility: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if !canHandle {
		result.ErrorMessage = fmt.Sprintf("GPU compatibility check failed: %s", message)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Create temporary directory for processing
	tempDir, err := os.MkdirTemp(i.config.TempDir, "interpolation_*")
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to create temp directory: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}
	// Enhanced cleanup with resource management
	defer func() {
		// Force cleanup of any remaining processes
		i.cleanupResources(tempDir)
		
		// Remove temporary directory
		if removeErr := os.RemoveAll(tempDir); removeErr != nil {
			fmt.Printf("Warning: failed to clean up temp directory %s: %v\n", tempDir, removeErr)
		}
	}()

	if progressCallback != nil {
		progressCallback(0, 100, "Initializing AI frame interpolation...")
	}

	// Monitor disk space requirements
	estimatedSizeGB := float64(result.OriginalFrameCount) * 0.01 // Rough estimate: 10MB per frame
	if err := i.monitorDiskSpace(tempDir, estimatedSizeGB); err != nil {
		result.ErrorMessage = fmt.Sprintf("disk space check failed: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Step 1: Extract frames from video
	framesDir := filepath.Join(tempDir, "frames")
	if err := os.MkdirAll(framesDir, 0755); err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to create frames directory: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if progressCallback != nil {
		progressCallback(10, 100, "Extracting video frames...")
	}

	if err := i.extractFrames(inputPath, framesDir, startTime, endTime, i.config.OriginalFPS); err != nil {
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
		progressCallback(20, 100, fmt.Sprintf("Processing %d frames with AI interpolation...", actualFrameCount))
	}

	// Step 2: Perform frame interpolation using RIFE
	interpolatedDir := filepath.Join(tempDir, "interpolated")
	if err := os.MkdirAll(interpolatedDir, 0755); err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to create interpolated directory: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Attempt frame interpolation with retry logic
	maxRetries := 2
	var interpolationError error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			if progressCallback != nil {
				progressCallback(20, 100, fmt.Sprintf("Retrying frame interpolation (attempt %d/%d)...", attempt+1, maxRetries+1))
			}
			// Wait before retry
			time.Sleep(2 * time.Second)
		}
		
		interpolationError = i.interpolateFrames(framesDir, interpolatedDir, progressCallback)
		if interpolationError == nil {
			break // Success
		}
		
		// Check if this is a recoverable error
		if !i.isRecoverableError(interpolationError) {
			break // Don't retry for non-recoverable errors
		}
	}
	
	if interpolationError != nil {
		result.ErrorMessage = fmt.Sprintf("failed to interpolate frames after %d attempts: %v", maxRetries+1, interpolationError)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if progressCallback != nil {
		progressCallback(70, 100, "Organizing interpolated frames...")
	}

	// Step 3: Organize interpolated frames for FFmpeg
	if err := i.OrganizeInterpolatedFrames(interpolatedDir); err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to organize interpolated frames: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if progressCallback != nil {
		progressCallback(80, 100, "Reassembling interpolated video...")
	}

	// Step 4: Extract and preserve audio
	audioPath := filepath.Join(tempDir, "audio.aac")
	audioCmd := exec.Command("ffmpeg", "-i", inputPath, 
		"-ss", fmt.Sprintf("%.3f", startTime),
		"-t", fmt.Sprintf("%.3f", endTime-startTime),
		"-vn", "-acodec", "copy", "-y", audioPath)
	audioCmd.Run() // Don't fail if no audio track exists

	// Step 4: Reassemble video with interpolated frames
	if err := i.reassembleVideo(interpolatedDir, audioPath, outputPath); err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to reassemble video: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Verify output file was created
	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		result.ErrorMessage = "interpolation completed but output file was not created"
		return result, fmt.Errorf(result.ErrorMessage)
	}

	result.ProcessingTime = time.Since(processingStartTime)
	result.Success = true

	if progressCallback != nil {
		progressCallback(100, 100, "AI frame interpolation completed successfully!")
	}

	return result, nil
}

// getVideoInfo extracts video information using ffprobe
func (i *Interpolator) getVideoInfo(videoPath string, startTime, endTime float64) (*upscaling.VideoSize, int, error) {
	// Get video dimensions
	cmd := exec.Command("ffprobe", "-v", "quiet", "-select_streams", "v:0",
		"-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", videoPath)

	output, err := cmd.Output()
	if err != nil {
		return nil, 0, fmt.Errorf("failed to get video dimensions: %v", err)
	}

	parts := strings.Split(strings.TrimSpace(string(output)), "x")
	if len(parts) < 2 {
		return nil, 0, fmt.Errorf("invalid video dimensions format")
	}

	width, err := strconv.Atoi(parts[0])
	if err != nil {
		return nil, 0, fmt.Errorf("invalid width: %v", err)
	}

	height, err := strconv.Atoi(parts[1])
	if err != nil {
		return nil, 0, fmt.Errorf("invalid height: %v", err)
	}

	// Calculate frame count for the specified time range
	duration := endTime - startTime
	frameCount := int(duration * float64(i.config.OriginalFPS))

	return &upscaling.VideoSize{Width: width, Height: height}, frameCount, nil
}

// extractFrames extracts frames from video using ffmpeg
func (i *Interpolator) extractFrames(inputPath, outputDir string, startTime, endTime float64, fps int) error {
	framePattern := filepath.Join(outputDir, "frame_%06d.png")
	
	cmd := exec.Command("ffmpeg",
		"-i", inputPath,
		"-ss", fmt.Sprintf("%.3f", startTime),
		"-t", fmt.Sprintf("%.3f", endTime-startTime),
		"-r", fmt.Sprintf("%d", fps),
		"-y", framePattern)
	
	return cmd.Run()
}

// interpolateFrames performs parallel RIFE-based frame interpolation with maximum GPU utilization
func (i *Interpolator) interpolateFrames(inputDir, outputDir string, progressCallback ProgressCallback) error {
	// Determine which script to use based on configuration
	scriptPath := i.config.ScriptPath
	if i.config.UseParallel {
		// Use the parallel processing script
		scriptPath = strings.Replace(scriptPath, "rife_interpolation.py", "rife_interpolation_parallel.py", 1)
	}

	// Build optimized RIFE command arguments for maximum GPU utilization
	args := []string{
		scriptPath,
		"--input", inputDir,
		"--output", outputDir,
		"--model", i.config.Model,
		"--multiplier", strconv.Itoa(i.config.Multiplier),
	}

	// Add GPU/CPU configuration
	if !i.config.UseGPU {
		args = append(args, "--cpu")
	} else if i.config.GPUDevice > 0 {
		args = append(args, "--gpu", strconv.Itoa(i.config.GPUDevice))
	}

	// Add aggressive batch size for maximum GPU utilization
	maxBatchSize := i.config.MaxBatchSize
	if maxBatchSize <= 0 {
		maxBatchSize = 64 // Default aggressive batch size
	}
	args = append(args, "--max-batch-size", strconv.Itoa(maxBatchSize))

	// Add quality profile settings (precision)
	if profile, exists := QualityProfiles[i.config.QualityLevel]; exists {
		args = append(args, "--precision", profile.Precision)
	}

	// Set up progress monitoring file
	progressFile := filepath.Join(i.config.TempDir, "interpolation_progress.json")
	args = append(args, "--progress-file", progressFile)

	// Set up output JSON for detailed results
	resultsFile := filepath.Join(i.config.TempDir, "interpolation_results.json")
	args = append(args, "--output-json", resultsFile)

	// Execute parallel RIFE interpolation
	cmd := exec.Command(i.config.PythonPath, args...)
	
	// Set up real-time progress monitoring
	if progressCallback != nil {
		go i.monitorProgressFromFile(progressFile, progressCallback)
	}

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("parallel RIFE interpolation failed: %v\nOutput: %s", err, string(output))
	}

	return nil
}

// monitorProgressFromFile monitors interpolation progress from JSON file updates
func (i *Interpolator) monitorProgressFromFile(progressFile string, progressCallback ProgressCallback) {
	for {
		if _, err := os.Stat(progressFile); err == nil {
			data, err := os.ReadFile(progressFile)
			if err == nil {
				var progress struct {
					Progress int    `json:"progress"`
					Message  string `json:"message"`
				}
				if json.Unmarshal(data, &progress) == nil {
					progressCallback(progress.Progress, 100, progress.Message)
					if progress.Progress >= 100 {
						break
					}
				}
			}
		}
		time.Sleep(500 * time.Millisecond) // Check every 500ms for responsive updates
	}
}

// monitorProgress monitors interpolation progress (fallback implementation)
func (i *Interpolator) monitorProgress(progressCallback ProgressCallback) {
	// Fallback progress monitor for non-parallel processing
	for progress := 20; progress < 80; progress += 5 {
		time.Sleep(2 * time.Second)
		progressCallback(progress, 100, fmt.Sprintf("Interpolating frames... %d%%", progress))
	}
}

// reassembleVideo combines interpolated frames back into a video
func (i *Interpolator) reassembleVideo(framesDir, audioPath, outputPath string) error {
	framePattern := filepath.Join(framesDir, "frame_%06d.png")
	
	var cmd *exec.Cmd
	
	// Check if audio file exists
	if _, err := os.Stat(audioPath); err == nil {
		// Reassemble with audio
		cmd = exec.Command("ffmpeg",
			"-framerate", strconv.Itoa(i.config.TargetFPS),
			"-i", framePattern,
			"-i", audioPath,
			"-c:v", "libx264",
			"-preset", "medium",
			"-crf", "18",
			"-pix_fmt", "yuv420p",
			"-c:a", "aac",
			"-movflags", "+faststart",
			"-y", outputPath)
	} else {
		// Reassemble without audio
		cmd = exec.Command("ffmpeg",
			"-framerate", strconv.Itoa(i.config.TargetFPS),
			"-i", framePattern,
			"-c:v", "libx264",
			"-preset", "medium",
			"-crf", "18",
			"-pix_fmt", "yuv420p",
			"-movflags", "+faststart",
			"-y", outputPath)
	}

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to reassemble video: %v\nOutput: %s", err, string(output))
	}

	return nil
}

// ValidateConfig validates interpolation configuration
func ValidateConfig(config *InterpolationConfig) error {
	if !config.Enabled {
		return nil
	}

	// Check Python path
	if config.PythonPath == "" {
		return fmt.Errorf("python path is required for frame interpolation")
	}

	baseCmd := strings.Fields(config.PythonPath)[0]
	if _, err := exec.LookPath(baseCmd); err != nil {
		return fmt.Errorf("python executable not found: %s", config.PythonPath)
	}

	// Check script path
	if config.ScriptPath == "" {
		return fmt.Errorf("interpolation script path is required")
	}

	if _, err := os.Stat(config.ScriptPath); os.IsNotExist(err) {
		return fmt.Errorf("interpolation script not found: %s", config.ScriptPath)
	}

	// Validate model
	if _, exists := InterpolationModels[config.Model]; !exists {
		return fmt.Errorf("invalid interpolation model: %s", config.Model)
	}

	// Validate frame rates
	if config.OriginalFPS <= 0 || config.TargetFPS <= 0 {
		return fmt.Errorf("frame rates must be positive")
	}

	if config.TargetFPS <= config.OriginalFPS {
		return fmt.Errorf("target FPS (%d) must be higher than original FPS (%d)", 
			config.TargetFPS, config.OriginalFPS)
	}

	// Validate multiplier
	multiplier := float64(config.TargetFPS) / float64(config.OriginalFPS)
	if multiplier < 2.0 || multiplier > 4.0 {
		return fmt.Errorf("frame rate multiplier (%.1fx) must be between 2x and 4x", multiplier)
	}

	// Validate quality level
	if _, exists := QualityProfiles[config.QualityLevel]; !exists {
		return fmt.Errorf("invalid quality level: %s", config.QualityLevel)
	}

	// Validate GPU settings
	if config.UseGPU && config.GPUDevice < 0 {
		return fmt.Errorf("invalid GPU device ID: %d", config.GPUDevice)
	}

	return nil
}

// GetDefaultConfig returns default interpolation configuration with aggressive GPU utilization
func GetDefaultConfig() *InterpolationConfig {
	// Use the same Python detection logic as upscaling
	venvPython := "./rife_env/bin/python"
	if runtime.GOOS == "windows" {
		venvPython = "./rife_env/Scripts/python.exe"
	}

	var pythonPath string
	if _, err := os.Stat(venvPython); err == nil {
		pythonPath = venvPython
	} else {
		// Fallback to system Python
		pythonPath = upscaling.DetectPythonPath()
		if pythonPath == "" {
			pythonPath = "python3"
		}
	}

	return &InterpolationConfig{
		Enabled:         false,
		Model:           "rife_v4.6",
		TargetFPS:       60,
		OriginalFPS:     30,
		Multiplier:      2,
		UseGPU:          true,
		GPUDevice:       0,
		PythonPath:      pythonPath,
		ScriptPath:      "scripts/rife_interpolation_parallel.py", // Use parallel script by default
		TempDir:         os.TempDir(),
		QualityLevel:    "balanced", // High GPU utilization balanced profile
		MaxBatchSize:    16,         // Conservative batch size for MPS compatibility
		UseParallel:     true,       // Enable parallel processing by default
		AsyncProcessing: true,       // Enable async CPU/GPU overlap
		GPUMemoryLimit:  0,          // Auto-detect GPU memory limit
	}
}

// isRecoverableError determines if an interpolation error can be retried
func (i *Interpolator) isRecoverableError(err error) bool {
	if err == nil {
		return false
	}
	
	errorMsg := strings.ToLower(err.Error())
	
	// Recoverable errors (temporary issues)
	recoverablePatterns := []string{
		"out of memory",
		"cuda out of memory", 
		"connection reset",
		"temporary failure",
		"device busy",
		"timeout",
		"interrupted",
	}
	
	for _, pattern := range recoverablePatterns {
		if strings.Contains(errorMsg, pattern) {
			return true
		}
	}
	
	// Non-recoverable errors (fundamental issues)
	nonRecoverablePatterns := []string{
		"model not found",
		"invalid model",
		"unsupported format",
		"permission denied",
		"no such file",
		"invalid argument",
		"configuration error",
	}
	
	for _, pattern := range nonRecoverablePatterns {
		if strings.Contains(errorMsg, pattern) {
			return false
		}
	}
	
	// Default to recoverable for unknown errors
	return true
}

// GetModelInfo returns information about available interpolation models
func GetModelInfo() map[string]string {
	return InterpolationModels
}

// cleanupResources performs comprehensive cleanup of interpolation resources
func (i *Interpolator) cleanupResources(tempDir string) {
	// Kill any hanging Python processes related to interpolation
	if runtime.GOOS != "windows" {
		// Unix-like systems
		exec.Command("pkill", "-f", "rife_interpolation").Run()
		exec.Command("pkill", "-f", "python.*interpolation").Run()
	}
	
	// Clean up any CUDA/GPU memory if using GPU
	if i.config.UseGPU {
		// Force GPU memory cleanup by running a small cleanup script
		cleanupScript := `
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()
`
		cleanupCmd := exec.Command(i.config.PythonPath, "-c", cleanupScript)
		cleanupCmd.Run() // Don't fail if this doesn't work
	}
	
	// Remove any partial/temporary files
	if tempDir != "" {
		// Remove any .tmp files
		tmpFiles, _ := filepath.Glob(filepath.Join(tempDir, "*.tmp"))
		for _, tmpFile := range tmpFiles {
			os.Remove(tmpFile)
		}
		
		// Remove any incomplete frame files
		incompleteFiles, _ := filepath.Glob(filepath.Join(tempDir, "*_incomplete*"))
		for _, incompleteFile := range incompleteFiles {
			os.Remove(incompleteFile)
		}
	}
}

// monitorDiskSpace checks available disk space during processing
func (i *Interpolator) monitorDiskSpace(workDir string, requiredGB float64) error {
	// Get disk usage information (simplified for cross-platform compatibility)
	if _, err := os.Stat(workDir); os.IsNotExist(err) {
		return fmt.Errorf("work directory does not exist: %s", workDir)
	}
	
	// This is a simplified check - in production you'd want more sophisticated disk space monitoring
	// For now, we'll just check if the directory is accessible
	testFile := filepath.Join(workDir, ".disk_test")
	if f, err := os.Create(testFile); err != nil {
		return fmt.Errorf("insufficient disk space or permissions in %s", workDir)
	} else {
		f.Close()
		os.Remove(testFile)
	}
	
	return nil
}

// OrganizeInterpolatedFrames reorganizes RIFE output frames into sequential numbering for FFmpeg
func (i *Interpolator) OrganizeInterpolatedFrames(framesDir string) error {
	// Find all frame files created by RIFE script
	pattern := filepath.Join(framesDir, "frame_*_*.png")
	frameFiles, err := filepath.Glob(pattern)
	if err != nil {
		return fmt.Errorf("failed to find interpolated frames: %v", err)
	}

	if len(frameFiles) == 0 {
		return fmt.Errorf("no interpolated frames found in %s", framesDir)
	}

	// Parse and sort frame files
	type FrameInfo struct {
		OriginalPath string
		PairIndex    int
		TimeIndex    int
		IsOriginal   bool
	}

	var frames []FrameInfo
	
	for _, filePath := range frameFiles {
		filename := filepath.Base(filePath)
		// Parse filename like "frame_000000_00.png" or "frame_000000_01.png"
		if !strings.HasPrefix(filename, "frame_") || !strings.HasSuffix(filename, ".png") {
			continue
		}
		
		// Remove "frame_" prefix and ".png" suffix
		nameWithoutExt := strings.TrimSuffix(strings.TrimPrefix(filename, "frame_"), ".png")
		parts := strings.Split(nameWithoutExt, "_")
		
		if len(parts) != 2 {
			continue
		}
		
		pairIndex, err1 := strconv.Atoi(parts[0])
		timeIndex, err2 := strconv.Atoi(parts[1])
		
		if err1 != nil || err2 != nil {
			continue
		}
		
		frames = append(frames, FrameInfo{
			OriginalPath: filePath,
			PairIndex:    pairIndex,
			TimeIndex:    timeIndex,
			IsOriginal:   timeIndex == 0,
		})
	}

	// Sort frames: first by pair index, then by time index
	// This ensures: frame_000000_00, frame_000000_01, frame_000002_00, frame_000002_01, etc.
	sort.Slice(frames, func(i, j int) bool {
		if frames[i].PairIndex == frames[j].PairIndex {
			return frames[i].TimeIndex < frames[j].TimeIndex
		}
		return frames[i].PairIndex < frames[j].PairIndex
	})

	// Rename files to sequential numbering
	for index, frame := range frames {
		sequentialName := fmt.Sprintf("frame_%06d.png", index)
		newPath := filepath.Join(framesDir, sequentialName)
		
		if err := os.Rename(frame.OriginalPath, newPath); err != nil {
			return fmt.Errorf("failed to rename %s to %s: %v", frame.OriginalPath, newPath, err)
		}
	}

	return nil
}

// GetQualityProfiles returns available quality profiles
func GetQualityProfiles() map[string]QualityProfile {
	return QualityProfiles
}