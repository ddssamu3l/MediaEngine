// internal/ffmpeg/ffmpeg.go
package ffmpeg

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/schollz/progressbar/v3"
)

// QualityProfile represents different quality presets
type QualityProfile struct {
	Name        string
	Description string
	Settings    map[string]interface{}
}

var QualityProfiles = map[string]QualityProfile{
	"high": {
		Name:        "High Quality",
		Description: "Best quality, larger file size",
		Settings: map[string]interface{}{
			"webp_quality": 95,
			"avif_crf":     18,
			"video_crf":    18,
			"gif_colors":   256,
		},
	},
	"balanced": {
		Name:        "Balanced",
		Description: "Good quality, moderate file size",
		Settings: map[string]interface{}{
			"webp_quality": 80,
			"avif_crf":     30,
			"video_crf":    23,
			"gif_colors":   128,
		},
	},
	"small": {
		Name:        "Small Size",
		Description: "Optimized for smaller files",
		Settings: map[string]interface{}{
			"webp_quality": 65,
			"avif_crf":     45,
			"video_crf":    28,
			"gif_colors":   64,
		},
	},
}

// MediaInfo holds detailed information about media files
type MediaInfo struct {
	Format   string
	Duration float64
	Width    int
	Height   int
	Codec    string
	Valid    bool
}

// FFProbeFormat represents the format section of FFprobe output
type FFProbeFormat struct {
	Filename       string `json:"filename"`
	FormatName     string `json:"format_name"`
	FormatLongName string `json:"format_long_name"`
	Duration       string `json:"duration"`
	Size           string `json:"size"`
}

// FFProbeStream represents a stream in FFprobe output
type FFProbeStream struct {
	CodecName string `json:"codec_name"`
	CodecType string `json:"codec_type"`
	Width     int    `json:"width"`
	Height    int    `json:"height"`
}

// FFProbeOutput represents the complete FFprobe JSON output
type FFProbeOutput struct {
	Format  FFProbeFormat   `json:"format"`
	Streams []FFProbeStream `json:"streams"`
}

// ConversionProgress represents the current state of conversion
type ConversionProgress struct {
	TotalFrames   int
	CurrentFrame  int
	CurrentTime   float64
	TotalDuration float64
	Speed         string
	FPS           string
}

// FrameExtractionConfig holds configuration for frame extraction
type FrameExtractionConfig struct {
	InputPath       string
	OutputDir       string
	StartTime       float64
	EndTime         float64
	FrameRate       int
	Format          string // png, jpg, bmp
	Quality         int    // for JPEG
	PixelFormat     string // rgb24, yuv420p, etc.
	ScaleFilter     string // for resizing frames
	DeinterlaceMode string // yadif, bwdif, none
	ColorSpace      string // bt709, bt601, etc.
	Threads         int    // number of threads for extraction
}

// FrameExtractionResult contains the results of frame extraction
type FrameExtractionResult struct {
	FrameCount      int
	OutputPattern   string
	FrameFiles      []string
	ExtractionTime  float64
	AverageFileSize int64
	Success         bool
	ErrorMessage    string
}

// IsFFmpegAvailable checks if FFmpeg and FFprobe are installed and accessible
func IsFFmpegAvailable() bool {
	// Check both FFmpeg and FFprobe
	if err := exec.Command("ffmpeg", "-version").Run(); err != nil {
		return false
	}
	if err := exec.Command("ffprobe", "-version").Run(); err != nil {
		return false
	}
	return true
}

// ValidateMediaFile uses FFprobe to validate and get information about the media file
func ValidateMediaFile(inputPath string) (*MediaInfo, error) {
	cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", inputPath)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to probe media file: %v", err)
	}

	// Parse JSON output properly
	var probeOutput FFProbeOutput
	if err := json.Unmarshal(output, &probeOutput); err != nil {
		return nil, fmt.Errorf("failed to parse FFprobe output: %v", err)
	}

	info := &MediaInfo{Valid: true}

	// Extract format information
	info.Format = probeOutput.Format.FormatName

	// Parse duration
	if probeOutput.Format.Duration != "" {
		if duration, err := strconv.ParseFloat(probeOutput.Format.Duration, 64); err == nil {
			info.Duration = duration
		}
	}

	// Extract codec and dimensions from video streams
	for _, stream := range probeOutput.Streams {
		if stream.CodecType == "video" {
			info.Codec = stream.CodecName
			info.Width = stream.Width
			info.Height = stream.Height
			break // Use first video stream
		}
	}

	// Basic validation - ensure we have essential information
	if info.Duration <= 0 {
		return nil, fmt.Errorf("video has zero or invalid duration")
	}

	if info.Format == "" {
		return nil, fmt.Errorf("unable to determine video format")
	}

	return info, nil
}

// ConvertMediaWithProgress performs media conversion with real-time progress tracking
func ConvertMediaWithProgress(inputPath, outputPath, outputFormat string, startTime, endTime float64, frameRate int, quality int, resolution string, profile string) error {
	// Validate input file first
	mediaInfo, err := ValidateMediaFile(inputPath)
	if err != nil {
		return fmt.Errorf("input validation failed: %v", err)
	}

	// Apply quality profile if specified
	if profile != "" && profile != "custom" {
		if profileSettings, exists := QualityProfiles[profile]; exists {
			quality = applyQualityProfile(outputFormat, profileSettings, quality)
		}
	}

	// Build FFmpeg command with optimized parameters
	args := buildFFmpegArgs(inputPath, outputPath, outputFormat, startTime, endTime, frameRate, quality, resolution, mediaInfo)

	// Create context for cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cmd := exec.CommandContext(ctx, "ffmpeg", args...)

	// Set up pipes for progress tracking
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %v", err)
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start FFmpeg: %v", err)
	}

	// Calculate expected duration for progress tracking
	duration := endTime - startTime
	if duration <= 0 {
		duration = mediaInfo.Duration
	}

	// Create progress bar
	bar := progressbar.NewOptions(int(duration*float64(frameRate)),
		progressbar.OptionSetDescription(fmt.Sprintf("Converting to %s", outputFormat)),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "█",
			SaucerHead:    "█",
			SaucerPadding: "░",
			BarStart:      "[",
			BarEnd:        "]",
		}),
		progressbar.OptionShowCount(),
		progressbar.OptionShowIts(),
		progressbar.OptionSetItsString("fps"),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionFullWidth(),
		progressbar.OptionSetRenderBlankState(true),
	)

	// Parse FFmpeg output for progress
	scanner := bufio.NewScanner(stderr)
	var lastError string

	go func() {
		defer bar.Finish()

		for scanner.Scan() {
			line := scanner.Text()

			// Capture errors
			if strings.Contains(line, "Error") || strings.Contains(line, "error") {
				lastError = line
			}

			// Parse progress information
			if progress := parseProgress(line, duration); progress != nil {
				// Update progress bar
				targetFrame := int(progress.CurrentTime * float64(frameRate))
				if targetFrame <= int(duration*float64(frameRate)) {
					bar.Set(targetFrame)
				}
			}
		}
	}()

	// Wait for command to complete
	if err := cmd.Wait(); err != nil {
		if lastError != "" {
			return fmt.Errorf("conversion failed: %s", parseFFmpegError(lastError, outputFormat))
		}
		return fmt.Errorf("conversion failed: %v", err)
	}

	bar.Finish()

	// Validate output file was created successfully
	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		return fmt.Errorf("conversion completed but output file was not created")
	}

	return nil
}

// ConvertMedia maintains backward compatibility while using the new progress-enabled function
func ConvertMedia(inputPath, outputPath, outputFormat string, startTime, endTime float64, frameRate int, quality int, resolution string) error {
	return ConvertMediaWithProgress(inputPath, outputPath, outputFormat, startTime, endTime, frameRate, quality, resolution, "balanced")
}

// buildFFmpegArgs constructs optimized FFmpeg arguments for each format
func buildFFmpegArgs(inputPath, outputPath, outputFormat string, startTime, endTime float64, frameRate int, quality int, resolution string, mediaInfo *MediaInfo) []string {
	args := []string{
		"-y", // overwrite output
		"-i", inputPath,
		"-ss", fmt.Sprintf("%.2f", startTime),
		"-t", fmt.Sprintf("%.2f", endTime-startTime),
		"-progress", "pipe:2", // Enable progress output
	}

	switch strings.ToLower(outputFormat) {
	case "gif":
		args = append(args, buildGIFArgs(frameRate, resolution, quality)...)
	case "apng":
		args = append(args, buildAPNGArgs(frameRate, resolution)...)
	case "webp":
		args = append(args, buildWebPArgs(frameRate, resolution, quality)...)
	case "avif":
		args = append(args, buildAVIFArgs(frameRate, resolution, quality)...)
	case "mp4":
		args = append(args, buildMP4Args(frameRate, resolution, quality)...)
	case "webm":
		args = append(args, buildWebMArgs(frameRate, resolution, quality)...)
	default:
		return nil
	}

	args = append(args, outputPath)
	return args
}

func buildGIFArgs(frameRate int, resolution string, quality int) []string {
	args := []string{"-f", "gif"}

	// Build video filter for GIF with palette optimization
	var filters []string
	if resolution != "Original" {
		filters = append(filters, fmt.Sprintf("scale=%s", resolution))
	}
	filters = append(filters, fmt.Sprintf("fps=%d", frameRate))

	// Advanced GIF palette optimization
	colors := 256
	if quality > 0 && quality < 256 {
		colors = quality
	}

	filters = append(filters, fmt.Sprintf("split[s0][s1];[s0]palettegen=max_colors=%d[p];[s1][p]paletteuse=dither=bayer", colors))

	if len(filters) > 0 {
		args = append(args, "-vf", strings.Join(filters, ","))
	}

	return args
}

func buildAPNGArgs(frameRate int, resolution string) []string {
	args := []string{"-f", "apng", "-plays", "0"}

	var filters []string
	if resolution != "Original" {
		filters = append(filters, fmt.Sprintf("scale=%s", resolution))
	}
	filters = append(filters, fmt.Sprintf("fps=%d", frameRate))

	if len(filters) > 0 {
		args = append(args, "-vf", strings.Join(filters, ","))
	}

	return args
}

func buildWebPArgs(frameRate int, resolution string, quality int) []string {
	args := []string{"-c:v", "libwebp", "-f", "webp"}

	// WebP quality settings
	if quality > 0 && quality <= 100 {
		args = append(args, "-quality", strconv.Itoa(quality))
	}

	// WebP animation settings
	args = append(args, "-loop", "0", "-preset", "default", "-an")

	var filters []string
	if resolution != "Original" {
		filters = append(filters, fmt.Sprintf("scale=%s", resolution))
	}
	filters = append(filters, fmt.Sprintf("fps=%d", frameRate))

	if len(filters) > 0 {
		args = append(args, "-vf", strings.Join(filters, ","))
	}

	return args
}

func buildAVIFArgs(frameRate int, resolution string, quality int) []string {
	args := []string{"-c:v", "libaom-av1", "-f", "avif"}

	// AVIF quality settings (CRF)
	if quality > 0 && quality <= 63 {
		args = append(args, "-crf", strconv.Itoa(quality))
	}

	// AVIF optimization
	args = append(args, "-cpu-used", "8", "-row-mt", "1", "-tiles", "2x2")

	var filters []string
	if resolution != "Original" {
		filters = append(filters, fmt.Sprintf("scale=%s", resolution))
	}
	filters = append(filters, fmt.Sprintf("fps=%d", frameRate))

	if len(filters) > 0 {
		args = append(args, "-vf", strings.Join(filters, ","))
	}

	return args
}

func buildMP4Args(frameRate int, resolution string, quality int) []string {
	args := []string{"-c:v", "libx264", "-f", "mp4"}

	// MP4 quality settings (CRF)
	if quality > 0 && quality <= 51 {
		args = append(args, "-crf", strconv.Itoa(quality))
	}

	// H.264 optimization
	args = append(args, "-preset", "fast", "-profile:v", "high", "-level", "4.0")
	args = append(args, "-movflags", "+faststart") // Web optimization
	args = append(args, "-r", strconv.Itoa(frameRate))

	if resolution != "Original" {
		args = append(args, "-vf", fmt.Sprintf("scale=%s", resolution))
	}

	return args
}

func buildWebMArgs(frameRate int, resolution string, quality int) []string {
	args := []string{"-c:v", "libvpx-vp9", "-f", "webm"}

	// WebM quality settings (CRF)
	if quality > 0 && quality <= 51 {
		args = append(args, "-crf", strconv.Itoa(quality))
	}

	// VP9 optimization
	args = append(args, "-b:v", "0", "-deadline", "good", "-cpu-used", "1")
	args = append(args, "-row-mt", "1", "-tile-columns", "2")
	args = append(args, "-r", strconv.Itoa(frameRate))

	if resolution != "Original" {
		args = append(args, "-vf", fmt.Sprintf("scale=%s", resolution))
	}

	return args
}

// parseProgress extracts progress information from FFmpeg output
func parseProgress(line string, totalDuration float64) *ConversionProgress {
	// FFmpeg progress format: frame=  123 fps= 25 q=28.0 size=    1024kB time=00:00:05.00 bitrate=1677.7kbits/s speed=   1x
	if !strings.Contains(line, "time=") {
		return nil
	}

	progress := &ConversionProgress{}

	// Extract current time
	if timeMatch := regexp.MustCompile(`time=(\d{2}):(\d{2}):(\d{2}\.\d{2})`).FindStringSubmatch(line); len(timeMatch) == 4 {
		hours, _ := strconv.Atoi(timeMatch[1])
		minutes, _ := strconv.Atoi(timeMatch[2])
		seconds, _ := strconv.ParseFloat(timeMatch[3], 64)
		progress.CurrentTime = float64(hours*3600+minutes*60) + seconds
	}

	// Extract frame count
	if frameMatch := regexp.MustCompile(`frame=\s*(\d+)`).FindStringSubmatch(line); len(frameMatch) == 2 {
		frame, _ := strconv.Atoi(frameMatch[1])
		progress.CurrentFrame = frame
	}

	// Extract speed
	if speedMatch := regexp.MustCompile(`speed=\s*([^\s]+)`).FindStringSubmatch(line); len(speedMatch) == 2 {
		progress.Speed = speedMatch[1]
	}

	// Extract FPS
	if fpsMatch := regexp.MustCompile(`fps=\s*([^\s]+)`).FindStringSubmatch(line); len(fpsMatch) == 2 {
		progress.FPS = fpsMatch[1]
	}

	progress.TotalDuration = totalDuration

	return progress
}

// parseFFmpegError provides user-friendly error messages for common FFmpeg issues
func parseFFmpegError(errorLine, outputFormat string) string {
	lowerError := strings.ToLower(errorLine)

	// Common codec issues
	if strings.Contains(lowerError, "unknown encoder") {
		return fmt.Sprintf("The %s format encoder is not available. Please check your FFmpeg installation.", outputFormat)
	}

	if strings.Contains(lowerError, "no such file") {
		return "Input file not found or cannot be accessed"
	}

	if strings.Contains(lowerError, "permission denied") {
		return "Permission denied - check file permissions for input/output locations"
	}

	if strings.Contains(lowerError, "invalid data") {
		return "Input file appears to be corrupted or in an unsupported format"
	}

	if strings.Contains(lowerError, "disk full") {
		return "Not enough disk space to complete conversion"
	}

	// Format-specific issues
	switch strings.ToLower(outputFormat) {
	case "avif":
		if strings.Contains(lowerError, "libaom") {
			return "AVIF encoder (libaom-av1) not available. Please install FFmpeg with AV1 support."
		}
	case "webp":
		if strings.Contains(lowerError, "libwebp") {
			return "WebP encoder not available. Please install FFmpeg with WebP support."
		}
	}

	return fmt.Sprintf("Conversion error: %s", errorLine)
}

// applyQualityProfile applies quality profile settings to the given quality value
func applyQualityProfile(outputFormat string, profile QualityProfile, defaultQuality int) int {
	switch strings.ToLower(outputFormat) {
	case "webp":
		if val, ok := profile.Settings["webp_quality"].(int); ok {
			return val
		}
	case "avif":
		if val, ok := profile.Settings["avif_crf"].(int); ok {
			return val
		}
	case "mp4", "webm":
		if val, ok := profile.Settings["video_crf"].(int); ok {
			return val
		}
	case "gif":
		if val, ok := profile.Settings["gif_colors"].(int); ok {
			return val
		}
	}

	return defaultQuality
}

// Legacy function for backward compatibility
func ConvertToGIF(inputPath, outputPath string, startTime, endTime float64, frameRate int, resolution string) error {
	return ConvertMedia(inputPath, outputPath, "GIF", startTime, endTime, frameRate, 0, resolution)
}

// ExtractFramesForInterpolation extracts frames optimized for AI frame interpolation
func ExtractFramesForInterpolation(config *FrameExtractionConfig, progressCallback func(int, int, string)) (*FrameExtractionResult, error) {
	result := &FrameExtractionResult{
		Success: false,
	}

	// Validate configuration
	if err := validateFrameExtractionConfig(config); err != nil {
		result.ErrorMessage = fmt.Sprintf("invalid configuration: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Create output directory
	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to create output directory: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Build output pattern
	framePattern := filepath.Join(config.OutputDir, "frame_%06d."+config.Format)
	result.OutputPattern = framePattern

	// Build FFmpeg arguments for optimal frame extraction
	args := buildFrameExtractionArgs(config, framePattern)

	if progressCallback != nil {
		progressCallback(0, 100, "Starting frame extraction...")
	}

	// Execute FFmpeg command
	startTime := time.Now()
	cmd := exec.Command("ffmpeg", args...)
	
	// Set up progress monitoring
	stderr, err := cmd.StderrPipe()
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to create stderr pipe: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if err := cmd.Start(); err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to start FFmpeg: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Monitor progress
	go func() {
		scanner := bufio.NewScanner(stderr)
		duration := config.EndTime - config.StartTime
		
		for scanner.Scan() {
			line := scanner.Text()
			if progressCallback != nil {
				// Parse FFmpeg progress output
				if progress := parseExtractionProgress(line, duration); progress >= 0 {
					progressCallback(progress, 100, fmt.Sprintf("Extracting frames... %d%%", progress))
				}
			}
		}
	}()

	if err := cmd.Wait(); err != nil {
		result.ErrorMessage = fmt.Sprintf("frame extraction failed: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	result.ExtractionTime = time.Since(startTime).Seconds()

	// Count and validate extracted frames
	frameFiles, err := filepath.Glob(framePattern)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to list extracted frames: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if len(frameFiles) == 0 {
		result.ErrorMessage = "no frames were extracted"
		return result, fmt.Errorf(result.ErrorMessage)
	}

	result.FrameCount = len(frameFiles)
	result.FrameFiles = frameFiles

	// Calculate average file size
	var totalSize int64
	for _, frameFile := range frameFiles {
		if stat, err := os.Stat(frameFile); err == nil {
			totalSize += stat.Size()
		}
	}
	result.AverageFileSize = totalSize / int64(len(frameFiles))

	result.Success = true

	if progressCallback != nil {
		progressCallback(100, 100, fmt.Sprintf("Extracted %d frames successfully", result.FrameCount))
	}

	return result, nil
}

// buildFrameExtractionArgs builds optimized FFmpeg arguments for frame extraction
func buildFrameExtractionArgs(config *FrameExtractionConfig, outputPattern string) []string {
	args := []string{
		"-i", config.InputPath,
		"-ss", fmt.Sprintf("%.3f", config.StartTime),
		"-t", fmt.Sprintf("%.3f", config.EndTime-config.StartTime),
	}

	// Frame rate control
	if config.FrameRate > 0 {
		args = append(args, "-r", strconv.Itoa(config.FrameRate))
	}

	// Build video filter chain
	var filters []string

	// Deinterlacing filter
	if config.DeinterlaceMode != "none" && config.DeinterlaceMode != "" {
		switch config.DeinterlaceMode {
		case "yadif":
			filters = append(filters, "yadif=0:-1:0") // Field rate, auto field order
		case "bwdif":
			filters = append(filters, "bwdif=0:-1:0") // Better quality deinterlacer
		}
	}

	// Scaling filter
	if config.ScaleFilter != "" {
		filters = append(filters, config.ScaleFilter)
	}

	// Color space conversion
	if config.ColorSpace != "" {
		filters = append(filters, fmt.Sprintf("colorspace=%s", config.ColorSpace))
	}

	// Apply video filters if any
	if len(filters) > 0 {
		args = append(args, "-vf", strings.Join(filters, ","))
	}

	// Pixel format for AI processing (RGB24 is optimal for most AI models)
	if config.PixelFormat != "" {
		args = append(args, "-pix_fmt", config.PixelFormat)
	} else {
		args = append(args, "-pix_fmt", "rgb24") // Default for AI processing
	}

	// Format-specific settings
	switch strings.ToLower(config.Format) {
	case "png":
		args = append(args, "-c:v", "png")
		args = append(args, "-compression_level", "1") // Fast compression for speed
	case "jpg", "jpeg":
		args = append(args, "-c:v", "mjpeg")
		if config.Quality > 0 {
			args = append(args, "-q:v", strconv.Itoa(config.Quality))
		} else {
			args = append(args, "-q:v", "2") // High quality for AI processing
		}
	case "bmp":
		args = append(args, "-c:v", "bmp")
	}

	// Threading for performance
	if config.Threads > 0 {
		args = append(args, "-threads", strconv.Itoa(config.Threads))
	} else {
		args = append(args, "-threads", "0") // Use all available threads
	}

	// Disable audio and subtitle streams
	args = append(args, "-an", "-sn")

	// Force overwrite and hide banner
	args = append(args, "-y", "-hide_banner", "-loglevel", "info")

	// Output pattern
	args = append(args, outputPattern)

	return args
}

// validateFrameExtractionConfig validates the frame extraction configuration
func validateFrameExtractionConfig(config *FrameExtractionConfig) error {
	if config.InputPath == "" {
		return fmt.Errorf("input path cannot be empty")
	}

	if config.OutputDir == "" {
		return fmt.Errorf("output directory cannot be empty")
	}

	if config.StartTime < 0 {
		return fmt.Errorf("start time cannot be negative")
	}

	if config.EndTime <= config.StartTime {
		return fmt.Errorf("end time must be greater than start time")
	}

	if config.FrameRate <= 0 {
		return fmt.Errorf("frame rate must be positive")
	}

	validFormats := map[string]bool{"png": true, "jpg": true, "jpeg": true, "bmp": true}
	if !validFormats[strings.ToLower(config.Format)] {
		return fmt.Errorf("unsupported format: %s (supported: png, jpg, bmp)", config.Format)
	}

	return nil
}

// parseExtractionProgress parses FFmpeg progress output for frame extraction
func parseExtractionProgress(line string, totalDuration float64) int {
	// Look for time progress in FFmpeg output
	timeRegex := regexp.MustCompile(`time=(\d+):(\d+):(\d+\.\d+)`)
	matches := timeRegex.FindStringSubmatch(line)
	
	if len(matches) == 4 {
		hours, _ := strconv.Atoi(matches[1])
		minutes, _ := strconv.Atoi(matches[2])
		seconds, _ := strconv.ParseFloat(matches[3], 64)
		
		currentTime := float64(hours*3600) + float64(minutes*60) + seconds
		if totalDuration > 0 {
			progress := int((currentTime / totalDuration) * 100)
			if progress > 100 {
				progress = 100
			}
			return progress
		}
	}
	
	return -1 // No progress information found
}

// GetOptimalFrameExtractionConfig returns optimized frame extraction configuration
func GetOptimalFrameExtractionConfig(inputPath, outputDir string, startTime, endTime float64, frameRate int) *FrameExtractionConfig {
	return &FrameExtractionConfig{
		InputPath:       inputPath,
		OutputDir:       outputDir,
		StartTime:       startTime,
		EndTime:         endTime,
		FrameRate:       frameRate,
		Format:          "png",           // PNG for lossless quality
		Quality:         0,               // Not used for PNG
		PixelFormat:     "rgb24",         // Optimal for AI processing
		ScaleFilter:     "",              // No scaling by default
		DeinterlaceMode: "yadif",         // Standard deinterlacing
		ColorSpace:      "bt709",         // Standard HD color space
		Threads:         0,               // Use all available threads
	}
}

// ExtractFramesWithBatching extracts frames in batches for memory efficiency
func ExtractFramesWithBatching(config *FrameExtractionConfig, batchSize int, progressCallback func(int, int, string)) (*FrameExtractionResult, error) {
	totalDuration := config.EndTime - config.StartTime
	batchDuration := totalDuration / float64(batchSize)
	
	var allFrameFiles []string
	var totalExtractionTime float64
	var totalFrameCount int
	
	for i := 0; i < batchSize; i++ {
		batchStart := config.StartTime + float64(i)*batchDuration
		batchEnd := batchStart + batchDuration
		
		// Ensure we don't exceed the end time
		if batchEnd > config.EndTime {
			batchEnd = config.EndTime
		}
		
		// Create batch-specific configuration
		batchConfig := *config
		batchConfig.StartTime = batchStart
		batchConfig.EndTime = batchEnd
		batchConfig.OutputDir = filepath.Join(config.OutputDir, fmt.Sprintf("batch_%03d", i))
		
		// Extract frames for this batch
		batchResult, err := ExtractFramesForInterpolation(&batchConfig, func(current, total int, message string) {
			overallProgress := (i*100 + current) / batchSize
			if progressCallback != nil {
				progressCallback(overallProgress, 100, fmt.Sprintf("Batch %d/%d: %s", i+1, batchSize, message))
			}
		})
		
		if err != nil {
			return nil, fmt.Errorf("batch %d failed: %v", i, err)
		}
		
		allFrameFiles = append(allFrameFiles, batchResult.FrameFiles...)
		totalExtractionTime += batchResult.ExtractionTime
		totalFrameCount += batchResult.FrameCount
	}
	
	// Create combined result
	result := &FrameExtractionResult{
		FrameCount:      totalFrameCount,
		OutputPattern:   filepath.Join(config.OutputDir, "*/frame_*."+config.Format),
		FrameFiles:      allFrameFiles,
		ExtractionTime:  totalExtractionTime,
		Success:         true,
	}
	
	// Calculate average file size
	if len(allFrameFiles) > 0 {
		var totalSize int64
		for _, frameFile := range allFrameFiles {
			if stat, err := os.Stat(frameFile); err == nil {
				totalSize += stat.Size()
			}
		}
		result.AverageFileSize = totalSize / int64(len(allFrameFiles))
	}
	
	return result, nil
}

// VideoReassemblyConfig holds configuration for video reassembly from frames
type VideoReassemblyConfig struct {
	FramesDir       string
	AudioPath       string // Optional audio file to include
	OutputPath      string
	FrameRate       int
	Width           int // Target width (0 = auto)
	Height          int // Target height (0 = auto)
	CodecSettings   VideoCodecSettings
	MetadataFile    string // Optional metadata file to preserve
	FramePattern    string // Pattern for frame files (e.g., "frame_%06d.png")
	StartFrame      int    // First frame number (default: 0)
	EndFrame        int    // Last frame number (0 = auto-detect)
	Threads         int    // Number of encoding threads
}

// VideoCodecSettings holds codec-specific settings for video reassembly
type VideoCodecSettings struct {
	Codec       string            // libx264, libx265, libvpx-vp9, etc.
	PixelFormat string            // yuv420p, yuv444p, rgb24, etc.
	CRF         int               // Constant Rate Factor (0-51 for x264/x265)
	Preset      string            // ultrafast, fast, medium, slow, veryslow
	Profile     string            // baseline, main, high
	Level       string            // 3.1, 4.0, 4.1, etc.
	Bitrate     string            // Target bitrate (e.g., "2M", "5000k")
	MaxBitrate  string            // Maximum bitrate
	BufferSize  string            // Buffer size
	GopSize     int               // GOP size (keyframe interval)
	BFrames     int               // Number of B-frames
	CustomArgs  map[string]string // Additional custom arguments
}

// VideoReassemblyResult contains the results of video reassembly
type VideoReassemblyResult struct {
	OutputPath       string
	FramesProcessed  int
	Duration         float64
	FileSize         int64
	AssemblyTime     float64
	AverageFrameRate float64
	VideoCodec       string
	AudioCodec       string
	Success          bool
	ErrorMessage     string
}

// ReassembleVideoFromFrames creates a video from interpolated frames with optimal settings
func ReassembleVideoFromFrames(config *VideoReassemblyConfig, progressCallback func(int, int, string)) (*VideoReassemblyResult, error) {
	result := &VideoReassemblyResult{
		OutputPath: config.OutputPath,
		Success:    false,
	}

	// Validate configuration
	if err := validateReassemblyConfig(config); err != nil {
		result.ErrorMessage = fmt.Sprintf("invalid configuration: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	startTime := time.Now()

	// Detect frame files and count them
	framePattern := filepath.Join(config.FramesDir, config.FramePattern)
	frameFiles, err := filepath.Glob(framePattern)
	if err != nil || len(frameFiles) == 0 {
		result.ErrorMessage = fmt.Sprintf("no frames found matching pattern: %s", framePattern)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	result.FramesProcessed = len(frameFiles)
	
	if progressCallback != nil {
		progressCallback(0, 100, "Preparing video reassembly...")
	}

	// Build FFmpeg arguments for optimal video reassembly
	args := buildReassemblyArgs(config, framePattern)

	if progressCallback != nil {
		progressCallback(10, 100, "Starting video encoding...")
	}

	// Execute FFmpeg command with progress monitoring
	cmd := exec.Command("ffmpeg", args...)
	
	// Set up progress monitoring
	stderr, err := cmd.StderrPipe()
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to create stderr pipe: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	if err := cmd.Start(); err != nil {
		result.ErrorMessage = fmt.Sprintf("failed to start FFmpeg: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Monitor progress
	go func() {
		scanner := bufio.NewScanner(stderr)
		expectedDuration := float64(result.FramesProcessed) / float64(config.FrameRate)
		
		for scanner.Scan() {
			line := scanner.Text()
			if progressCallback != nil {
				if progress := parseReassemblyProgress(line, expectedDuration); progress >= 0 {
					progressCallback(10+int(float64(progress)*0.8), 100, fmt.Sprintf("Encoding video... %d%%", progress))
				}
			}
		}
	}()

	if err := cmd.Wait(); err != nil {
		result.ErrorMessage = fmt.Sprintf("video reassembly failed: %v", err)
		return result, fmt.Errorf(result.ErrorMessage)
	}

	result.AssemblyTime = time.Since(startTime).Seconds()

	// Verify output file and get statistics
	if stat, err := os.Stat(config.OutputPath); err == nil {
		result.FileSize = stat.Size()
	} else {
		result.ErrorMessage = "video reassembly completed but output file was not created"
		return result, fmt.Errorf(result.ErrorMessage)
	}

	// Get video information
	if mediaInfo, err := ValidateMediaFile(config.OutputPath); err == nil {
		result.Duration = mediaInfo.Duration
		result.VideoCodec = mediaInfo.Codec
		result.AverageFrameRate = float64(result.FramesProcessed) / result.Duration
	}

	result.Success = true

	if progressCallback != nil {
		progressCallback(100, 100, fmt.Sprintf("Video reassembly completed: %d frames in %.2fs", 
			result.FramesProcessed, result.Duration))
	}

	return result, nil
}

// buildReassemblyArgs builds optimized FFmpeg arguments for video reassembly
func buildReassemblyArgs(config *VideoReassemblyConfig, framePattern string) []string {
	args := []string{}

	// Input frame rate and pattern
	args = append(args, "-framerate", strconv.Itoa(config.FrameRate))
	args = append(args, "-i", framePattern)

	// Add audio if specified
	if config.AudioPath != "" {
		if _, err := os.Stat(config.AudioPath); err == nil {
			args = append(args, "-i", config.AudioPath)
		}
	}

	// Video codec settings
	codec := config.CodecSettings.Codec
	if codec == "" {
		codec = "libx264" // Default to H.264
	}
	args = append(args, "-c:v", codec)

	// Pixel format
	pixFmt := config.CodecSettings.PixelFormat
	if pixFmt == "" {
		pixFmt = "yuv420p" // Default for compatibility
	}
	args = append(args, "-pix_fmt", pixFmt)

	// Quality settings
	if config.CodecSettings.CRF > 0 {
		args = append(args, "-crf", strconv.Itoa(config.CodecSettings.CRF))
	} else {
		args = append(args, "-crf", "18") // High quality default
	}

	// Preset for encoding speed vs compression
	preset := config.CodecSettings.Preset
	if preset == "" {
		preset = "medium" // Balanced default
	}
	args = append(args, "-preset", preset)

	// Profile and level
	if config.CodecSettings.Profile != "" {
		args = append(args, "-profile:v", config.CodecSettings.Profile)
	}
	if config.CodecSettings.Level != "" {
		args = append(args, "-level", config.CodecSettings.Level)
	}

	// Bitrate settings
	if config.CodecSettings.Bitrate != "" {
		args = append(args, "-b:v", config.CodecSettings.Bitrate)
	}
	if config.CodecSettings.MaxBitrate != "" {
		args = append(args, "-maxrate", config.CodecSettings.MaxBitrate)
	}
	if config.CodecSettings.BufferSize != "" {
		args = append(args, "-bufsize", config.CodecSettings.BufferSize)
	}

	// GOP size (keyframe interval)
	if config.CodecSettings.GopSize > 0 {
		args = append(args, "-g", strconv.Itoa(config.CodecSettings.GopSize))
	}

	// B-frames
	if config.CodecSettings.BFrames >= 0 {
		args = append(args, "-bf", strconv.Itoa(config.CodecSettings.BFrames))
	}

	// Custom arguments
	for key, value := range config.CodecSettings.CustomArgs {
		args = append(args, key, value)
	}

	// Audio codec (if audio is present)
	if config.AudioPath != "" {
		args = append(args, "-c:a", "aac")
		args = append(args, "-b:a", "128k") // Standard audio bitrate
	}

	// Threading
	if config.Threads > 0 {
		args = append(args, "-threads", strconv.Itoa(config.Threads))
	}

	// Output optimization
	args = append(args, "-movflags", "+faststart") // Web optimization
	args = append(args, "-avoid_negative_ts", "make_zero") // Timestamp handling

	// Force overwrite and set log level
	args = append(args, "-y", "-hide_banner", "-loglevel", "info")

	// Output file
	args = append(args, config.OutputPath)

	return args
}

// validateReassemblyConfig validates video reassembly configuration
func validateReassemblyConfig(config *VideoReassemblyConfig) error {
	if config.FramesDir == "" {
		return fmt.Errorf("frames directory cannot be empty")
	}

	if config.OutputPath == "" {
		return fmt.Errorf("output path cannot be empty")
	}

	if config.FrameRate <= 0 {
		return fmt.Errorf("frame rate must be positive")
	}

	if config.FramePattern == "" {
		config.FramePattern = "frame_%06d.png" // Default pattern
	}

	// Ensure frames directory exists
	if _, err := os.Stat(config.FramesDir); os.IsNotExist(err) {
		return fmt.Errorf("frames directory does not exist: %s", config.FramesDir)
	}

	return nil
}

// parseReassemblyProgress parses FFmpeg progress output for video reassembly
func parseReassemblyProgress(line string, expectedDuration float64) int {
	// Look for time progress in FFmpeg output
	timeRegex := regexp.MustCompile(`time=(\d+):(\d+):(\d+\.\d+)`)
	matches := timeRegex.FindStringSubmatch(line)
	
	if len(matches) == 4 {
		hours, _ := strconv.Atoi(matches[1])
		minutes, _ := strconv.Atoi(matches[2])
		seconds, _ := strconv.ParseFloat(matches[3], 64)
		
		currentTime := float64(hours*3600) + float64(minutes*60) + seconds
		if expectedDuration > 0 {
			progress := int((currentTime / expectedDuration) * 100)
			if progress > 100 {
				progress = 100
			}
			return progress
		}
	}
	
	return -1 // No progress information found
}

// GetOptimalReassemblyConfig returns optimized video reassembly configuration
func GetOptimalReassemblyConfig(framesDir, outputPath string, frameRate int, quality string) *VideoReassemblyConfig {
	codecSettings := VideoCodecSettings{
		Codec:       "libx264",
		PixelFormat: "yuv420p",
		Preset:      "medium",
		Profile:     "high",
		GopSize:     frameRate * 2, // 2-second GOP
		BFrames:     2,
	}

	// Adjust quality based on profile
	switch quality {
	case "high":
		codecSettings.CRF = 18
		codecSettings.Preset = "slow"
	case "medium", "balanced":
		codecSettings.CRF = 23
		codecSettings.Preset = "medium"
	case "low", "fast":
		codecSettings.CRF = 28
		codecSettings.Preset = "fast"
	default:
		codecSettings.CRF = 23
		codecSettings.Preset = "medium"
	}

	return &VideoReassemblyConfig{
		FramesDir:     framesDir,
		OutputPath:    outputPath,
		FrameRate:     frameRate,
		CodecSettings: codecSettings,
		FramePattern:  "frame_%06d.png",
		Threads:       0, // Use all available threads
	}
}

// ReassembleWithMetadataPreservation reassembles video while preserving metadata from original
func ReassembleWithMetadataPreservation(config *VideoReassemblyConfig, originalVideoPath string, progressCallback func(int, int, string)) (*VideoReassemblyResult, error) {
	// Extract metadata from original video
	originalInfo, err := ValidateMediaFile(originalVideoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read original video metadata: %v", err)
	}

	// Adjust configuration based on original video
	if config.Width == 0 || config.Height == 0 {
		config.Width = originalInfo.Width
		config.Height = originalInfo.Height
	}

	// Extract audio from original if not provided
	if config.AudioPath == "" {
		tempAudioPath := filepath.Join(filepath.Dir(config.OutputPath), "temp_audio.aac")
		audioCmd := exec.Command("ffmpeg", "-i", originalVideoPath, "-vn", "-acodec", "copy", "-y", tempAudioPath)
		if audioCmd.Run() == nil {
			config.AudioPath = tempAudioPath
			// Schedule cleanup
			defer func() {
				os.Remove(tempAudioPath)
			}()
		}
	}

	// Perform reassembly
	return ReassembleVideoFromFrames(config, progressCallback)
}
