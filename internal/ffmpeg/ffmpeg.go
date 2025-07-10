// internal/ffmpeg/ffmpeg.go
package ffmpeg

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"

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
