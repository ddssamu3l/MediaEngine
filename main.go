// main.go
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"mediaengine/internal/ffmpeg"
	"mediaengine/internal/ui"
	"mediaengine/internal/upscaling"
	"mediaengine/internal/validation"
	"mediaengine/internal/video"

	"github.com/charmbracelet/lipgloss"
	"github.com/manifoldco/promptui"
)

const (
	MaxFileSizeMB    = 500
	MaxFileSizeBytes = MaxFileSizeMB * 1024 * 1024
)

// Supported formats
var (
	SupportedInputFormats  = []string{".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv"}
	SupportedOutputFormats = []string{"GIF", "APNG", "WebP", "AVIF", "MP4", "WebM"}

	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#7C3AED")).
			MarginBottom(1)

	promptStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#06B6D4")).
			Bold(true)

	errorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#EF4444")).
			Bold(true)

	successStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#10B981")).
			Bold(true)

	warningStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#F59E0B")).
			Bold(true)

	infoStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3B82F6"))
)

type ConversionConfig struct {
	InputPath      string
	OutputPath     string
	OutputFormat   string
	StartTime      float64
	EndTime        float64
	FrameRate      int
	Resolution     string
	Quality        int
	QualityProfile string
	// Add upscaling configuration
	UseUpscaling   bool
	UpscalingModel string
	UpscalingScale int
}

func main() {
	fmt.Println(titleStyle.Render("🎬 Universal Media Engine"))
	fmt.Println("Professional video-to-media conversion with real-time progress tracking\n")

	// Check if FFmpeg and FFprobe are available
	fmt.Print("🔍 Checking system requirements... ")
	if !ffmpeg.IsFFmpegAvailable() {
		fmt.Println("❌")
		fmt.Println(errorStyle.Render("❌ FFmpeg or FFprobe is not installed or not in PATH"))
		fmt.Println("Please install both FFmpeg and FFprobe:")
		fmt.Println("\nInstallation instructions:")
		fmt.Println("• macOS: brew install ffmpeg")
		fmt.Println("• Ubuntu/Debian: sudo apt install ffmpeg")
		fmt.Println("• Windows: Download from https://ffmpeg.org/download.html")
		fmt.Println("\nNote: Both FFmpeg and FFprobe are required for full functionality.")
		os.Exit(1)
	}
	fmt.Println("✅")

	config := &ConversionConfig{}

	// Get and validate input video path with comprehensive validation
	config.InputPath = getInputPathWithValidation()

	// Get and display video information
	videoInfo, err := video.GetVideoInfo(config.InputPath)
	if err != nil {
		fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Error reading video: %v", err)))
		os.Exit(1)
	}

	ui.DisplayVideoInfo(videoInfo)

	// Validate file size with warning
	if videoInfo.FileSize > MaxFileSizeBytes {
		fmt.Println(warningStyle.Render(fmt.Sprintf("⚠️  Warning: File size (%.1f MB) exceeds recommended limit of %d MB",
			float64(videoInfo.FileSize)/(1024*1024), MaxFileSizeMB)))
		if !confirmProceed("Do you want to proceed anyway? (y/N): ", false) {
			fmt.Println("Operation cancelled.")
			os.Exit(0)
		}
	}

	// Get conversion parameters
	config.OutputFormat = getOutputFormat()
	config.QualityProfile = getQualityProfile()
	config.StartTime = getStartTime(videoInfo.Duration)
	config.EndTime = getEndTime(videoInfo.Duration, config.StartTime)
	config.FrameRate = getFrameRate()
	config.Resolution = getResolution()

	// Add upscaling preferences
	config.UseUpscaling, config.UpscalingModel, config.UpscalingScale = getUpscalingPreferences()

	config.Quality = getQuality(config.OutputFormat, config.QualityProfile)
	config.OutputPath = getOutputPath(config.OutputFormat)

	// Show conversion summary
	showConversionSummary(config, videoInfo)
	if !confirmProceed("Proceed with conversion? (Y/n): ", true) {
		fmt.Println("Operation cancelled.")
		os.Exit(0)
	}

	// Perform conversion with progress tracking
	fmt.Println("\n" + promptStyle.Render("🔄 Starting conversion..."))

	var conversionInputPath = config.InputPath

	// AI Upscaling phase (if enabled)
	if config.UseUpscaling {
		fmt.Println("🚀 AI Upscaling enabled - this will take longer but provide better quality")

		// Set up upscaling configuration
		upscalingConfig := upscaling.GetDefaultConfig()
		upscalingConfig.Enabled = true
		upscalingConfig.Model = config.UpscalingModel
		upscalingConfig.Scale = config.UpscalingScale

		// Validate upscaling configuration
		if err := upscaling.ValidateConfig(upscalingConfig); err != nil {
			fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Upscaling configuration error: %v", err)))
			fmt.Println("⚠️  Continuing with standard conversion...")
		} else {
			// Create upscaler
			upscaler := upscaling.NewUpscaler(upscalingConfig)

			// Create temporary upscaled video path
			tempDir := os.TempDir()
			upscaledVideoPath := filepath.Join(tempDir, fmt.Sprintf("upscaled_%d.mp4", time.Now().Unix()))

			// Ensure cleanup of temporary file
			defer func() {
				if _, err := os.Stat(upscaledVideoPath); err == nil {
					if removeErr := os.Remove(upscaledVideoPath); removeErr != nil {
						fmt.Printf("Warning: failed to clean up temporary file %s: %v\n", upscaledVideoPath, removeErr)
					}
				}
			}()

			// Create progress callback for upscaling
			progressCallback := func(current, total int, message string) {
				if current == total {
					fmt.Printf("\r✅ %s\n", message)
				} else {
					fmt.Printf("\r🔄 [%d%%] %s", current, message)
				}
			}

			// Perform AI upscaling
			fmt.Println("🎯 Starting AI upscaling process...")
			upscalingResult, err := upscaler.UpscaleVideo(config.InputPath, upscaledVideoPath, progressCallback)

			if err != nil {
				fmt.Println(errorStyle.Render(fmt.Sprintf("❌ AI upscaling failed: %v", err)))
				fmt.Println("⚠️  Falling back to standard conversion...")
			} else if upscalingResult.Success {
				fmt.Println(successStyle.Render("✅ AI upscaling completed successfully!"))
				fmt.Printf("📈 Upscaled from %dx%d to %dx%d (%d frames processed in %v)\n",
					upscalingResult.OriginalSize.Width, upscalingResult.OriginalSize.Height,
					upscalingResult.UpscaledSize.Width, upscalingResult.UpscaledSize.Height,
					upscalingResult.FramesProcessed, upscalingResult.ProcessingTime.Round(time.Second))

				// Use upscaled video as input for format conversion
				conversionInputPath = upscaledVideoPath
				fmt.Println("🔄 Converting upscaled video to final format...")
			} else {
				fmt.Println(errorStyle.Render("❌ AI upscaling completed but was not successful"))
				fmt.Println("⚠️  Falling back to standard conversion...")
			}
		}
	}

	// Regular format conversion phase
	err = ffmpeg.ConvertMediaWithProgress(conversionInputPath, config.OutputPath, config.OutputFormat,
		config.StartTime, config.EndTime, config.FrameRate, config.Quality, config.Resolution, config.QualityProfile)

	if err != nil {
		fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Conversion failed: %v", err)))

		// Provide helpful suggestions based on error type
		if strings.Contains(err.Error(), "encoder") {
			fmt.Println("\n💡 Suggestions:")
			fmt.Println("• Check that your FFmpeg installation supports the selected output format")
			fmt.Println("• Try a different output format")
			fmt.Println("• Update FFmpeg to the latest version")
		}

		os.Exit(1)
	}

	fmt.Println(successStyle.Render("✅ Conversion completed successfully!"))
	fmt.Printf("📁 %s saved to: %s\n", config.OutputFormat, config.OutputPath)

	// Show output file info
	if stat, err := os.Stat(config.OutputPath); err == nil {
		fmt.Printf("📊 Output size: %s\n", ui.FormatFileSize(stat.Size()))

		// Calculate compression ratio
		if videoInfo.FileSize > 0 {
			ratio := float64(stat.Size()) / float64(videoInfo.FileSize)
			if ratio < 1.0 {
				fmt.Printf("📈 Compression: %.1f%% of original size\n", ratio*100)
			} else {
				fmt.Printf("📈 Size change: %.1f%% of original size\n", ratio*100)
			}
		}
	}
}

func getInputPathWithValidation() string {
	prompt := promptui.Prompt{
		Label: "📁 Enter video file path",
		Validate: func(input string) error {
			return validation.ValidateInputPath(input)
		},
	}

	result, err := prompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	// Clean the input path
	cleanedPath := cleanInputPath(result)

	// Perform comprehensive validation
	fmt.Print("🔍 Validating file content... ")
	validationResult, err := validation.ValidateInputPathComprehensive(cleanedPath)
	if err != nil {
		fmt.Println("❌")
		fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Validation failed: %v", err)))
		os.Exit(1)
	}
	fmt.Println("✅")

	// Display warnings if any
	if len(validationResult.Warnings) > 0 {
		fmt.Println(warningStyle.Render("⚠️  Warnings:"))
		for _, warning := range validationResult.Warnings {
			fmt.Printf("  • %s\n", warning)
		}
		fmt.Println()
	}

	// Display file information
	if validationResult.ActualFormat != "" {
		fmt.Printf("📋 Detected format: %s\n", infoStyle.Render(validationResult.ActualFormat))
	}
	if validationResult.Codec != "" {
		fmt.Printf("🎞️  Video codec: %s\n", infoStyle.Render(validationResult.Codec))
	}
	if validationResult.Duration > 0 {
		fmt.Printf("⏱️  Duration: %s\n", infoStyle.Render(ui.FormatDuration(validationResult.Duration)))
	}

	return cleanedPath
}

func cleanInputPath(input string) string {
	// Clean the input: trim whitespace and remove surrounding quotes
	cleanedPath := strings.TrimSpace(input)

	// Remove surrounding quotes (single or double) that Finder might add
	if len(cleanedPath) >= 2 {
		if (cleanedPath[0] == '\'' && cleanedPath[len(cleanedPath)-1] == '\'') ||
			(cleanedPath[0] == '"' && cleanedPath[len(cleanedPath)-1] == '"') {
			cleanedPath = cleanedPath[1 : len(cleanedPath)-1]
		}
	}

	// Trim again after removing quotes
	cleanedPath = strings.TrimSpace(cleanedPath)

	// Convert to absolute path and clean it
	absPath, err := filepath.Abs(cleanedPath)
	if err != nil {
		fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Invalid path: %v", err)))
		os.Exit(1)
	}

	return filepath.Clean(absPath)
}

func getStartTime(duration float64) float64 {
	durationStr := ui.FormatDuration(duration)
	prompt := promptui.Prompt{
		Label:   fmt.Sprintf("⏰ Enter start time in seconds (0 - %.2f) [%s]", duration, durationStr),
		Default: "0",
		Validate: func(input string) error {
			if strings.TrimSpace(input) == "" {
				return nil // Allow empty for default
			}
			value, err := strconv.ParseFloat(input, 64)
			if err != nil {
				return fmt.Errorf("invalid number format")
			}
			if value < 0 {
				return fmt.Errorf("start time cannot be negative")
			}
			if value >= duration {
				return fmt.Errorf("start time must be less than video duration (%.2f seconds)", duration)
			}
			return nil
		},
	}

	result, err := prompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	if strings.TrimSpace(result) == "" {
		return 0
	}

	startTime, _ := strconv.ParseFloat(result, 64)
	return startTime
}

func getEndTime(duration, startTime float64) float64 {
	durationStr := ui.FormatDuration(duration)
	prompt := promptui.Prompt{
		Label:   fmt.Sprintf("⏰ Enter end time in seconds (%.2f - %.2f) [%s]", startTime, duration, durationStr),
		Default: fmt.Sprintf("%.2f", duration),
		Validate: func(input string) error {
			if strings.TrimSpace(input) == "" {
				return nil // Allow empty for default
			}
			value, err := strconv.ParseFloat(input, 64)
			if err != nil {
				return fmt.Errorf("invalid number format")
			}
			if value <= startTime {
				return fmt.Errorf("end time must be greater than start time (%.2f)", startTime)
			}
			if value > duration {
				return fmt.Errorf("end time cannot exceed video duration (%.2f seconds)", duration)
			}
			return nil
		},
	}

	result, err := prompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	if strings.TrimSpace(result) == "" {
		return duration
	}

	endTime, _ := strconv.ParseFloat(result, 64)
	return endTime
}

func getFrameRate() int {
	prompt := promptui.Prompt{
		Label:   "🎞️  Enter frame rate (1-60 fps)",
		Default: "15",
		Validate: func(input string) error {
			if strings.TrimSpace(input) == "" {
				return nil // Allow empty for default
			}
			value, err := strconv.Atoi(input)
			if err != nil {
				return fmt.Errorf("invalid number format")
			}
			if value < 1 || value > 60 {
				return fmt.Errorf("frame rate must be between 1 and 60 fps")
			}
			return nil
		},
	}

	result, err := prompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	if strings.TrimSpace(result) == "" {
		return 15
	}

	frameRate, _ := strconv.Atoi(result)
	return frameRate
}

func getResolution() string {
	resolutions := []string{
		"320x240   (QVGA)",
		"480x360   (Small)",
		"640x480   (VGA)",
		"800x600   (SVGA)",
		"1024x768  (XGA)",
		"1280x720  (HD)",
		"1920x1080 (Full HD)",
		"2560x1440 (QHD)",
		"3840x2160 (4K UHD)",
		"Original  (Keep original resolution)",
	}

	prompt := promptui.Select{
		Label:        "📐 Select output resolution",
		Items:        resolutions,
		Size:         10,
		HideSelected: true,
		Templates: &promptui.SelectTemplates{
			Label:    "{{ . }}",
			Active:   "▶ {{ . | cyan | bold }}",
			Inactive: "  {{ . | faint }}",
			Selected: "{{ . | green | bold }}",
		},
	}

	index, _, err := prompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	// Extract resolution from the selected option
	resolutionMappings := []string{
		"320x240",
		"480x360",
		"640x480",
		"800x600",
		"1024x768",
		"1280x720",
		"1920x1080",
		"2560x1440",
		"3840x2160",
		"Original",
	}

	return resolutionMappings[index]
}

// Add new function to get upscaling preferences
func getUpscalingPreferences() (bool, string, int) {
	// Check if upscaling is available
	upscaler := upscaling.NewUpscaler(upscaling.GetDefaultConfig())
	if !upscaler.IsAvailable() {
		fmt.Println(warningStyle.Render("⚠️  AI Upscaling not available (Real-ESRGAN not installed)"))
		return false, "", 0
	}

	// Ask if user wants to use upscaling
	useUpscaling := confirmProceed("🚀 Enable AI upscaling? (y/N): ", false)
	if !useUpscaling {
		return false, "", 0
	}

	// Get upscaling model
	models := []string{
		"General Purpose 4x (Best for photos/real content)",
		"General Purpose 2x (Faster, good quality)",
		"Anime/Cartoon 4x (Optimized for animated content)",
	}

	prompt := promptui.Select{
		Label:        "🎯 Select AI upscaling model",
		Items:        models,
		Size:         3,
		HideSelected: true,
		Templates: &promptui.SelectTemplates{
			Label:    "{{ . }}",
			Active:   "▶ {{ . | cyan | bold }}",
			Inactive: "  {{ . | faint }}",
			Selected: "{{ . | green | bold }}",
		},
	}

	index, _, err := prompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	modelMappings := []string{"general_4x", "general_2x", "anime_4x"}
	scaleMappings := []int{4, 2, 4}

	return true, modelMappings[index], scaleMappings[index]
}

func getOutputFormat() string {
	formats := []string{
		"GIF       (Graphics Interchange Format - Animated)",
		"APNG      (Animated PNG - Lossless)",
		"WebP      (Google WebP - Modern, Efficient)",
		"AVIF      (AV1 Image Format - Next-gen)",
		"MP4       (MPEG-4 Video - Universal)",
		"WebM      (WebM Video - Web Optimized)",
	}

	prompt := promptui.Select{
		Label:        "🎯 Select output format",
		Items:        formats,
		Size:         6,
		HideSelected: true,
		Templates: &promptui.SelectTemplates{
			Label:    "{{ . }}",
			Active:   "▶ {{ . | cyan | bold }}",
			Inactive: "  {{ . | faint }}",
			Selected: "{{ . | green | bold }}",
		},
	}

	index, _, err := prompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	return SupportedOutputFormats[index]
}

func getQualityProfile() string {
	profiles := []string{
		"High Quality  (Best visual quality, larger files)",
		"Balanced      (Good quality, moderate file size)",
		"Small Size    (Optimized for smaller files)",
		"Custom        (Set quality manually)",
	}

	prompt := promptui.Select{
		Label:        "⚡ Select quality profile",
		Items:        profiles,
		Size:         4,
		HideSelected: true,
		Templates: &promptui.SelectTemplates{
			Label:    "{{ . }}",
			Active:   "▶ {{ . | cyan | bold }}",
			Inactive: "  {{ . | faint }}",
			Selected: "{{ . | green | bold }}",
		},
	}

	index, _, err := prompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	profileMappings := []string{"high", "balanced", "small", "custom"}
	return profileMappings[index]
}

func getQuality(outputFormat, qualityProfile string) int {
	// If using a predefined profile, return 0 (will be handled by the profile system)
	if qualityProfile != "custom" {
		return 0
	}

	// Different quality ranges for different formats
	var maxQuality int
	var defaultQuality int
	var qualityDesc string

	switch outputFormat {
	case "GIF":
		maxQuality = 256
		defaultQuality = 128
		qualityDesc = "GIF palette colors (16-256, higher = better quality)"
	case "APNG":
		return 0 // APNG doesn't use quality setting
	case "WebP":
		maxQuality = 100
		defaultQuality = 80
		qualityDesc = "WebP quality (0-100, higher is better)"
	case "AVIF":
		maxQuality = 63
		defaultQuality = 30
		qualityDesc = "AVIF CRF (0-63, lower is better quality)"
	case "MP4", "WebM":
		maxQuality = 51
		defaultQuality = 23
		qualityDesc = "Video CRF (0-51, lower is better quality)"
	default:
		return 0
	}

	prompt := promptui.Prompt{
		Label:   fmt.Sprintf("⚡ %s", qualityDesc),
		Default: fmt.Sprintf("%d", defaultQuality),
		Validate: func(input string) error {
			if strings.TrimSpace(input) == "" {
				return nil // Allow empty for default
			}
			value, err := strconv.Atoi(input)
			if err != nil {
				return fmt.Errorf("invalid number format")
			}
			if outputFormat == "GIF" {
				if value < 16 || value > maxQuality {
					return fmt.Errorf("GIF colors must be between 16 and %d", maxQuality)
				}
			} else {
				if value < 0 || value > maxQuality {
					return fmt.Errorf("quality must be between 0 and %d", maxQuality)
				}
			}
			return nil
		},
	}

	result, err := prompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	if strings.TrimSpace(result) == "" {
		return defaultQuality
	}

	quality, _ := strconv.Atoi(result)
	return quality
}

func getOutputPath(outputFormat string) string {
	// Get file extension based on format
	var ext string
	switch outputFormat {
	case "GIF":
		ext = ".gif"
	case "APNG":
		ext = ".png"
	case "WebP":
		ext = ".webp"
	case "AVIF":
		ext = ".avif"
	case "MP4":
		ext = ".mp4"
	case "WebM":
		ext = ".webm"
	default:
		ext = ".gif"
	}

	prompt := promptui.Prompt{
		Label: fmt.Sprintf("💾 Enter output %s path (press Enter for current directory)", outputFormat),
		Validate: func(input string) error {
			if strings.TrimSpace(input) == "" {
				return nil // Allow empty for default (current directory)
			}
			return validation.ValidateOutputPath(input)
		},
	}

	result, err := prompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	// Handle empty input (default to current directory)
	if strings.TrimSpace(result) == "" {
		cwd, _ := os.Getwd()
		return filepath.Join(cwd, "output"+ext)
	}

	// Process the path and ensure correct extension
	processedPath := processOutputPath(result)

	// Ensure correct extension
	currentExt := strings.ToLower(filepath.Ext(processedPath))
	if currentExt != ext {
		// Remove current extension if it exists and add correct one
		if currentExt != "" {
			processedPath = strings.TrimSuffix(processedPath, currentExt)
		}
		processedPath += ext
	}

	return processedPath
}

// processOutputPath handles path cleaning, quote removal, and extension addition
func processOutputPath(input string) string {
	// Clean the input: trim whitespace and remove surrounding quotes
	cleanedPath := strings.TrimSpace(input)

	// Remove surrounding quotes (single or double) that Finder might add
	if len(cleanedPath) >= 2 {
		if (cleanedPath[0] == '\'' && cleanedPath[len(cleanedPath)-1] == '\'') ||
			(cleanedPath[0] == '"' && cleanedPath[len(cleanedPath)-1] == '"') {
			cleanedPath = cleanedPath[1 : len(cleanedPath)-1]
		}
	}

	// Trim again after removing quotes
	cleanedPath = strings.TrimSpace(cleanedPath)

	// Convert to absolute path
	absPath, err := filepath.Abs(cleanedPath)
	if err != nil {
		// Return the original if we can't convert to absolute path
		// The validation will catch this error
		return cleanedPath
	}

	// Check if the path exists and is a directory - if so, append output filename
	if stat, err := os.Stat(absPath); err == nil && stat.IsDir() {
		return filepath.Join(absPath, "output")
	}

	return filepath.Clean(absPath)
}

func confirmProceed(message string, defaultYes bool) bool {
	defaultValue := "n"
	if defaultYes {
		defaultValue = "y"
	}

	prompt := promptui.Prompt{
		Label:   message,
		Default: defaultValue,
		Validate: func(input string) error {
			if strings.TrimSpace(input) == "" {
				return nil // Allow empty for default
			}
			lower := strings.ToLower(strings.TrimSpace(input))
			if lower == "y" || lower == "yes" || lower == "n" || lower == "no" {
				return nil
			}
			return fmt.Errorf("please enter y/yes or n/no")
		},
	}

	result, err := prompt.Run()
	if err != nil {
		return false
	}

	// Handle empty input (use the specified default)
	if strings.TrimSpace(result) == "" {
		return defaultYes
	}

	lower := strings.ToLower(strings.TrimSpace(result))
	return lower == "y" || lower == "yes"
}

func showConversionSummary(config *ConversionConfig, videoInfo *video.VideoInfo) {
	fmt.Println("\n" + promptStyle.Render("📋 Conversion Summary:"))

	duration := config.EndTime - config.StartTime

	fmt.Printf("• Input: %s\n", filepath.Base(config.InputPath))
	fmt.Printf("• Output: %s (%s)\n", filepath.Base(config.OutputPath), config.OutputFormat)
	fmt.Printf("• Clip duration: %s (%.2f - %.2f seconds) \n",
		ui.FormatDuration(duration), config.StartTime, config.EndTime)
	fmt.Printf("• Frame rate: %d fps\n", config.FrameRate)
	fmt.Printf("• Resolution: %s\n", config.Resolution)

	// Add upscaling info
	if config.UseUpscaling {
		fmt.Printf("• AI Upscaling: %s (%dx)\n",
			upscaling.UpscalingModelDescriptions[config.UpscalingModel],
			config.UpscalingScale)
		fmt.Printf("• Final resolution will be %dx larger\n", config.UpscalingScale)
	}

	// Display quality profile
	if profile, exists := ffmpeg.QualityProfiles[config.QualityProfile]; exists {
		fmt.Printf("• Quality profile: %s (%s)\n", profile.Name, profile.Description)
	} else if config.QualityProfile == "custom" && config.Quality > 0 {
		fmt.Printf("• Custom quality: %d\n", config.Quality)
	}

	estimatedFrames := int(duration * float64(config.FrameRate))
	fmt.Printf("• Estimated frames: %d\n", estimatedFrames)

	// Show format-specific optimizations
	switch config.OutputFormat {
	case "MP4":
		fmt.Printf("• Codec: H.264 with web optimization\n")
	case "WebM":
		fmt.Printf("• Codec: VP9 with advanced compression\n")
	case "AVIF":
		fmt.Printf("• Codec: AV1 with next-generation compression\n")
	case "WebP":
		fmt.Printf("• Codec: WebP with animation support\n")
	case "GIF":
		fmt.Printf("• Optimization: Advanced palette generation\n")
	case "APNG":
		fmt.Printf("• Format: Lossless animated PNG\n")
	}
}
