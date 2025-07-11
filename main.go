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
	// AI scaling configuration (always enabled)
	UseUpscaling   bool
	UpscalingModel string
	UpscalingScale int    // Positive for upscaling, negative for downscaling
	ScalingMode    string // "upscale" or "downscale"
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
	config.FrameRate = getFrameRate(videoInfo)
	config.Resolution = "Original" // Always keep original resolution

	// AI scaling is now enabled by default
	config.UpscalingModel, config.UpscalingScale = getScalingPreferences(videoInfo)
	if config.UpscalingScale != 0 {
		config.UseUpscaling = true
		if config.UpscalingScale > 0 {
			config.ScalingMode = "upscale"
		} else {
			config.ScalingMode = "downscale"
		}
	} else {
		config.UseUpscaling = false
		config.ScalingMode = "none"
	}

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

	// AI Scaling phase (always enabled)
	if config.UseUpscaling && config.ScalingMode != "none" {
		if config.ScalingMode == "upscale" {
			fmt.Println("🚀 AI Upscaling enabled - this will take longer but provide better quality")
		} else if config.ScalingMode == "downscale" {
			fmt.Printf("📉 AI Downscaling enabled - reducing quality by %dx\n", -config.UpscalingScale)
		}

		// Set up scaling configuration
		upscalingConfig := upscaling.GetDefaultConfig()
		upscalingConfig.Enabled = true
		upscalingConfig.Model = config.UpscalingModel
		upscalingConfig.Scale = config.UpscalingScale
		upscalingConfig.IsDownscale = config.UpscalingScale < 0

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

			// Perform AI scaling
			if config.ScalingMode == "upscale" {
				fmt.Println("🎯 Starting AI upscaling process...")
			} else {
				fmt.Println("🎯 Starting AI downscaling process...")
			}
			upscalingResult, err := upscaler.UpscaleVideo(config.InputPath, upscaledVideoPath, config.StartTime, config.EndTime, config.FrameRate, progressCallback)

			if err != nil {
				fmt.Println(errorStyle.Render(fmt.Sprintf("❌ AI %s failed: %v", config.ScalingMode, err)))
				fmt.Println("⚠️  Falling back to standard conversion...")
			} else if upscalingResult.Success {
				fmt.Println(successStyle.Render(fmt.Sprintf("✅ AI %s completed successfully!", config.ScalingMode)))
				if config.ScalingMode == "upscale" {
					fmt.Printf("📈 Upscaled from %dx%d to %dx%d (%d frames processed in %v)\n",
						upscalingResult.OriginalSize.Width, upscalingResult.OriginalSize.Height,
						upscalingResult.UpscaledSize.Width, upscalingResult.UpscaledSize.Height,
						upscalingResult.FramesProcessed, upscalingResult.ProcessingTime.Round(time.Second))
				} else {
					fmt.Printf("📉 Downscaled from %dx%d to %dx%d (%d frames processed in %v)\n",
						upscalingResult.OriginalSize.Width, upscalingResult.OriginalSize.Height,
						upscalingResult.UpscaledSize.Width, upscalingResult.UpscaledSize.Height,
						upscalingResult.FramesProcessed, upscalingResult.ProcessingTime.Round(time.Second))
				}

				// Use scaled video as input for format conversion
				conversionInputPath = upscaledVideoPath
				fmt.Println("🔄 Converting scaled video to final format...")
			} else {
				fmt.Println(errorStyle.Render(fmt.Sprintf("❌ AI %s completed but was not successful", config.ScalingMode)))
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
		Label: "📁 Source video path (relative and absolute paths are supported)",
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
		Label: fmt.Sprintf("⏰ Enter start time in seconds (0 - %.2f) [%s] (press Enter for default: 0)", duration, durationStr),
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
		Label: fmt.Sprintf("⏰ Enter end time in seconds (%.2f - %.2f) [%s] (press Enter for default: %.2f)", startTime, duration, durationStr, duration),
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

func getFrameRate(videoInfo *video.VideoInfo) int {
	originalFPS := int(videoInfo.FrameRate + 0.5) // Round to nearest integer
	if originalFPS < 1 {
		originalFPS = 30 // Fallback if frame rate detection failed
	}

	prompt := promptui.Prompt{
		Label: fmt.Sprintf("🎞️  Enter frame rate (1-%d fps) (press Enter for default: %d)", originalFPS, originalFPS),
		Validate: func(input string) error {
			if strings.TrimSpace(input) == "" {
				return nil // Allow empty for default
			}
			value, err := strconv.Atoi(input)
			if err != nil {
				return fmt.Errorf("invalid number format")
			}
			if value < 1 || value > originalFPS {
				return fmt.Errorf("frame rate must be between 1 and %d fps (original video frame rate)", originalFPS)
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
		return originalFPS
	}

	frameRate, _ := strconv.Atoi(result)
	return frameRate
}


// Get scaling preferences (upscale/downscale/none)
func getScalingPreferences(videoInfo *video.VideoInfo) (string, int) {
	// Check if AI scaling is available
	config := upscaling.GetDefaultConfig()
	upscaler := upscaling.NewUpscaler(config)
	if !upscaler.IsAvailable() {
		fmt.Println(warningStyle.Render("⚠️  AI Scaling not available (Real-ESRGAN not installed)"))
		return "", 0
	}

	// Show GPU information
	gpuInfo := upscaling.GetGPUInfo(config.PythonPath)
	fmt.Printf("💻 %s\n", infoStyle.Render(gpuInfo))

	// Show current resolution
	fmt.Printf("📐 Current video resolution: %s\n", infoStyle.Render(fmt.Sprintf("%dx%d", videoInfo.Width, videoInfo.Height)))

	// Get scaling mode
	scalingOptions := []string{
		"Keep Original Resolution (No AI processing)",
		"Upscale - Enhance quality (2x, 4x)",
		"Downscale - Reduce quality (1/2x, 1/4x, 1/8x)",
	}

	prompt := promptui.Select{
		Label:        "🎯 Select AI scaling mode",
		Items:        scalingOptions,
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

	switch index {
	case 0: // Keep original
		return "", 0
	case 1: // Upscale
		return getUpscaleOptions()
	case 2: // Downscale
		return getDownscaleOptions()
	}

	return "", 0
}

// Get upscaling options
func getUpscaleOptions() (string, int) {
	models := []string{
		"General Purpose 4x (Best for photos/real content)",
		"General Purpose 2x (Faster, good quality)",
		"Anime/Cartoon 4x (Optimized for animated content)",
	}

	prompt := promptui.Select{
		Label:        "🚀 Select upscaling model",
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

	return modelMappings[index], scaleMappings[index]
}

// Get downscaling options
func getDownscaleOptions() (string, int) {
	downscaleOptions := []string{
		"Reduce by 2x (1/2 original size)",
		"Reduce by 4x (1/4 original size)",
		"Reduce by 8x (1/8 original size)",
	}

	prompt := promptui.Select{
		Label:        "📉 Select downscaling magnitude",
		Items:        downscaleOptions,
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

	// Return negative values to indicate downscaling
	scaleMappings := []int{-2, -4, -8}
	
	// For downscaling, we'll use the general_2x model as it's faster
	return "general_2x", scaleMappings[index]
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
		Label: fmt.Sprintf("⚡ %s (press Enter for default: %d)", qualityDesc, defaultQuality),
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

	// Step 1: Get output directory
	currentDir := func() string {
		if cwd, err := os.Getwd(); err == nil {
			return cwd
		}
		return "."
	}()

	dirPrompt := promptui.Prompt{
		Label: fmt.Sprintf("📁 Enter output directory path (press Enter for default: %s)", currentDir),
		Validate: func(input string) error {
			if strings.TrimSpace(input) == "" {
				return nil // Allow empty for default (current directory)
			}
			return validation.ValidateOutputPath(input)
		},
	}

	dirResult, err := dirPrompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	// Handle directory selection
	var outputDir string
	if strings.TrimSpace(dirResult) == "" {
		outputDir, _ = os.Getwd()
	} else {
		outputDir = processOutputPath(dirResult)
	}

	// Step 2: Get filename
	defaultFilename := "output" + ext

	filenamePrompt := promptui.Prompt{
		Label: fmt.Sprintf("📝 Enter output filename (without the %s extension) (press Enter for default: output)", ext),
		Validate: func(input string) error {
			if strings.TrimSpace(input) == "" {
				return nil // Allow empty for default
			}
			// Basic filename validation
			filename := strings.TrimSpace(input)
			if strings.ContainsAny(filename, `<>:"/\|?*`) {
				return fmt.Errorf("filename contains invalid characters")
			}
			return nil
		},
	}

	filenameResult, err := filenamePrompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	// Handle filename
	var filename string
	if strings.TrimSpace(filenameResult) == "" {
		filename = defaultFilename
	} else {
		filename = strings.TrimSpace(filenameResult) + ext
	}

	return filepath.Join(outputDir, filename)
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

	return filepath.Clean(absPath)
}

func confirmProceed(message string, defaultYes bool) bool {
	defaultStr := "N"
	if defaultYes {
		defaultStr = "Y"
	}

	prompt := promptui.Prompt{
		Label: fmt.Sprintf("%s (press Enter for default: %s)", message, defaultStr),
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

	// Add scaling info
	if config.UseUpscaling && config.ScalingMode != "none" {
		if config.ScalingMode == "upscale" {
			fmt.Printf("• AI Upscaling: %s (%dx)\n",
				upscaling.UpscalingModelDescriptions[config.UpscalingModel],
				config.UpscalingScale)
			fmt.Printf("• Final resolution will be %dx larger\n", config.UpscalingScale)
		} else if config.ScalingMode == "downscale" {
			fmt.Printf("• AI Downscaling: Reducing by %dx\n", -config.UpscalingScale)
			fmt.Printf("• Final resolution will be %d%% of original\n", 100/(-config.UpscalingScale))
		}
		// Show GPU acceleration info
		upscalingConfig := upscaling.GetDefaultConfig()
		gpuInfo := upscaling.GetGPUInfo(upscalingConfig.PythonPath)
		fmt.Printf("• Acceleration: %s\n", gpuInfo)
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
