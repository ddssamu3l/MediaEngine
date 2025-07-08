// main.go
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"videotogif/internal/ffmpeg"
	"videotogif/internal/ui"
	"videotogif/internal/validation"
	"videotogif/internal/video"

	"github.com/charmbracelet/lipgloss"
	"github.com/manifoldco/promptui"
)

const (
	MaxFileSizeMB    = 500
	MaxFileSizeBytes = MaxFileSizeMB * 1024 * 1024
)

var (
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
)

type ConversionConfig struct {
	InputPath  string
	OutputPath string
	StartTime  float64
	EndTime    float64
	FrameRate  int
	Resolution string
}

func main() {
	fmt.Println(titleStyle.Render("🎬 VideoToGIF Converter"))
	fmt.Println("Convert MP4 videos to GIF with ease!\n")

	// Check if FFmpeg is available
	if !ffmpeg.IsFFmpegAvailable() {
		fmt.Println(errorStyle.Render("❌ FFmpeg is not installed or not in PATH"))
		fmt.Println("Please install FFmpeg and try again.")
		fmt.Println("\nInstallation instructions:")
		fmt.Println("• macOS: brew install ffmpeg")
		fmt.Println("• Ubuntu/Debian: sudo apt install ffmpeg")
		fmt.Println("• Windows: Download from https://ffmpeg.org/download.html")
		os.Exit(1)
	}

	config := &ConversionConfig{}

	// Get and validate input video path
	config.InputPath = getInputPath()

	// Get and display video information
	videoInfo, err := video.GetVideoInfo(config.InputPath)
	if err != nil {
		fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Error reading video: %v", err)))
		os.Exit(1)
	}

	ui.DisplayVideoInfo(videoInfo)

	// Validate file size
	if videoInfo.FileSize > MaxFileSizeBytes {
		fmt.Println(warningStyle.Render(fmt.Sprintf("⚠️  Warning: File size (%.1f MB) exceeds recommended limit of %d MB",
			float64(videoInfo.FileSize)/(1024*1024), MaxFileSizeMB)))
		if !confirmProceed("Do you want to proceed anyway? (y/N): ") {
			fmt.Println("Operation cancelled.")
			os.Exit(0)
		}
	}

	// Get conversion parameters
	config.StartTime = getStartTime(videoInfo.Duration)
	config.EndTime = getEndTime(videoInfo.Duration, config.StartTime)
	config.FrameRate = getFrameRate()
	config.Resolution = getResolution()
	config.OutputPath = getOutputPath()

	// Validate output path
	if err := validation.ValidateOutputPath(config.OutputPath); err != nil {
		fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Output path error: %v", err)))
		os.Exit(1)
	}

	// Show conversion summary
	showConversionSummary(config, videoInfo)
	if !confirmProceed("Proceed with conversion? (Y/n): ") {
		fmt.Println("Operation cancelled.")
		os.Exit(0)
	}

	// Perform conversion
	fmt.Println("\n" + promptStyle.Render("🔄 Converting video to GIF..."))

	err = ffmpeg.ConvertToGIF(config.InputPath, config.OutputPath,
		config.StartTime, config.EndTime, config.FrameRate, config.Resolution)

	if err != nil {
		fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Conversion failed: %v", err)))
		os.Exit(1)
	}

	fmt.Println(successStyle.Render("✅ Conversion completed successfully!"))
	fmt.Printf("📁 GIF saved to: %s\n", config.OutputPath)

	// Show output file info
	if stat, err := os.Stat(config.OutputPath); err == nil {
		fmt.Printf("📊 Output size: %s\n", ui.FormatFileSize(stat.Size()))
	}
}

func getInputPath() string {
	prompt := promptui.Prompt{
		Label: "📁 Enter video file path",
		Validate: func(input string) error {
			return validation.ValidateInputPath(input)
		},
		Templates: &promptui.PromptTemplates{
			Prompt:  "{{ . }}",
			Valid:   "{{ . | green }}",
			Invalid: "{{ . | red }}",
			Success: "{{ . | bold }}",
		},
	}

	result, err := prompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	// Convert to absolute path and clean it
	absPath, err := filepath.Abs(strings.TrimSpace(result))
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
		Label:   "🎞️  Enter frame rate (1-30 fps)",
		Default: "15",
		Validate: func(input string) error {
			if strings.TrimSpace(input) == "" {
				return nil // Allow empty for default
			}
			value, err := strconv.Atoi(input)
			if err != nil {
				return fmt.Errorf("invalid number format")
			}
			if value < 1 || value > 30 {
				return fmt.Errorf("frame rate must be between 1 and 30 fps")
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
		"Original  (Keep original resolution)",
	}

	prompt := promptui.Select{
		Label:        "📐 Select output resolution",
		Items:        resolutions,
		Size:         8,
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
		"Original",
	}

	return resolutionMappings[index]
}

func getOutputPath() string {
	cwd, _ := os.Getwd()
	prompt := promptui.Prompt{
		Label:   "💾 Enter output GIF path",
		Default: filepath.Join(cwd, "output.gif"),
		Validate: func(input string) error {
			if strings.TrimSpace(input) == "" {
				return fmt.Errorf("output path cannot be empty")
			}
			return validation.ValidateOutputPathBasic(input)
		},
		Templates: &promptui.PromptTemplates{
			Prompt:  "{{ . }}",
			Valid:   "{{ . | green }}",
			Invalid: "{{ . | red }}",
			Success: "{{ . | bold }}",
		},
	}

	result, err := prompt.Run()
	if err != nil {
		fmt.Println(errorStyle.Render("❌ Operation cancelled"))
		os.Exit(1)
	}

	// Convert to absolute path and ensure .gif extension
	absPath, err := filepath.Abs(strings.TrimSpace(result))
	if err != nil {
		fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Invalid path: %v", err)))
		os.Exit(1)
	}

	// Ensure .gif extension
	if !strings.HasSuffix(strings.ToLower(absPath), ".gif") {
		absPath += ".gif"
	}

	return filepath.Clean(absPath)
}

func confirmProceed(message string) bool {
	prompt := promptui.Prompt{
		Label:     message,
		IsConfirm: true,
	}

	result, err := prompt.Run()
	if err != nil {
		return false
	}

	return strings.ToLower(result) == "y" || strings.ToLower(result) == "yes"
}

func showConversionSummary(config *ConversionConfig, videoInfo *video.VideoInfo) {
	fmt.Println("\n" + promptStyle.Render("📋 Conversion Summary:"))

	duration := config.EndTime - config.StartTime

	fmt.Printf("• Input: %s\n", filepath.Base(config.InputPath))
	fmt.Printf("• Output: %s\n", filepath.Base(config.OutputPath))
	fmt.Printf("• Clip duration: %s (%.2f - %.2f seconds)\n",
		ui.FormatDuration(duration), config.StartTime, config.EndTime)
	fmt.Printf("• Frame rate: %d fps\n", config.FrameRate)
	fmt.Printf("• Resolution: %s\n", config.Resolution)

	estimatedFrames := int(duration * float64(config.FrameRate))
	fmt.Printf("• Estimated frames: %d\n", estimatedFrames)
}
