// main.go
package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"videotogif/internal/ffmpeg"
	"videotogif/internal/ui"
	"videotogif/internal/video"

	"github.com/charmbracelet/lipgloss"
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
		os.Exit(1)
	}

	config := &ConversionConfig{}
	scanner := bufio.NewScanner(os.Stdin)

	// Get input video path
	config.InputPath = getInputPath(scanner)

	// Get and display video information
	videoInfo, err := video.GetVideoInfo(config.InputPath)
	if err != nil {
		fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Error reading video: %v", err)))
		os.Exit(1)
	}

	ui.DisplayVideoInfo(videoInfo)

	// Get conversion parameters
	config.StartTime = getStartTime(scanner, videoInfo.Duration)
	config.EndTime = getEndTime(scanner, videoInfo.Duration, config.StartTime)
	config.FrameRate = getFrameRate(scanner)
	config.Resolution = getResolution()
	config.OutputPath = getOutputPath(scanner)

	// Perform conversion
	fmt.Println("\n" + promptStyle.Render("🔄 Converting video to GIF..."))

	err = ffmpeg.ConvertToGIF(config.InputPath, config.OutputPath,
		config.StartTime, config.EndTime, config.FrameRate, config.Resolution)

	if err != nil {
		fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Conversion failed: %v", err)))
		os.Exit(1)
	}

	fmt.Println(successStyle.Render("✅ Conversion completed successfully!"))
	fmt.Printf("GIF saved to: %s\n", config.OutputPath)
}

func getInputPath(scanner *bufio.Scanner) string {
	for {
		fmt.Print(promptStyle.Render("📁 Enter video file path: "))
		scanner.Scan()
		path := strings.TrimSpace(scanner.Text())

		if path == "" {
			fmt.Println(errorStyle.Render("❌ Path cannot be empty"))
			continue
		}

		// Convert to absolute path
		absPath, err := filepath.Abs(path)
		if err != nil {
			fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Invalid path: %v", err)))
			continue
		}

		// Check if file exists
		if _, err := os.Stat(absPath); os.IsNotExist(err) {
			fmt.Println(errorStyle.Render("❌ File does not exist"))
			continue
		}

		return absPath
	}
}

func getStartTime(scanner *bufio.Scanner, duration float64) float64 {
	for {
		fmt.Printf(promptStyle.Render("⏰ Enter start time in seconds (0 - %.2f): "), duration)
		scanner.Scan()
		input := strings.TrimSpace(scanner.Text())

		if input == "" {
			return 0
		}

		startTime, err := strconv.ParseFloat(input, 64)
		if err != nil {
			fmt.Println(errorStyle.Render("❌ Invalid number"))
			continue
		}

		if startTime < 0 || startTime >= duration {
			fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Start time must be between 0 and %.2f", duration)))
			continue
		}

		return startTime
	}
}

func getEndTime(scanner *bufio.Scanner, duration, startTime float64) float64 {
	for {
		fmt.Printf(promptStyle.Render("⏰ Enter end time in seconds (%.2f - %.2f): "), startTime, duration)
		scanner.Scan()
		input := strings.TrimSpace(scanner.Text())

		if input == "" {
			return duration
		}

		endTime, err := strconv.ParseFloat(input, 64)
		if err != nil {
			fmt.Println(errorStyle.Render("❌ Invalid number"))
			continue
		}

		if endTime <= startTime || endTime > duration {
			fmt.Println(errorStyle.Render(fmt.Sprintf("❌ End time must be between %.2f and %.2f", startTime, duration)))
			continue
		}

		return endTime
	}
}

func getFrameRate(scanner *bufio.Scanner) int {
	for {
		fmt.Print(promptStyle.Render("🎞️  Enter frame rate (1-30, default 15): "))
		scanner.Scan()
		input := strings.TrimSpace(scanner.Text())

		if input == "" {
			return 15
		}

		frameRate, err := strconv.Atoi(input)
		if err != nil {
			fmt.Println(errorStyle.Render("❌ Invalid number"))
			continue
		}

		if frameRate < 1 || frameRate > 30 {
			fmt.Println(errorStyle.Render("❌ Frame rate must be between 1 and 30"))
			continue
		}

		return frameRate
	}
}

func getResolution() string {
	resolutions := []string{
		"320x240",
		"480x360",
		"640x480",
		"800x600",
		"1024x768",
		"1280x720",
		"1920x1080",
		"Original",
	}

	fmt.Println(promptStyle.Render("📐 Select resolution:"))
	for i, res := range resolutions {
		fmt.Printf("  %d) %s\n", i+1, res)
	}

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print(promptStyle.Render("Enter choice (1-8, default 4): "))
		scanner.Scan()
		input := strings.TrimSpace(scanner.Text())

		if input == "" {
			return resolutions[3] // 800x600
		}

		choice, err := strconv.Atoi(input)
		if err != nil || choice < 1 || choice > len(resolutions) {
			fmt.Println(errorStyle.Render("❌ Invalid choice"))
			continue
		}

		return resolutions[choice-1]
	}
}

func getOutputPath(scanner *bufio.Scanner) string {
	for {
		fmt.Print(promptStyle.Render("💾 Enter output GIF path (press Enter for current directory): "))
		scanner.Scan()
		path := strings.TrimSpace(scanner.Text())

		if path == "" {
			return "output.gif"
		}

		// Convert to absolute path
		absPath, err := filepath.Abs(path)
		if err != nil {
			fmt.Println(errorStyle.Render(fmt.Sprintf("❌ Invalid path: %v", err)))
			continue
		}

		// Ensure .gif extension
		if !strings.HasSuffix(strings.ToLower(absPath), ".gif") {
			absPath += ".gif"
		}

		return absPath
	}
}
