// internal/ffmpeg/ffmpeg.go
package ffmpeg

import (
	"bufio"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"

	"github.com/schollz/progressbar/v3"
)

func IsFFmpegAvailable() bool {
	_, err := exec.LookPath("ffmpeg")
	return err == nil
}

func ConvertToGIF(inputPath, outputPath string, startTime, endTime float64, frameRate int, resolution string) error {
	duration := endTime - startTime

	// Build ffmpeg command with progress reporting
	args := []string{
		"-y", // Overwrite output file
		"-i", inputPath,
		"-ss", fmt.Sprintf("%.3f", startTime),
		"-t", fmt.Sprintf("%.3f", duration),
		"-r", strconv.Itoa(frameRate),
		"-progress", "pipe:2", // Output progress to stderr
		"-loglevel", "info", // Enable info logging for progress
	}

	// Add resolution scaling if not original
	if resolution != "Original" {
		args = append(args, "-vf", fmt.Sprintf("fps=%d,scale=%s:flags=lanczos", frameRate, resolution))
	} else {
		args = append(args, "-vf", fmt.Sprintf("fps=%d", frameRate))
	}

	// Add GIF optimization parameters
	args = append(args,
		"-f", "gif",
		outputPath,
	)

	// Create progress bar
	totalFrames := int(duration * float64(frameRate))
	bar := progressbar.NewOptions(totalFrames,
		progressbar.OptionSetDescription("Converting"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "█",
			SaucerHead:    "█",
			SaucerPadding: "░",
			BarStart:      "▐",
			BarEnd:        "▌",
		}),
		progressbar.OptionShowCount(),
		progressbar.OptionShowIts(),
		progressbar.OptionSetWidth(50),
		progressbar.OptionSetRenderBlankState(true),
		progressbar.OptionSetElapsedTime(true),
		progressbar.OptionSetPredictTime(true),
	)

	// Start conversion
	cmd := exec.Command("ffmpeg", args...)

	// Capture stderr for progress parsing
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %v", err)
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start ffmpeg: %v", err)
	}

	// Parse progress in a separate goroutine
	progressDone := make(chan bool)
	go func() {
		defer close(progressDone)
		parseFFmpegProgress(bufio.NewReader(stderr), bar, totalFrames)
	}()

	// Wait for completion
	err = cmd.Wait()

	// Wait for progress parsing to complete
	<-progressDone
	bar.Finish()

	if err != nil {
		return fmt.Errorf("ffmpeg conversion failed: %v", err)
	}

	return nil
}

// parseFFmpegProgress parses FFmpeg's progress output and updates the progress bar
func parseFFmpegProgress(stderr *bufio.Reader, bar *progressbar.ProgressBar, totalFrames int) {
	scanner := bufio.NewScanner(stderr)
	frameRegex := regexp.MustCompile(`frame=\s*(\d+)`)
	timeRegex := regexp.MustCompile(`time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})`)

	var lastFrame int

	for scanner.Scan() {
		line := scanner.Text()

		// Try to extract frame number
		if matches := frameRegex.FindStringSubmatch(line); len(matches) > 1 {
			if frame, err := strconv.Atoi(matches[1]); err == nil {
				if frame > lastFrame {
					bar.Set(frame)
					lastFrame = frame
				}
			}
		}

		// Also try to extract time progress as backup
		if matches := timeRegex.FindStringSubmatch(line); len(matches) > 4 {
			hours, _ := strconv.Atoi(matches[1])
			minutes, _ := strconv.Atoi(matches[2])
			seconds, _ := strconv.Atoi(matches[3])
			centiseconds, _ := strconv.Atoi(matches[4])

			totalSeconds := float64(hours*3600+minutes*60+seconds) + float64(centiseconds)/100.0

			// Calculate approximate frame based on time (fallback method)
			if lastFrame == 0 && totalSeconds > 0 {
				// Estimate frames per second from total frames and duration
				estimatedFrame := int(totalSeconds * float64(totalFrames) / (float64(totalFrames) / 30.0)) // Assume 30fps for estimation
				if estimatedFrame > 0 && estimatedFrame <= totalFrames {
					bar.Set(estimatedFrame)
				}
			}
		}

		// Check for completion indicators
		if strings.Contains(line, "video:") && strings.Contains(line, "audio:") {
			bar.Set(totalFrames)
			break
		}
	}
}

// ValidateFFmpegInstallation checks if FFmpeg is properly installed with required features
func ValidateFFmpegInstallation() error {
	// Check if ffmpeg exists
	if !IsFFmpegAvailable() {
		return fmt.Errorf("ffmpeg is not installed or not in PATH")
	}

	// Check if ffprobe exists (needed for video info)
	if _, err := exec.LookPath("ffprobe"); err != nil {
		return fmt.Errorf("ffprobe is not installed or not in PATH")
	}

	// Test ffmpeg with a basic command to ensure it works
	cmd := exec.Command("ffmpeg", "-version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg installation appears to be broken: %v", err)
	}

	return nil
}

// GetFFmpegVersion returns the version of FFmpeg
func GetFFmpegVersion() (string, error) {
	cmd := exec.Command("ffmpeg", "-version")
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("failed to get ffmpeg version: %v", err)
	}

	lines := strings.Split(string(output), "\n")
	if len(lines) > 0 {
		// Extract version from first line
		versionLine := lines[0]
		if strings.Contains(versionLine, "ffmpeg version") {
			parts := strings.Split(versionLine, " ")
			if len(parts) >= 3 {
				return parts[2], nil
			}
		}
	}

	return "unknown", nil
}
