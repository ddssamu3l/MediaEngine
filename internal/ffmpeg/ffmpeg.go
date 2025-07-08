// internal/ffmpeg/ffmpeg.go
package ffmpeg

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/schollz/progressbar/v3"
)

func IsFFmpegAvailable() bool {
	_, err := exec.LookPath("ffmpeg")
	return err == nil
}

func ConvertToGIF(inputPath, outputPath string, startTime, endTime float64, frameRate int, resolution string) error {
	duration := endTime - startTime

	// Build ffmpeg command
	args := []string{
		"-i", inputPath,
		"-ss", fmt.Sprintf("%.2f", startTime),
		"-t", fmt.Sprintf("%.2f", duration),
		"-r", strconv.Itoa(frameRate),
	}

	// Add resolution scaling if not original
	if resolution != "Original" {
		args = append(args, "-vf", fmt.Sprintf("scale=%s", resolution))
	}

	args = append(args, "-y", outputPath)

	// Create progress bar
	bar := progressbar.NewOptions(int(duration),
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
	)

	// Start conversion
	cmd := exec.Command("ffmpeg", args...)

	// Start the command
	err := cmd.Start()
	if err != nil {
		return fmt.Errorf("failed to start ffmpeg: %v", err)
	}

	// Simulate progress (since getting real-time progress from ffmpeg is complex)
	go func() {
		totalSteps := int(duration * 10) // Update every 100ms
		for i := 0; i <= totalSteps; i++ {
			bar.Set(i * int(duration) / totalSteps)
			time.Sleep(100 * time.Millisecond)

			// Check if process is still running
			if cmd.Process != nil {
				if proc, err := exec.Command("ps", "-p", strconv.Itoa(cmd.Process.Pid)).Output(); err != nil || !strings.Contains(string(proc), "ffmpeg") {
					break
				}
			}
		}
		bar.Finish()
	}()

	// Wait for completion
	err = cmd.Wait()
	if err != nil {
		return fmt.Errorf("ffmpeg conversion failed: %v", err)
	}

	return nil
}
