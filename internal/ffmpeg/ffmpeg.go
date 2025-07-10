// internal/ffmpeg/ffmpeg.go
package ffmpeg

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// IsFFmpegAvailable checks if FFmpeg is installed and accessible
func IsFFmpegAvailable() bool {
	cmd := exec.Command("ffmpeg", "-version")
	err := cmd.Run()
	return err == nil
}

// ConvertMedia handles conversion to various output formats
func ConvertMedia(inputPath, outputPath, outputFormat string, startTime, endTime float64, frameRate int, quality int, resolution string) error {
	switch outputFormat {
	case "GIF":
		return convertToGIF(inputPath, outputPath, startTime, endTime, frameRate, resolution)
	case "APNG":
		return convertToAPNG(inputPath, outputPath, startTime, endTime, frameRate, resolution)
	case "WebP":
		return convertToWebP(inputPath, outputPath, startTime, endTime, frameRate, quality, resolution)
	case "AVIF":
		return convertToAVIF(inputPath, outputPath, startTime, endTime, frameRate, quality, resolution)
	case "MP4":
		return convertToMP4(inputPath, outputPath, startTime, endTime, frameRate, quality, resolution)
	case "WebM":
		return convertToWebM(inputPath, outputPath, startTime, endTime, frameRate, quality, resolution)
	default:
		return fmt.Errorf("unsupported output format: %s", outputFormat)
	}
}

func convertToGIF(inputPath, outputPath string, startTime, endTime float64, frameRate int, resolution string) error {
	args := []string{
		"-i", inputPath,
		"-ss", fmt.Sprintf("%.2f", startTime),
		"-t", fmt.Sprintf("%.2f", endTime-startTime),
		"-vf", buildVideoFilter(frameRate, resolution, "gif"),
		"-y", outputPath,
	}

	cmd := exec.Command("ffmpeg", args...)
	return cmd.Run()
}

func convertToAPNG(inputPath, outputPath string, startTime, endTime float64, frameRate int, resolution string) error {
	args := []string{
		"-i", inputPath,
		"-ss", fmt.Sprintf("%.2f", startTime),
		"-t", fmt.Sprintf("%.2f", endTime-startTime),
		"-vf", buildVideoFilter(frameRate, resolution, "apng"),
		"-f", "apng",
		"-y", outputPath,
	}

	cmd := exec.Command("ffmpeg", args...)
	return cmd.Run()
}

func convertToWebP(inputPath, outputPath string, startTime, endTime float64, frameRate, quality int, resolution string) error {
	args := []string{
		"-i", inputPath,
		"-ss", fmt.Sprintf("%.2f", startTime),
		"-t", fmt.Sprintf("%.2f", endTime-startTime),
		"-vf", buildVideoFilter(frameRate, resolution, "webp"),
		"-c:v", "libwebp",
		"-quality", strconv.Itoa(quality),
		"-y", outputPath,
	}

	cmd := exec.Command("ffmpeg", args...)
	return cmd.Run()
}

func convertToAVIF(inputPath, outputPath string, startTime, endTime float64, frameRate, quality int, resolution string) error {
	args := []string{
		"-i", inputPath,
		"-ss", fmt.Sprintf("%.2f", startTime),
		"-t", fmt.Sprintf("%.2f", endTime-startTime),
		"-vf", buildVideoFilter(frameRate, resolution, "avif"),
		"-c:v", "libaom-av1",
		"-crf", strconv.Itoa(quality),
		"-y", outputPath,
	}

	cmd := exec.Command("ffmpeg", args...)
	return cmd.Run()
}

func convertToMP4(inputPath, outputPath string, startTime, endTime float64, frameRate, quality int, resolution string) error {
	args := []string{
		"-i", inputPath,
		"-ss", fmt.Sprintf("%.2f", startTime),
		"-t", fmt.Sprintf("%.2f", endTime-startTime),
		"-c:v", "libx264",
		"-crf", strconv.Itoa(quality),
		"-r", strconv.Itoa(frameRate),
	}

	if resolution != "Original" {
		args = append(args, "-vf", fmt.Sprintf("scale=%s", resolution))
	}

	args = append(args, "-y", outputPath)

	cmd := exec.Command("ffmpeg", args...)
	return cmd.Run()
}

func convertToWebM(inputPath, outputPath string, startTime, endTime float64, frameRate, quality int, resolution string) error {
	args := []string{
		"-i", inputPath,
		"-ss", fmt.Sprintf("%.2f", startTime),
		"-t", fmt.Sprintf("%.2f", endTime-startTime),
		"-c:v", "libvpx-vp9",
		"-crf", strconv.Itoa(quality),
		"-r", strconv.Itoa(frameRate),
	}

	if resolution != "Original" {
		args = append(args, "-vf", fmt.Sprintf("scale=%s", resolution))
	}

	args = append(args, "-y", outputPath)

	cmd := exec.Command("ffmpeg", args...)
	return cmd.Run()
}

func buildVideoFilter(frameRate int, resolution, format string) string {
	var filters []string

	// Add scaling if not original resolution
	if resolution != "Original" {
		filters = append(filters, fmt.Sprintf("scale=%s", resolution))
	}

	// Add frame rate
	filters = append(filters, fmt.Sprintf("fps=%d", frameRate))

	// Format-specific optimizations
	switch format {
	case "gif":
		// Add palette optimization for GIF
		filters = append(filters, "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse")
	case "webp":
		// WebP-specific optimizations can be added here
	}

	return strings.Join(filters, ",")
}

// Keep the old ConvertToGIF function for backward compatibility
func ConvertToGIF(inputPath, outputPath string, startTime, endTime float64, frameRate int, resolution string) error {
	return convertToGIF(inputPath, outputPath, startTime, endTime, frameRate, resolution)
}
