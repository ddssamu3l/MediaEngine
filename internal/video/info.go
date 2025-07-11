// internal/video/info.go
package video

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

type VideoInfo struct {
	Filepath  string
	FileSize  int64
	Width     int
	Height    int
	Duration  float64
	Format    string
	Bitrate   int64
	FrameRate float64
}

type FFProbeOutput struct {
	Streams []struct {
		Width        int    `json:"width"`
		Height       int    `json:"height"`
		Duration     string `json:"duration"`
		CodecType    string `json:"codec_type"`
		RFrameRate   string `json:"r_frame_rate"`
		AvgFrameRate string `json:"avg_frame_rate"`
	} `json:"streams"`
	Format struct {
		Duration string `json:"duration"`
		Bitrate  string `json:"bit_rate"`
		Format   string `json:"format_name"`
	} `json:"format"`
}

func GetVideoInfo(filepath string) (*VideoInfo, error) {
	// Get file size
	fileInfo, err := os.Stat(filepath)
	if err != nil {
		return nil, err
	}

	// Use ffprobe to get video information
	cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", filepath)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to run ffprobe: %v", err)
	}

	var probe FFProbeOutput
	if err := json.Unmarshal(output, &probe); err != nil {
		return nil, fmt.Errorf("failed to parse ffprobe output: %v", err)
	}

	info := &VideoInfo{
		Filepath: filepath,
		FileSize: fileInfo.Size(),
		Format:   probe.Format.Format,
	}

	// Find video stream
	for _, stream := range probe.Streams {
		if stream.CodecType == "video" {
			info.Width = stream.Width
			info.Height = stream.Height

			// Parse frame rate
			if frameRate := parseFrameRate(stream.RFrameRate); frameRate > 0 {
				info.FrameRate = frameRate
			} else if frameRate := parseFrameRate(stream.AvgFrameRate); frameRate > 0 {
				info.FrameRate = frameRate
			} else {
				info.FrameRate = 30.0 // Default fallback
			}
			break
		}
	}

	// Parse duration
	if probe.Format.Duration != "" {
		if duration, err := strconv.ParseFloat(probe.Format.Duration, 64); err == nil {
			info.Duration = duration
		}
	}

	// Parse bitrate
	if probe.Format.Bitrate != "" {
		if bitrate, err := strconv.ParseInt(probe.Format.Bitrate, 10, 64); err == nil {
			info.Bitrate = bitrate
		}
	}

	return info, nil
}

// parseFrameRate parses frame rate from FFprobe format (e.g., "30/1" or "30000/1001")
func parseFrameRate(frameRateStr string) float64 {
	if frameRateStr == "" || frameRateStr == "0/0" {
		return 0
	}

	// Handle fractional frame rates like "30000/1001"
	if strings.Contains(frameRateStr, "/") {
		parts := strings.Split(frameRateStr, "/")
		if len(parts) == 2 {
			numerator, err1 := strconv.ParseFloat(parts[0], 64)
			denominator, err2 := strconv.ParseFloat(parts[1], 64)
			if err1 == nil && err2 == nil && denominator != 0 {
				return numerator / denominator
			}
		}
	} else {
		// Handle simple numeric frame rates
		if rate, err := strconv.ParseFloat(frameRateStr, 64); err == nil {
			return rate
		}
	}

	return 0
}
