// internal/video/info.go
package video

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strconv"
)

type VideoInfo struct {
	Filepath string
	FileSize int64
	Width    int
	Height   int
	Duration float64
	Format   string
	Bitrate  int64
}

type FFProbeOutput struct {
	Streams []struct {
		Width     int    `json:"width"`
		Height    int    `json:"height"`
		Duration  string `json:"duration"`
		CodecType string `json:"codec_type"`
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
