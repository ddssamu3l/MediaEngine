// internal/ui/ui.go
package ui

import (
	"fmt"
	"path/filepath"
	"videotogif/internal/video"

	"github.com/charmbracelet/lipgloss"
)

var (
	infoStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#7C3AED")).
			Padding(1, 2).
			MarginTop(1).
			MarginBottom(1)

	labelStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#6B7280")).
			Bold(true)

	valueStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#111827"))
)

func DisplayVideoInfo(info *video.VideoInfo) {
	content := fmt.Sprintf(
		"%s %s\n"+
			"%s %s\n"+
			"%s %dx%d\n"+
			"%s %s\n"+
			"%s %s\n"+
			"%s %s",
		labelStyle.Render("📁 File:"), valueStyle.Render(filepath.Base(info.Filepath)),
		labelStyle.Render("📊 Size:"), valueStyle.Render(FormatFileSize(info.FileSize)),
		labelStyle.Render("📐 Dimensions:"), info.Width, info.Height,
		labelStyle.Render("🎬 Format:"), valueStyle.Render(info.Format),
		labelStyle.Render("⚡ Bitrate:"), valueStyle.Render(formatBitrate(info.Bitrate)),
		labelStyle.Render("⏱️  Duration:"), valueStyle.Render(FormatDuration(info.Duration)),
	)

	fmt.Println(infoStyle.Render(content))
}

// FormatFileSize converts bytes to human-readable format
func FormatFileSize(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// FormatDuration converts seconds to MM:SS format
func FormatDuration(seconds float64) string {
	totalSeconds := int(seconds)
	minutes := totalSeconds / 60
	remainingSeconds := totalSeconds % 60

	return fmt.Sprintf("%02d:%02d", minutes, remainingSeconds)
}

func formatBitrate(bitrate int64) string {
	if bitrate == 0 {
		return "Unknown"
	}
	return fmt.Sprintf("%.1f kbps", float64(bitrate)/1000)
}
