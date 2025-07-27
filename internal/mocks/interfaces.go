// Package mocks provides interfaces and mock implementations for testing
package mocks

import (
	"io"
	"os"
	"time"
)

// FileSystemInterface abstracts file system operations for testing
type FileSystemInterface interface {
	Stat(name string) (os.FileInfo, error)
	Open(name string) (*os.File, error)
	Create(name string) (*os.File, error)
	Remove(name string) error
	MkdirAll(path string, perm os.FileMode) error
	WriteFile(filename string, data []byte, perm os.FileMode) error
	ReadFile(filename string) ([]byte, error)
	TempDir(dir, prefix string) (string, error)
	RemoveAll(path string) error
}

// CommandExecutorInterface abstracts command execution for testing
type CommandExecutorInterface interface {
	Execute(name string, args ...string) ([]byte, error)
	ExecuteWithStdin(name string, stdin io.Reader, args ...string) ([]byte, error)
	IsAvailable(command string) bool
}

// UserInteractionInterface abstracts user prompts for testing
type UserInteractionInterface interface {
	PromptForFloat(label string, defaultValue float64, validator func(float64) error) (float64, error)
	PromptForString(label string, defaultValue string, validator func(string) error) (string, error)
	PromptForSelect(label string, items []string) (string, error)
	PromptForConfirm(label string) (bool, error)
}

// TimeInterface abstracts time operations for testing
type TimeInterface interface {
	Now() time.Time
	Sleep(d time.Duration)
}

// VideoInfoInterface abstracts video information extraction
type VideoInfoInterface interface {
	GetVideoDuration(filePath string) (float64, error)
	GetVideoResolution(filePath string) (int, int, error)
	GetVideoFrameRate(filePath string) (float64, error)
	ValidateVideoFile(filePath string) error
}

// UpscalingInterface abstracts AI upscaling operations
type UpscalingInterface interface {
	IsAvailable(pythonPath string) bool
	ValidateConfig(config UpscalingConfig) error
	EstimateMemoryUsage(width, height int, scale int) (int64, error)
	GetGPUInfo(pythonPath string) (GPUInfo, error)
	ProcessFrame(inputPath, outputPath string, config UpscalingConfig) error
}

// UpscalingConfig represents upscaling configuration
type UpscalingConfig struct {
	Model      string
	Scale      int
	PythonPath string
	ModelPath  string
}

// GPUInfo represents GPU information
type GPUInfo struct {
	HasCUDA   bool
	HasMPS    bool
	GPUMemory int64
	GPUName   string
}

// ConversionInterface abstracts media conversion operations
type ConversionInterface interface {
	ConvertToGIF(inputPath, outputPath string, config ConversionConfig) error
	ConvertToWebP(inputPath, outputPath string, config ConversionConfig) error
	ConvertToMP4(inputPath, outputPath string, config ConversionConfig) error
	ConvertToAPNG(inputPath, outputPath string, config ConversionConfig) error
	ConvertToAVIF(inputPath, outputPath string, config ConversionConfig) error
	ConvertToWebM(inputPath, outputPath string, config ConversionConfig) error
	ValidateOutput(outputPath string) error
}

// ConversionConfig represents conversion configuration
type ConversionConfig struct {
	StartTime   float64
	EndTime     float64
	FrameRate   float64
	Quality     string
	Width       int
	Height      int
	EnableAI    bool
	AIConfig    UpscalingConfig
}