// internal/upscaling/upscaling.go
package upscaling

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// UpscalingConfig holds configuration for AI upscaling
type UpscalingConfig struct {
	Enabled    bool
	Model      string // RealESRGAN_x4plus, RealESRGAN_x2plus, RealESRGAN_x4plus_anime_6B
	Scale      int    // 2, 3, or 4
	UseGPU     bool
	GPUDevice  int
	PythonPath string
	ScriptPath string
	TempDir    string
}

// UpscalingModels defines available models
var UpscalingModels = map[string]string{
	"general_4x": "RealESRGAN_x4plus",
	"general_2x": "RealESRGAN_x2plus",
	"anime_4x":   "RealESRGAN_x4plus_anime_6B",
}

// UpscalingModelDescriptions provides user-friendly descriptions
var UpscalingModelDescriptions = map[string]string{
	"general_4x": "General Purpose 4x (Best for photos/real content)",
	"general_2x": "General Purpose 2x (Faster, good quality)",
	"anime_4x":   "Anime/Cartoon 4x (Optimized for animated content)",
}

// Upscaler handles AI upscaling operations
type Upscaler struct {
	config *UpscalingConfig
}

// NewUpscaler creates a new upscaler instance
func NewUpscaler(config *UpscalingConfig) *Upscaler {
	return &Upscaler{config: config}
}

// IsAvailable checks if Real-ESRGAN is available
func (u *Upscaler) IsAvailable() bool {
	if u.config.PythonPath == "" {
		return false
	}

	// Check if Python environment exists
	if _, err := os.Stat(u.config.PythonPath); os.IsNotExist(err) {
		return false
	}

	// Check if script exists
	if _, err := os.Stat(u.config.ScriptPath); os.IsNotExist(err) {
		return false
	}

	// Test if realesrgan package is available
	cmd := exec.Command(u.config.PythonPath, "-c", "import realesrgan; print('OK')")
	output, err := cmd.Output()
	if err != nil {
		return false
	}

	return strings.TrimSpace(string(output)) == "OK"
}

// UpscaleFrame upscales a single frame using Real-ESRGAN
func (u *Upscaler) UpscaleFrame(inputPath, outputPath string) error {
	if !u.config.Enabled {
		return fmt.Errorf("upscaling is disabled")
	}

	if !u.IsAvailable() {
		return fmt.Errorf("Real-ESRGAN is not available")
	}

	// Build command arguments
	args := []string{
		u.config.ScriptPath,
		inputPath,
		outputPath,
		"--model", u.config.Model,
		"--scale", strconv.Itoa(u.config.Scale),
	}

	// Add GPU flag if enabled
	if u.config.UseGPU {
		args = append(args, "--gpu", strconv.Itoa(u.config.GPUDevice))
	}

	// Execute upscaling
	cmd := exec.Command(u.config.PythonPath, args...)
	output, err := cmd.CombinedOutput()

	if err != nil {
		return fmt.Errorf("upscaling failed: %v\nOutput: %s", err, string(output))
	}

	// Verify output file was created
	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		return fmt.Errorf("upscaling completed but output file was not created")
	}

	return nil
}

// GetModelInfo returns information about available models
func GetModelInfo() map[string]string {
	return UpscalingModelDescriptions
}

// ValidateConfig validates upscaling configuration
func ValidateConfig(config *UpscalingConfig) error {
	if !config.Enabled {
		return nil
	}

	// Check Python path
	if config.PythonPath == "" {
		return fmt.Errorf("Python path is required for upscaling")
	}

	if _, err := os.Stat(config.PythonPath); os.IsNotExist(err) {
		return fmt.Errorf("Python executable not found: %s", config.PythonPath)
	}

	// Check script path
	if config.ScriptPath == "" {
		return fmt.Errorf("upscaling script path is required")
	}

	if _, err := os.Stat(config.ScriptPath); os.IsNotExist(err) {
		return fmt.Errorf("upscaling script not found: %s", config.ScriptPath)
	}

	// Validate model
	if _, exists := UpscalingModels[config.Model]; !exists {
		return fmt.Errorf("invalid upscaling model: %s", config.Model)
	}

	// Validate scale
	if config.Scale < 2 || config.Scale > 4 {
		return fmt.Errorf("upscaling scale must be 2, 3, or 4")
	}

	// Validate GPU settings
	if config.UseGPU && config.GPUDevice < 0 {
		return fmt.Errorf("invalid GPU device ID: %d", config.GPUDevice)
	}

	return nil
}

// GetDefaultConfig returns default upscaling configuration
func GetDefaultConfig() *UpscalingConfig {
	return &UpscalingConfig{
		Enabled:    false,
		Model:      "general_4x",
		Scale:      4,
		UseGPU:     false,
		GPUDevice:  0,
		PythonPath: "python3",
		ScriptPath: "scripts/upscale_frame.py",
		TempDir:    os.TempDir(),
	}
}
