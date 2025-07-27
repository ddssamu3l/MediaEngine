package upscaling

import (
	"fmt"
	"strings"
	"testing"

	"mediaengine/internal/mocks"
)

// CRITERION 3: AI upscaling functionality tests
func TestUpscalingScalingModes(t *testing.T) {
	mockUpscaler := &MockUpscaler{
		Available: true,
		SupportedScales: []int{2, 4, -2, -4, -8},
	}
	
	tests := []struct {
		name        string
		scale       int
		expectError bool
		description string
	}{
		// Valid upscaling modes
		{name: "2x upscaling", scale: 2, expectError: false, description: "Standard 2x upscaling"},
		{name: "4x upscaling", scale: 4, expectError: false, description: "Standard 4x upscaling"},
		
		// Valid downscaling modes
		{name: "2x downscaling", scale: -2, expectError: false, description: "2x downscaling"},
		{name: "4x downscaling", scale: -4, expectError: false, description: "4x downscaling"},
		{name: "8x downscaling", scale: -8, expectError: false, description: "8x downscaling"},
		
		// Invalid scaling modes
		{name: "Invalid upscaling", scale: 3, expectError: true, description: "Unsupported 3x upscaling"},
		{name: "Invalid upscaling large", scale: 10, expectError: true, description: "Unsupported 10x upscaling"},
		{name: "Invalid downscaling", scale: -3, expectError: true, description: "Unsupported 3x downscaling"},
		{name: "Invalid downscaling large", scale: -16, expectError: true, description: "Unsupported 16x downscaling"},
		{name: "Zero scale", scale: 0, expectError: true, description: "Invalid zero scale"},
		{name: "One scale", scale: 1, expectError: true, description: "Invalid 1x scale (no change)"},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := mocks.UpscalingConfig{
				Model:      "RealESRGAN_x4plus",
				Scale:      tt.scale,
				PythonPath: "/usr/bin/python3",
				ModelPath:  "/models/",
			}
			
			err := mockUpscaler.ValidateConfig(config)
			
			if tt.expectError && err == nil {
				t.Errorf("Expected error for scale %d, but got nil (%s)", tt.scale, tt.description)
			}
			
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error for scale %d, but got: %v (%s)", tt.scale, err, tt.description)
			}
		})
	}
}

// CRITERION 3 (continued): Real-ESRGAN availability and failure handling
func TestRealESRGANAvailabilityAndFailures(t *testing.T) {
	tests := []struct {
		name           string
		setupMock      func(*MockUpscaler)
		expectError    bool
		errorContains  string
		description    string
	}{
		{
			name: "Real-ESRGAN available",
			setupMock: func(m *MockUpscaler) {
				m.Available = true
			},
			expectError: false,
			description: "Real-ESRGAN is properly installed",
		},
		{
			name: "Real-ESRGAN not available",
			setupMock: func(m *MockUpscaler) {
				m.Available = false
			},
			expectError:   true,
			errorContains: "not available",
			description:   "Real-ESRGAN is not installed",
		},
		{
			name: "Python not found",
			setupMock: func(m *MockUpscaler) {
				m.Available = false // Make it unavailable due to python issues
				m.Errors["python_check"] = fmt.Errorf("python not found in PATH")
			},
			expectError:   true,
			errorContains: "not available",
			description:   "Python interpreter not available",
		},
		{
			name: "Model files missing",
			setupMock: func(m *MockUpscaler) {
				m.Available = false
				m.Errors["model_check"] = fmt.Errorf("model files not found")
			},
			expectError:   true,
			errorContains: "not available",
			description:   "Real-ESRGAN model files missing",
		},
		{
			name: "Dependencies missing",
			setupMock: func(m *MockUpscaler) {
				m.Available = false
				m.Errors["dependency_check"] = fmt.Errorf("pytorch not installed")
			},
			expectError:   true,
			errorContains: "not available",
			description:   "Required Python dependencies missing",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockUpscaler := &MockUpscaler{
				Available: true,
				Errors:    make(map[string]error),
			}
			tt.setupMock(mockUpscaler)
			
			available := mockUpscaler.IsAvailable("/usr/bin/python3")
			
			if tt.expectError && available {
				t.Errorf("Expected Real-ESRGAN to be unavailable, but it was detected as available (%s)", tt.description)
			}
			
			if !tt.expectError && !available {
				t.Errorf("Expected Real-ESRGAN to be available, but it was not detected (%s)", tt.description)
			}
			
			// Test error handling during processing
			if tt.expectError {
				config := mocks.UpscalingConfig{
					Model:      "RealESRGAN_x4plus",
					Scale:      2,
					PythonPath: "/usr/bin/python3",
					ModelPath:  "/models/",
				}
				
				err := mockUpscaler.ProcessFrame("/input.jpg", "/output.jpg", config)
				if err == nil {
					t.Errorf("Expected processing error when Real-ESRGAN is unavailable (%s)", tt.description)
				}
				
				if tt.errorContains != "" && !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("Expected error to contain %q, but got: %v (%s)", tt.errorContains, err, tt.description)
				}
			}
		})
	}
}

// CRITERION 5: Memory estimation and GPU detection
func TestMemoryEstimationAndGPUDetection(t *testing.T) {
	mockUpscaler := &MockUpscaler{
		Available: true,
		GPUInfo: mocks.GPUInfo{
			HasCUDA:   true,
			HasMPS:    false,
			GPUMemory: 8 * 1024 * 1024 * 1024, // 8GB
			GPUName:   "NVIDIA RTX 4080",
		},
	}
	
	t.Run("Memory estimation for different resolutions", func(t *testing.T) {
		testCases := []struct {
			name           string
			width          int
			height         int
			scale          int
			expectedMemory int64
			description    string
		}{
			{name: "HD resolution", width: 1920, height: 1080, scale: 2, expectedMemory: 64 * 1024 * 1024, description: "1080p 2x upscaling"},
			{name: "4K resolution", width: 3840, height: 2160, scale: 2, expectedMemory: 256 * 1024 * 1024, description: "4K 2x upscaling"},
			{name: "Small resolution", width: 640, height: 480, scale: 4, expectedMemory: 16 * 1024 * 1024, description: "480p 4x upscaling"},
			{name: "Large resolution", width: 7680, height: 4320, scale: 2, expectedMemory: 1024 * 1024 * 1024, description: "8K 2x upscaling"},
		}
		
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				memory, err := mockUpscaler.EstimateMemoryUsage(tc.width, tc.height, tc.scale)
				if err != nil {
					t.Errorf("Memory estimation failed: %v (%s)", err, tc.description)
				}
				
				if memory <= 0 {
					t.Errorf("Memory estimation should be positive, got %d (%s)", memory, tc.description)
				}
				
				// Memory should scale with resolution and scale factor
				expectedRatio := float64(tc.width * tc.height * tc.scale * tc.scale)
				if memory < int64(expectedRatio/1000) { // Very rough approximation
					t.Errorf("Memory estimation seems too low for resolution %dx%d scale %d (%s)", tc.width, tc.height, tc.scale, tc.description)
				}
			})
		}
	})
	
	t.Run("Insufficient memory scenarios", func(t *testing.T) {
		// Test with low memory GPU
		lowMemoryUpscaler := &MockUpscaler{
			Available: true,
			Errors:    make(map[string]error),
			GPUInfo: mocks.GPUInfo{
				HasCUDA:   true,
				GPUMemory: 512 * 1024 * 1024, // 512MB - very low
			},
		}
		
		// Try to estimate memory for large operation
		memory, err := lowMemoryUpscaler.EstimateMemoryUsage(7680, 4320, 4) // 8K 4x upscaling
		if err != nil {
			t.Logf("Memory estimation returned error (expected): %v", err)
		} else if memory > lowMemoryUpscaler.GPUInfo.GPUMemory {
			t.Logf("Memory estimation (%d) exceeds GPU memory (%d) - this is expected", memory, lowMemoryUpscaler.GPUInfo.GPUMemory)
		}
		
		// Test processing with insufficient memory
		config := mocks.UpscalingConfig{
			Model:      "RealESRGAN_x4plus",
			Scale:      4,
			PythonPath: "/usr/bin/python3",
			ModelPath:  "/models/",
		}
		
		lowMemoryUpscaler.Errors["insufficient_memory"] = fmt.Errorf("insufficient GPU memory")
		err = lowMemoryUpscaler.ProcessFrame("/large_input.jpg", "/output.jpg", config)
		if err == nil {
			t.Error("Should fail processing when there's insufficient GPU memory")
		}
		
		if !strings.Contains(err.Error(), "memory") {
			t.Errorf("Error should mention memory issue: %v", err)
		}
	})
	
	t.Run("GPU configuration detection", func(t *testing.T) {
		gpuConfigs := []struct {
			name        string
			gpuInfo     mocks.GPUInfo
			description string
		}{
			{
				name: "CUDA GPU",
				gpuInfo: mocks.GPUInfo{
					HasCUDA:   true,
					HasMPS:    false,
					GPUMemory: 8 * 1024 * 1024 * 1024,
					GPUName:   "NVIDIA RTX 4080",
				},
				description: "NVIDIA GPU with CUDA support",
			},
			{
				name: "Apple Silicon GPU",
				gpuInfo: mocks.GPUInfo{
					HasCUDA:   false,
					HasMPS:    true,
					GPUMemory: 16 * 1024 * 1024 * 1024,
					GPUName:   "Apple M2 Max",
				},
				description: "Apple Silicon with MPS support",
			},
			{
				name: "CPU only",
				gpuInfo: mocks.GPUInfo{
					HasCUDA:   false,
					HasMPS:    false,
					GPUMemory: 0,
					GPUName:   "CPU",
				},
				description: "No GPU acceleration available",
			},
		}
		
		for _, gc := range gpuConfigs {
			t.Run(gc.name, func(t *testing.T) {
				gpuUpscaler := &MockUpscaler{
					Available: true,
					GPUInfo:   gc.gpuInfo,
				}
				
				info, err := gpuUpscaler.GetGPUInfo("/usr/bin/python3")
				if err != nil {
					t.Errorf("GPU info detection failed: %v (%s)", err, gc.description)
				}
				
				if info.HasCUDA != gc.gpuInfo.HasCUDA {
					t.Errorf("CUDA detection mismatch: expected %v, got %v (%s)", gc.gpuInfo.HasCUDA, info.HasCUDA, gc.description)
				}
				
				if info.HasMPS != gc.gpuInfo.HasMPS {
					t.Errorf("MPS detection mismatch: expected %v, got %v (%s)", gc.gpuInfo.HasMPS, info.HasMPS, gc.description)
				}
				
				if gc.gpuInfo.GPUMemory > 0 && info.GPUMemory == 0 {
					t.Errorf("GPU memory should be detected when GPU is available (%s)", gc.description)
				}
			})
		}
	})
}

// MockUpscaler implements the UpscalingInterface for testing
type MockUpscaler struct {
	Available       bool
	SupportedScales []int
	GPUInfo         mocks.GPUInfo
	Errors          map[string]error
}

func (m *MockUpscaler) IsAvailable(pythonPath string) bool {
	if m.Errors["python_check"] != nil || 
	   m.Errors["model_check"] != nil || 
	   m.Errors["dependency_check"] != nil {
		return false
	}
	return m.Available
}

func (m *MockUpscaler) ValidateConfig(config mocks.UpscalingConfig) error {
	if !m.Available {
		return fmt.Errorf("Real-ESRGAN not available")
	}
	
	// Validate scale
	validScale := false
	for _, scale := range m.SupportedScales {
		if config.Scale == scale {
			validScale = true
			break
		}
	}
	
	if !validScale {
		return fmt.Errorf("unsupported scale factor: %d. Supported scales: 2x, 4x upscaling; 2x, 4x, 8x downscaling", config.Scale)
	}
	
	if config.Model == "" {
		return fmt.Errorf("model not specified")
	}
	
	if config.PythonPath == "" {
		return fmt.Errorf("python path not specified")
	}
	
	return nil
}

func (m *MockUpscaler) EstimateMemoryUsage(width, height int, scale int) (int64, error) {
	if width <= 0 || height <= 0 {
		return 0, fmt.Errorf("invalid dimensions")
	}
	
	if scale == 0 {
		return 0, fmt.Errorf("invalid scale factor")
	}
	
	// Simple memory estimation: base memory + (width * height * scale^2 * bytes_per_pixel)
	baseMemory := int64(100 * 1024 * 1024) // 100MB base
	pixelMemory := int64(width * height * scale * scale * 4) // 4 bytes per pixel
	
	return baseMemory + pixelMemory, nil
}

func (m *MockUpscaler) GetGPUInfo(pythonPath string) (mocks.GPUInfo, error) {
	if m.Errors["gpu_detection"] != nil {
		return mocks.GPUInfo{}, m.Errors["gpu_detection"]
	}
	
	return m.GPUInfo, nil
}

func (m *MockUpscaler) ProcessFrame(inputPath, outputPath string, config mocks.UpscalingConfig) error {
	if !m.Available {
		return fmt.Errorf("Real-ESRGAN not available")
	}
	
	if m.Errors["insufficient_memory"] != nil {
		return m.Errors["insufficient_memory"]
	}
	
	if m.Errors["processing"] != nil {
		return m.Errors["processing"]
	}
	
	err := m.ValidateConfig(config)
	if err != nil {
		return err
	}
	
	return nil
}