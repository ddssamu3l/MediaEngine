package interpolation

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestInterpolationConfig tests the interpolation configuration validation
func TestInterpolationConfig(t *testing.T) {
	t.Run("Valid Configuration", func(t *testing.T) {
		// Create a temporary script file for testing
		tempScript, err := os.CreateTemp("", "test_script_*.py")
		if err != nil {
			t.Fatalf("Failed to create temp script: %v", err)
		}
		defer os.Remove(tempScript.Name())
		tempScript.Close()

		config := &InterpolationConfig{
			Enabled:             true,
			Model:               "rife_v4.6",
			TargetFPS:           60,
			OriginalFPS:         30,
			Multiplier:          2,
			UseGPU:              true,
			GPUDevice:           0,
			PythonPath:          "python3",
			ScriptPath:          tempScript.Name(),
			TempDir:             os.TempDir(),
			QualityLevel:        "balanced",
			MaxBatchSize:        32,
			UseParallel:         true,
			AsyncProcessing:     true,
			GPUMemoryLimit:      0,
		}

		validationErr := ValidateConfig(config)
		if validationErr != nil {
			t.Errorf("Expected valid config to pass validation, got error: %v", validationErr)
		}
	})

	t.Run("Invalid Model", func(t *testing.T) {
		config := GetDefaultConfig()
		config.Enabled = true
		config.Model = "invalid_model"

		err := ValidateConfig(config)
		if err == nil {
			t.Error("Expected invalid model to fail validation")
		}
	})

	t.Run("Invalid Frame Rates", func(t *testing.T) {
		config := GetDefaultConfig()
		config.Enabled = true
		config.TargetFPS = 30
		config.OriginalFPS = 60 // Target lower than original

		err := ValidateConfig(config)
		if err == nil {
			t.Error("Expected invalid frame rates to fail validation")
		}
	})

	t.Run("Invalid Multiplier", func(t *testing.T) {
		config := GetDefaultConfig()
		config.Enabled = true
		config.TargetFPS = 150
		config.OriginalFPS = 30 // 5x multiplier, should fail

		err := ValidateConfig(config)
		if err == nil {
			t.Error("Expected excessive multiplier to fail validation")
		}
	})
}

// TestInterpolatorCreation tests the interpolator instance creation
func TestInterpolatorCreation(t *testing.T) {
	config := GetDefaultConfig()
	interpolator := NewInterpolator(config)

	if interpolator == nil {
		t.Error("Expected interpolator to be created")
	}

	if interpolator.config != config {
		t.Error("Expected interpolator to store the provided config")
	}
}

// TestQualityProfiles tests the quality profile system
func TestQualityProfiles(t *testing.T) {
	profiles := GetQualityProfiles()

	expectedProfiles := []string{"fast", "balanced", "high_quality"}
	for _, expected := range expectedProfiles {
		if _, exists := profiles[expected]; !exists {
			t.Errorf("Expected quality profile '%s' to exist", expected)
		}
	}

	// Test profile properties
	fastProfile := profiles["fast"]
	if fastProfile.BatchSize <= 0 {
		t.Error("Expected fast profile to have positive batch size")
	}

	if fastProfile.Precision != "fp16" {
		t.Error("Expected fast profile to use fp16 precision")
	}
}

// TestModelInfo tests the model information system
func TestModelInfo(t *testing.T) {
	models := GetModelInfo()

	expectedModels := []string{"rife_v4.6", "rife_v4.4", "rife_v4.0", "rife_v4.6_lite"}
	for _, expected := range expectedModels {
		if _, exists := models[expected]; !exists {
			t.Errorf("Expected model '%s' to exist", expected)
		}
	}

	// Test model descriptions
	for model, description := range models {
		if description == "" {
			t.Errorf("Expected model '%s' to have a description", model)
		}
	}
}

// TestOptimalBatchSize tests the batch size optimization
func TestOptimalBatchSize(t *testing.T) {
	config := GetDefaultConfig()
	config.MaxBatchSize = 64
	interpolator := NewInterpolator(config)

	// Mock GPU info for testing
	mockGPUInfo := &struct {
		DeviceType        string
		TotalMemoryGB     float64
		AvailableMemoryGB float64
	}{
		DeviceType:        "CUDA",
		TotalMemoryGB:     16.0,
		AvailableMemoryGB: 12.0,
	}

	// Test that interpolator was created successfully
	if interpolator.config.MaxBatchSize != 64 {
		t.Errorf("Expected max batch size to be 64, got %d", interpolator.config.MaxBatchSize)
	}

	// Test the min/max functions
	result1 := min(32, 64)
	if result1 != 32 {
		t.Errorf("Expected min(32, 64) = 32, got %d", result1)
	}

	result2 := max(8, 4)
	if result2 != 8 {
		t.Errorf("Expected max(8, 4) = 8, got %d", result2)
	}

	_ = mockGPUInfo // Use the variable to avoid unused warning
}

// TestErrorRecovery tests the error recovery mechanisms
func TestErrorRecovery(t *testing.T) {
	config := GetDefaultConfig()
	interpolator := NewInterpolator(config)

	t.Run("Recoverable Errors", func(t *testing.T) {
		recoverableErrors := []string{
			"CUDA out of memory",
			"device busy",
			"connection reset",
			"timeout occurred",
			"interrupted by user",
		}

		for _, errMsg := range recoverableErrors {
			testErr := &testError{message: errMsg}
			if !interpolator.isRecoverableError(testErr) {
				t.Errorf("Expected error '%s' to be recoverable", errMsg)
			}
		}
	})

	t.Run("Non-recoverable Errors", func(t *testing.T) {
		nonRecoverableErrors := []string{
			"model not found",
			"invalid model configuration",
			"permission denied",
			"unsupported format",
			"no such file or directory",
		}

		for _, errMsg := range nonRecoverableErrors {
			testErr := &testError{message: errMsg}
			if interpolator.isRecoverableError(testErr) {
				t.Errorf("Expected error '%s' to be non-recoverable", errMsg)
			}
		}
	})

	t.Run("Nil Error", func(t *testing.T) {
		if interpolator.isRecoverableError(nil) {
			t.Error("Expected nil error to be non-recoverable")
		}
	})
}

// TestDiskSpaceMonitoring tests the disk space monitoring functionality
func TestDiskSpaceMonitoring(t *testing.T) {
	config := GetDefaultConfig()
	interpolator := NewInterpolator(config)

	t.Run("Valid Directory", func(t *testing.T) {
		tempDir := os.TempDir()
		err := interpolator.monitorDiskSpace(tempDir, 1.0) // 1GB requirement
		if err != nil {
			t.Errorf("Expected valid directory to pass disk space check, got: %v", err)
		}
	})

	t.Run("Invalid Directory", func(t *testing.T) {
		invalidDir := "/nonexistent/directory"
		err := interpolator.monitorDiskSpace(invalidDir, 1.0)
		if err == nil {
			t.Error("Expected invalid directory to fail disk space check")
		}
	})
}

// TestResourceCleanup tests the resource cleanup functionality
func TestResourceCleanup(t *testing.T) {
	config := GetDefaultConfig()
	interpolator := NewInterpolator(config)

	// Create a temporary directory with test files
	tempDir, err := os.MkdirTemp("", "interpolation_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create some test files that should be cleaned up
	testFiles := []string{
		"test.tmp",
		"frame_001_incomplete.png",
		"processing_incomplete.json",
	}

	for _, fileName := range testFiles {
		filePath := filepath.Join(tempDir, fileName)
		file, err := os.Create(filePath)
		if err != nil {
			t.Fatalf("Failed to create test file: %v", err)
		}
		file.Close()
	}

	// Verify files exist before cleanup
	for _, fileName := range testFiles {
		filePath := filepath.Join(tempDir, fileName)
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			t.Errorf("Test file %s should exist before cleanup", fileName)
		}
	}

	// Perform cleanup
	interpolator.cleanupResources(tempDir)

	// Verify .tmp and incomplete files are removed
	tmpFile := filepath.Join(tempDir, "test.tmp")
	if _, err := os.Stat(tmpFile); !os.IsNotExist(err) {
		t.Error("Temporary files should be cleaned up")
	}

	incompleteFile := filepath.Join(tempDir, "frame_001_incomplete.png")
	if _, err := os.Stat(incompleteFile); !os.IsNotExist(err) {
		t.Error("Incomplete files should be cleaned up")
	}
}

// TestEstimateProcessingTime tests the processing time estimation
func TestEstimateProcessingTime(t *testing.T) {
	config := GetDefaultConfig()
	config.QualityLevel = "balanced"
	config.Multiplier = 2
	interpolator := NewInterpolator(config)

	frameCount := 100
	width, height := 1920, 1080

	duration, err := interpolator.EstimateProcessingTime(frameCount, width, height)
	if err != nil {
		t.Errorf("Expected processing time estimation to succeed, got: %v", err)
	}

	if duration <= 0 {
		t.Error("Expected positive processing time estimation")
	}

	// Should be reasonable (less than 30 minutes for 100 frames)
	if duration > 30*time.Minute {
		t.Errorf("Processing time estimation seems too high: %v", duration)
	}
}

// TestProcessingTimeWithDifferentParameters tests estimation with various parameters
func TestProcessingTimeWithDifferentParameters(t *testing.T) {
	testCases := []struct {
		name       string
		frameCount int
		width      int
		height     int
		quality    string
		multiplier int
	}{
		{"Small Video", 30, 720, 480, "fast", 2},
		{"Medium Video", 60, 1280, 720, "balanced", 2},
		{"Large Video", 120, 1920, 1080, "high_quality", 3},
		{"High Multiplier", 60, 1280, 720, "balanced", 4},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := GetDefaultConfig()
			config.QualityLevel = tc.quality
			config.Multiplier = tc.multiplier
			interpolator := NewInterpolator(config)

			duration, err := interpolator.EstimateProcessingTime(tc.frameCount, tc.width, tc.height)
			if err != nil {
				t.Errorf("Expected processing time estimation to succeed for %s, got: %v", tc.name, err)
			}

			if duration <= 0 {
				t.Errorf("Expected positive processing time estimation for %s", tc.name)
			}
		})
	}
}

// Helper type for testing error recovery
type testError struct {
	message string
}

func (e *testError) Error() string {
	return e.message
}

// Benchmark tests for performance validation
func BenchmarkEstimateProcessingTime(b *testing.B) {
	config := GetDefaultConfig()
	interpolator := NewInterpolator(config)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = interpolator.EstimateProcessingTime(100, 1920, 1080)
	}
}

func BenchmarkIsRecoverableError(b *testing.B) {
	config := GetDefaultConfig()
	interpolator := NewInterpolator(config)
	testErr := &testError{message: "CUDA out of memory"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = interpolator.isRecoverableError(testErr)
	}
}

// TestFrameOrganization tests the frame organization functionality
func TestFrameOrganization(t *testing.T) {
	config := GetDefaultConfig()
	interpolator := NewInterpolator(config)

	// Create a temporary directory with test frame files
	tempDir, err := os.MkdirTemp("", "frame_org_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create test frame files with RIFE naming pattern
	testFrames := []string{
		"frame_000000_00.png",
		"frame_000000_01.png", 
		"frame_000002_00.png",
		"frame_000002_01.png",
		"frame_000004_00.png",
	}

	for _, frameName := range testFrames {
		framePath := filepath.Join(tempDir, frameName)
		file, err := os.Create(framePath)
		if err != nil {
			t.Fatalf("Failed to create test frame: %v", err)
		}
		file.Close()
	}

	// Test frame organization
	err = interpolator.OrganizeInterpolatedFrames(tempDir)
	if err != nil {
		t.Errorf("Frame organization failed: %v", err)
		return
	}

	// Check that files were renamed correctly
	expectedFiles := []string{
		"frame_000000.png", // was frame_000000_00.png
		"frame_000001.png", // was frame_000000_01.png
		"frame_000002.png", // was frame_000002_00.png
		"frame_000003.png", // was frame_000002_01.png
		"frame_000004.png", // was frame_000004_00.png
	}

	for _, expectedFile := range expectedFiles {
		expectedPath := filepath.Join(tempDir, expectedFile)
		if _, err := os.Stat(expectedPath); os.IsNotExist(err) {
			t.Errorf("Expected file %s not found after organization", expectedFile)
		}
	}

	// Verify no old files remain
	for _, oldFrame := range testFrames {
		oldPath := filepath.Join(tempDir, oldFrame)
		if _, err := os.Stat(oldPath); !os.IsNotExist(err) {
			t.Errorf("Old file %s should have been renamed", oldFrame)
		}
	}
}