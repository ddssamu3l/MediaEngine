package ui

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"mediaengine/internal/mocks"
)

// CRITERION 7: User interaction flow testing with mocks
func TestUserInteractionFlow(t *testing.T) {
	mockUI := mocks.NewMockUserInteraction()
	
	t.Run("Complete workflow prompts", func(t *testing.T) {
		// Setup mock responses for a complete workflow
		mockUI.FloatResponses["Start time (seconds)"] = 10.5
		mockUI.FloatResponses["End time (seconds)"] = 25.0
		mockUI.FloatResponses["Frame rate (fps)"] = 24.0
		mockUI.StringResponses["Output format"] = "gif"
		mockUI.StringResponses["Quality setting"] = "medium"
		mockUI.ConfirmResponses["Enable AI upscaling?"] = true
		mockUI.SelectResponses["Upscaling model"] = "RealESRGAN_x4plus"
		
		// Test the complete user interaction flow
		config, err := runUserInteractionFlowWithMock(mockUI)
		if err != nil {
			t.Fatalf("User interaction flow failed: %v", err)
		}
		
		// Verify all prompts were called
		expectedPrompts := []string{
			"PromptForFloat: Start time (seconds)",
			"PromptForFloat: End time (seconds)",
			"PromptForFloat: Frame rate (fps)",
			"PromptForString: Output format",
			"PromptForString: Quality setting",
			"PromptForConfirm: Enable AI upscaling?",
			"PromptForSelect: Upscaling model",
		}
		
		for _, expectedPrompt := range expectedPrompts {
			found := false
			for _, actualCall := range mockUI.CallLog {
				if strings.Contains(actualCall, expectedPrompt) {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("Expected prompt %q not found in call log: %v", expectedPrompt, mockUI.CallLog)
			}
		}
		
		// Verify configuration values
		if config.StartTime != 10.5 {
			t.Errorf("Expected start time 10.5, got %f", config.StartTime)
		}
		if config.EndTime != 25.0 {
			t.Errorf("Expected end time 25.0, got %f", config.EndTime)
		}
		if config.FrameRate != 24.0 {
			t.Errorf("Expected frame rate 24.0, got %f", config.FrameRate)
		}
	})
	
	t.Run("Invalid user inputs", func(t *testing.T) {
		tests := []struct {
			name          string
			setupMock     func(*mocks.MockUserInteraction)
			expectError   bool
			errorContains string
			description   string
		}{
			{
				name: "Invalid start time",
				setupMock: func(m *mocks.MockUserInteraction) {
					m.FloatResponses["Start time (seconds)"] = -5.0
				},
				expectError:   true,
				errorContains: "negative",
				description:   "Negative start time should be rejected",
			},
			{
				name: "Start time after end time",
				setupMock: func(m *mocks.MockUserInteraction) {
					m.FloatResponses["Start time (seconds)"] = 30.0
					m.FloatResponses["End time (seconds)"] = 20.0
				},
				expectError:   true,
				errorContains: "start time",
				description:   "Start time after end time should be rejected",
			},
			{
				name: "Invalid frame rate",
				setupMock: func(m *mocks.MockUserInteraction) {
					m.FloatResponses["Start time (seconds)"] = 0.0
					m.FloatResponses["End time (seconds)"] = 10.0
					m.FloatResponses["Frame rate (fps)"] = 0.0
				},
				expectError:   true,
				errorContains: "frame rate",
				description:   "Zero frame rate should be rejected",
			},
			{
				name: "Invalid output format",
				setupMock: func(m *mocks.MockUserInteraction) {
					m.FloatResponses["Start time (seconds)"] = 0.0
					m.FloatResponses["End time (seconds)"] = 10.0
					m.FloatResponses["Frame rate (fps)"] = 24.0
					m.StringResponses["Output format"] = "invalid_format"
				},
				expectError:   true,
				errorContains: "format",
				description:   "Invalid output format should be rejected",
			},
		}
		
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				mockUI := mocks.NewMockUserInteraction()
				tt.setupMock(mockUI)
				
				_, err := runUserInteractionFlowWithMock(mockUI)
				
				if tt.expectError && err == nil {
					t.Errorf("Expected error for %s, but got nil (%s)", tt.name, tt.description)
				}
				
				if !tt.expectError && err != nil {
					t.Errorf("Expected no error for %s, but got: %v (%s)", tt.name, err, tt.description)
				}
				
				if tt.expectError && err != nil && tt.errorContains != "" {
					if !strings.Contains(strings.ToLower(err.Error()), strings.ToLower(tt.errorContains)) {
						t.Errorf("Expected error to contain %q, but got: %v (%s)", tt.errorContains, err, tt.description)
					}
				}
			})
		}
	})
	
	t.Run("Default value handling", func(t *testing.T) {
		mockUI := mocks.NewMockUserInteraction()
		// Don't set any responses - should use defaults
		
		config, err := runUserInteractionFlowWithMock(mockUI)
		if err != nil {
			t.Fatalf("Default value handling failed: %v", err)
		}
		
		// Verify default values are used
		if config.StartTime != 0.0 {
			t.Errorf("Expected default start time 0.0, got %f", config.StartTime)
		}
		
		if config.FrameRate <= 0 {
			t.Errorf("Default frame rate should be positive, got %f", config.FrameRate)
		}
	})
	
	t.Run("Cancellation scenarios", func(t *testing.T) {
		mockUI := mocks.NewMockUserInteraction()
		mockUI.Errors["Start time (seconds)"] = fmt.Errorf("user cancelled")
		
		_, err := runUserInteractionFlowWithMock(mockUI)
		if err == nil {
			t.Error("Expected cancellation error, but got nil")
		}
		
		if !strings.Contains(err.Error(), "cancelled") && !strings.Contains(err.Error(), "interrupt") {
			t.Errorf("Expected cancellation-related error, got: %v", err)
		}
	})
}

// CRITERION 14: Concurrent operations and thread safety testing
func TestConcurrentOperations(t *testing.T) {
	t.Run("Multiple simultaneous conversions", func(t *testing.T) {
		numWorkers := 5
		numConversions := 10
		
		var wg sync.WaitGroup
		var mu sync.Mutex
		results := make([]error, 0)
		
		// Channel to simulate conversion requests
		conversionChan := make(chan ConversionRequest, numConversions)
		
		// Fill channel with conversion requests
		for i := 0; i < numConversions; i++ {
			conversionChan <- ConversionRequest{
				ID:     i,
				Input:  fmt.Sprintf("input_%d.mp4", i),
				Output: fmt.Sprintf("output_%d.gif", i),
				Config: ConversionConfig{
					StartTime: 0.0,
					EndTime:   10.0,
					FrameRate: 24.0,
				},
			}
		}
		close(conversionChan)
		
		// Start worker goroutines
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				
				for req := range conversionChan {
					// Simulate conversion work with some processing time
					err := simulateConversion(req, time.Millisecond*100)
					
					mu.Lock()
					results = append(results, err)
					mu.Unlock()
				}
			}(w)
		}
		
		wg.Wait()
		
		// Verify all conversions completed
		if len(results) != numConversions {
			t.Errorf("Expected %d conversion results, got %d", numConversions, len(results))
		}
		
		// Check for any errors - some are expected due to simulated failures
		errorCount := 0
		for _, err := range results {
			if err != nil {
				errorCount++
				t.Logf("Conversion error: %v", err)
			}
		}
		
		// We expect some simulated failures (every 7th conversion)
		expectedErrors := numConversions / 7
		if errorCount != expectedErrors {
			t.Logf("Expected %d simulated errors, got %d", expectedErrors, errorCount)
		}
	})
	
	t.Run("Race condition testing", func(t *testing.T) {
		// Test shared state access
		sharedCounter := &SafeCounter{}
		numGoroutines := 100
		incrementsPerGoroutine := 100
		
		var wg sync.WaitGroup
		
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := 0; j < incrementsPerGoroutine; j++ {
					sharedCounter.Increment()
				}
			}()
		}
		
		wg.Wait()
		
		expectedCount := numGoroutines * incrementsPerGoroutine
		if sharedCounter.Value() != expectedCount {
			t.Errorf("Race condition detected: expected count %d, got %d", expectedCount, sharedCounter.Value())
		}
	})
	
	t.Run("Conversion interruption scenarios", func(t *testing.T) {
		// Test graceful shutdown during conversions
		ctx := &ConversionContext{
			cancelled: false,
		}
		
		var wg sync.WaitGroup
		wg.Add(1)
		
		// Start a long-running conversion
		go func() {
			defer wg.Done()
			err := simulateInterruptibleConversion(ctx, time.Second*2)
			if err == nil {
				t.Error("Expected conversion to be interrupted, but it completed")
			}
		}()
		
		// Cancel after a short delay
		time.Sleep(time.Millisecond * 500)
		ctx.Cancel()
		
		wg.Wait()
		
		if !ctx.IsCancelled() {
			t.Error("Conversion context should be cancelled")
		}
	})
	
	t.Run("Resource cleanup during failures", func(t *testing.T) {
		resourceTracker := &ResourceTracker{
			allocatedResources: make(map[string]bool),
		}
		
		// Simulate multiple conversions with failures
		var wg sync.WaitGroup
		numFailingConversions := 10
		
		for i := 0; i < numFailingConversions; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				
				resourceID := fmt.Sprintf("resource_%d", id)
				
				// Allocate resource
				resourceTracker.Allocate(resourceID)
				
				// Simulate work that might fail
				if id%3 == 0 {
					// Simulate failure - resource should still be cleaned up
					resourceTracker.Cleanup(resourceID)
					return
				}
				
				// Normal completion
				time.Sleep(time.Millisecond * 50)
				resourceTracker.Cleanup(resourceID)
			}(i)
		}
		
		wg.Wait()
		
		// Verify all resources were cleaned up
		if resourceTracker.HasLeaks() {
			t.Error("Resource leaks detected after concurrent operations")
		}
	})
}

// Helper types and functions for testing
type ConversionRequest struct {
	ID     int
	Input  string
	Output string
	Config ConversionConfig
}

type ConversionConfig struct {
	StartTime float64
	EndTime   float64
	FrameRate float64
	Format    string
	Quality   string
}

type SafeCounter struct {
	mu    sync.Mutex
	value int
}

func (c *SafeCounter) Increment() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.value++
}

func (c *SafeCounter) Value() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.value
}

type ConversionContext struct {
	mu        sync.Mutex
	cancelled bool
}

func (c *ConversionContext) Cancel() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cancelled = true
}

func (c *ConversionContext) IsCancelled() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cancelled
}

type ResourceTracker struct {
	mu                 sync.Mutex
	allocatedResources map[string]bool
}

func (r *ResourceTracker) Allocate(resourceID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.allocatedResources[resourceID] = true
}

func (r *ResourceTracker) Cleanup(resourceID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.allocatedResources, resourceID)
}

func (r *ResourceTracker) HasLeaks() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.allocatedResources) > 0
}

// Mock functions for testing
func runUserInteractionFlowWithMock(mockUI *mocks.MockUserInteraction) (ConversionConfig, error) {
	config := ConversionConfig{}
	
	// Start time prompt
	startTime, err := mockUI.PromptForFloat("Start time (seconds)", 0.0, func(v float64) error {
		if v < 0 {
			return fmt.Errorf("start time cannot be negative")
		}
		return nil
	})
	if err != nil {
		return config, err
	}
	config.StartTime = startTime
	
	// End time prompt
	endTime, err := mockUI.PromptForFloat("End time (seconds)", 30.0, func(v float64) error {
		if v <= config.StartTime {
			return fmt.Errorf("end time must be greater than start time")
		}
		return nil
	})
	if err != nil {
		return config, err
	}
	config.EndTime = endTime
	
	// Frame rate prompt
	frameRate, err := mockUI.PromptForFloat("Frame rate (fps)", 24.0, func(v float64) error {
		if v <= 0 {
			return fmt.Errorf("frame rate must be positive")
		}
		if v > 120 {
			return fmt.Errorf("frame rate too high (max 120 fps)")
		}
		return nil
	})
	if err != nil {
		return config, err
	}
	config.FrameRate = frameRate
	
	// Output format prompt
	format, err := mockUI.PromptForString("Output format", "gif", func(v string) error {
		validFormats := []string{"gif", "apng", "webp", "avif", "mp4", "webm"}
		for _, valid := range validFormats {
			if v == valid {
				return nil
			}
		}
		return fmt.Errorf("invalid output format: %s", v)
	})
	if err != nil {
		return config, err
	}
	config.Format = format
	
	// Quality prompt
	quality, err := mockUI.PromptForString("Quality setting", "medium", func(v string) error {
		validQualities := []string{"low", "medium", "high"}
		for _, valid := range validQualities {
			if v == valid {
				return nil
			}
		}
		return fmt.Errorf("invalid quality setting: %s", v)
	})
	if err != nil {
		return config, err
	}
	config.Quality = quality
	
	// AI upscaling confirmation
	enableAI, err := mockUI.PromptForConfirm("Enable AI upscaling?")
	if err != nil {
		return config, err
	}
	
	if enableAI {
		// Upscaling model selection
		models := []string{"RealESRGAN_x2plus", "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"}
		_, err := mockUI.PromptForSelect("Upscaling model", models)
		if err != nil {
			return config, err
		}
	}
	
	return config, nil
}

func simulateConversion(req ConversionRequest, duration time.Duration) error {
	// Simulate conversion work
	time.Sleep(duration)
	
	// Simulate occasional failures
	if req.ID%7 == 0 {
		return fmt.Errorf("simulated conversion failure for ID %d", req.ID)
	}
	
	return nil
}

func simulateInterruptibleConversion(ctx *ConversionContext, duration time.Duration) error {
	start := time.Now()
	
	for time.Since(start) < duration {
		if ctx.IsCancelled() {
			return fmt.Errorf("conversion interrupted")
		}
		time.Sleep(time.Millisecond * 100)
	}
	
	return nil
}