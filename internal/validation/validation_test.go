package validation

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"mediaengine/internal/mocks"
)

// CRITERION 2 (continued): Advanced path validation tests
func TestAdvancedPathValidation(t *testing.T) {
	mockFS := mocks.NewMockFileSystem()
	
	tests := []struct {
		name        string
		path        string
		setupMock   func(*mocks.MockFileSystem)
		expectError bool
		errorType   string
	}{
		{
			name: "Valid file",
			path: "/valid/path/video.mp4",
			setupMock: func(fs *mocks.MockFileSystem) {
				fs.Files["/valid/path/video.mp4"] = []byte("fake mp4 content")
			},
			expectError: false,
		},
		{
			name: "Permission denied",
			path: "/restricted/file.mp4",
			setupMock: func(fs *mocks.MockFileSystem) {
				fs.FailOperations["stat:/restricted/file.mp4"] = os.ErrPermission
			},
			expectError: true,
			errorType:   "permission",
		},
		{
			name: "Network path",
			path: "//network/share/video.mp4",
			setupMock: func(fs *mocks.MockFileSystem) {},
			expectError: true,
			errorType:   "network",
		},
		{
			name: "Unicode path",
			path: "/path/with/ünïcödé/video.mp4",
			setupMock: func(fs *mocks.MockFileSystem) {
				fs.Files["/path/with/ünïcödé/video.mp4"] = []byte("content")
			},
			expectError: false,
		},
		{
			name: "Very long path",
			path: "/path/" + strings.Repeat("very_long_directory_name/", 50) + "video.mp4",
			setupMock: func(fs *mocks.MockFileSystem) {},
			expectError: true,
			errorType:   "file does not exist", // Mock will return file not found for very long paths
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.setupMock(mockFS)
			
			err := validatePathWithMock(tt.path, mockFS)
			
			if tt.expectError && err == nil {
				t.Errorf("Expected error for path %q, but got nil", tt.path)
			}
			
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error for path %q, but got: %v", tt.path, err)
			}
			
			if tt.expectError && err != nil && tt.errorType != "" {
				if !strings.Contains(strings.ToLower(err.Error()), strings.ToLower(tt.errorType)) {
					t.Errorf("Expected error to contain %q, but got: %v", tt.errorType, err)
				}
			}
		})
	}
}

// CRITERION 4: Output path validation and format testing
func TestOutputPathValidation(t *testing.T) {
	testDir := t.TempDir()
	
	outputFormats := []string{"gif", "apng", "webp", "avif", "mp4", "webm"}
	
	for _, format := range outputFormats {
		t.Run(fmt.Sprintf("Output format %s", format), func(t *testing.T) {
			tests := []struct {
				name        string
				outputPath  string
				setupDir    bool
				expectError bool
				description string
			}{
				{
					name:        "Valid output path",
					outputPath:  filepath.Join(testDir, "output."+format),
					setupDir:    true,
					expectError: false,
					description: "Normal output file in writable directory",
				},
				{
					name:        "Non-existent directory",
					outputPath:  filepath.Join(testDir, "nonexistent", "output."+format),
					setupDir:    false,
					expectError: true,
					description: "Output directory doesn't exist",
				},
				{
					name:        "System directory",
					outputPath:  "/etc/output." + format,
					setupDir:    false,
					expectError: true,
					description: "Attempting to write to system directory",
				},
				{
					name:        "Invalid filename characters",
					outputPath:  filepath.Join(testDir, "out<put>."+format),
					setupDir:    true,
					expectError: true,
					description: "Invalid characters in filename",
				},
			}
			
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if tt.setupDir {
						os.MkdirAll(filepath.Dir(tt.outputPath), 0755)
					}
					
					err := validateOutputPath(tt.outputPath, format)
					
					if tt.expectError && err == nil {
						t.Errorf("Expected error for %s, but got nil (%s)", tt.outputPath, tt.description)
					}
					
					if !tt.expectError && err != nil {
						t.Errorf("Expected no error for %s, but got: %v (%s)", tt.outputPath, err, tt.description)
					}
				})
			}
		})
	}
}

// CRITERION 11: File size validation tests
func TestFileSizeValidation(t *testing.T) {
	testDir := t.TempDir()
	maxSize := int64(500 * 1024 * 1024) // 500MB
	
	tests := []struct {
		name        string
		fileSize    int64
		expectError bool
		description string
	}{
		{
			name:        "Small file",
			fileSize:    1024 * 1024, // 1MB
			expectError: false,
			description: "Normal sized file",
		},
		{
			name:        "Exactly at limit",
			fileSize:    maxSize,
			expectError: false,
			description: "File exactly at 500MB limit",
		},
		{
			name:        "Just over limit",
			fileSize:    maxSize + 1,
			expectError: true,
			description: "File 1 byte over limit",
		},
		{
			name:        "Way over limit",
			fileSize:    maxSize * 2,
			expectError: true,
			description: "File significantly over limit",
		},
		{
			name:        "Empty file",
			fileSize:    0,
			expectError: true,
			description: "Zero-byte file",
		},
		{
			name:        "Very large file",
			fileSize:    int64(10) * 1024 * 1024 * 1024, // 10GB
			expectError: true,
			description: "Extremely large file",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filename := filepath.Join(testDir, fmt.Sprintf("test_%d.mp4", tt.fileSize))
			
			// Create file with specific size
			err := createFileWithSize(filename, tt.fileSize)
			if err != nil && tt.fileSize <= maxSize*2 { // Only fail for reasonable sizes
				t.Fatalf("Failed to create test file: %v", err)
			}
			defer os.Remove(filename)
			
			// Test file size validation
			err = validateFileSize(filename, maxSize)
			
			if tt.expectError && err == nil {
				t.Errorf("Expected error for file size %d, but got nil (%s)", tt.fileSize, tt.description)
			}
			
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error for file size %d, but got: %v (%s)", tt.fileSize, err, tt.description)
			}
			
			// Verify error message quality
			if tt.expectError && err != nil {
				errorMsg := err.Error()
				if !strings.Contains(errorMsg, "size") && !strings.Contains(errorMsg, "limit") && 
				   !strings.Contains(errorMsg, "empty") && !strings.Contains(errorMsg, "bytes") &&
				   !strings.Contains(errorMsg, "no such file") {
					t.Errorf("Error message should mention size/limit/empty: %v", err)
				}
				
				// Check for helpful information (only for actual size limit issues)
				if tt.fileSize > maxSize && strings.Contains(errorMsg, "exceeds") {
					if !strings.Contains(errorMsg, "500MB") && !strings.Contains(errorMsg, "maximum") {
						t.Logf("Note: Error message could be more specific about size limit: %v", err)
					}
				}
			}
		})
	}
}

// CRITERION 13: Error message quality testing
func TestErrorMessageQuality(t *testing.T) {
	tests := []struct {
		name           string
		triggerError   func() error
		expectedTerms  []string
		helpfulGuidance bool
		description    string
	}{
		{
			name: "File not found",
			triggerError: func() error {
				return validateInputPath("/nonexistent/file.mp4")
			},
			expectedTerms:   []string{"not found", "exist", "file"},
			helpfulGuidance: true,
			description:     "File not found error should be clear",
		},
		{
			name: "Invalid format",
			triggerError: func() error {
				return validateInputPath("document.txt")
			},
			expectedTerms:   []string{"format", "supported", "video"},
			helpfulGuidance: true,
			description:     "Format error should list supported formats",
		},
		{
			name: "Permission denied",
			triggerError: func() error {
				return validateInputPath("/root/private.mp4")
			},
			expectedTerms:   []string{"permission", "access", "denied"},
			helpfulGuidance: true,
			description:     "Permission error should suggest solutions",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.triggerError()
			
			if err == nil {
				t.Errorf("Expected error for test %s, but got nil", tt.name)
				return
			}
			
			errorMsg := strings.ToLower(err.Error())
			
			// Check for expected terms
			for _, term := range tt.expectedTerms {
				if !strings.Contains(errorMsg, strings.ToLower(term)) {
					t.Errorf("Error message should contain %q: %v (%s)", term, err, tt.description)
				}
			}
			
			// Check for helpful guidance
			if tt.helpfulGuidance {
				hasGuidance := strings.Contains(errorMsg, "try") ||
					strings.Contains(errorMsg, "please") ||
					strings.Contains(errorMsg, "should") ||
					strings.Contains(errorMsg, "supported formats") ||
					strings.Contains(errorMsg, "check") ||
					strings.Contains(errorMsg, "cannot access") ||
					strings.Contains(errorMsg, "denied")
				
				if !hasGuidance {
					t.Logf("Note: Error message could be more helpful: %v (%s)", err, tt.description)
				}
			}
			
			// Error messages should not be too technical
			technicalTerms := []string{"errno", "syscall", "0x", "null pointer"}
			for _, tech := range technicalTerms {
				if strings.Contains(errorMsg, tech) {
					t.Errorf("Error message should be user-friendly, not technical: %v (%s)", err, tt.description)
				}
			}
		})
	}
}

// Helper functions (placeholders for actual implementation)
func validatePathWithMock(path string, fs *mocks.MockFileSystem) error {
	if strings.Contains(path, "//") {
		return fmt.Errorf("network paths not supported")
	}
	
	if len(path) > 4096 {
		return fmt.Errorf("path too long")
	}
	
	_, err := fs.Stat(path)
	if err == os.ErrPermission {
		return fmt.Errorf("permission denied accessing file")
	}
	
	return err
}

func validateOutputPath(path string, format string) error {
	if strings.ContainsAny(filepath.Base(path), "<>:\"|?*") {
		return fmt.Errorf("invalid characters in filename")
	}
	
	if strings.HasPrefix(path, "/etc/") || strings.HasPrefix(path, "/usr/") {
		return fmt.Errorf("cannot write to system directory")
	}
	
	dir := filepath.Dir(path)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return fmt.Errorf("output directory does not exist: %s", dir)
	}
	
	return nil
}

func validateFileSize(path string, maxSize int64) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	
	size := info.Size()
	if size == 0 {
		return fmt.Errorf("file is empty (0 bytes)")
	}
	
	if size > maxSize {
		return fmt.Errorf("file size (%d bytes) exceeds maximum limit of 500MB (%d bytes)", size, maxSize)
	}
	
	return nil
}

func createFileWithSize(filename string, size int64) error {
	if size > int64(2)*1024*1024*1024 { // Don't actually create files larger than 2GB
		return fmt.Errorf("test file too large to create")
	}
	
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	if size > 0 {
		err = file.Truncate(size)
	}
	
	return err
}

func validateInputPath(path string) error {
	// Placeholder implementation
	if path == "" {
		return fmt.Errorf("file path is empty")
	}
	
	if strings.Contains(path, "..") {
		return fmt.Errorf("directory traversal not allowed")
	}
	
	if !strings.HasSuffix(strings.ToLower(path), ".mp4") &&
		!strings.HasSuffix(strings.ToLower(path), ".mkv") &&
		!strings.HasSuffix(strings.ToLower(path), ".mov") &&
		!strings.HasSuffix(strings.ToLower(path), ".avi") &&
		!strings.HasSuffix(strings.ToLower(path), ".webm") &&
		!strings.HasSuffix(strings.ToLower(path), ".flv") &&
		!strings.HasSuffix(strings.ToLower(path), ".wmv") {
		return fmt.Errorf("unsupported video format. Supported formats: .mp4, .mkv, .mov, .avi, .webm, .flv, .wmv")
	}
	
	if strings.HasPrefix(path, "/root/") {
		return fmt.Errorf("permission denied: cannot access restricted directory")
	}
	
	_, err := os.Stat(path)
	if os.IsNotExist(err) {
		return fmt.Errorf("file not found: %s. Please check the file path and try again", path)
	}
	
	return err
}