// internal/validation/validation.go
package validation

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	// Maximum file size in bytes (500MB)
	MaxFileSizeBytes = 500 * 1024 * 1024
)

// ValidateInputPath performs comprehensive validation of input video file paths
func ValidateInputPath(input string) error {
	if strings.TrimSpace(input) == "" {
		return fmt.Errorf("path cannot be empty")
	}

	path := strings.TrimSpace(input)

	// Security: Check for directory traversal attempts
	if strings.Contains(path, "..") {
		return fmt.Errorf("path cannot contain '..' (directory traversal)")
	}

	// Convert to absolute path for consistent handling
	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("invalid path format: %v", err)
	}

	// Clean the path to remove any redundant elements
	cleanPath := filepath.Clean(absPath)

	// Check if file exists
	fileInfo, err := os.Stat(cleanPath)
	if os.IsNotExist(err) {
		return fmt.Errorf("file does not exist: %s", cleanPath)
	}
	if err != nil {
		return fmt.Errorf("cannot access file: %v", err)
	}

	// Check if it's a file (not a directory)
	if fileInfo.IsDir() {
		return fmt.Errorf("path points to a directory, not a file: %s", cleanPath)
	}

	// Check file extension (must be .mp4)
	ext := strings.ToLower(filepath.Ext(cleanPath))
	if ext != ".mp4" {
		return fmt.Errorf("file must have .mp4 extension, got: %s", ext)
	}

	// Check file size
	if fileInfo.Size() == 0 {
		return fmt.Errorf("file is empty")
	}

	if fileInfo.Size() > MaxFileSizeBytes {
		sizeMB := float64(fileInfo.Size()) / (1024 * 1024)
		return fmt.Errorf("file size (%.1f MB) exceeds maximum allowed size of %d MB", sizeMB, MaxFileSizeBytes/(1024*1024))
	}

	// Check read permissions
	file, err := os.Open(cleanPath)
	if err != nil {
		return fmt.Errorf("cannot read file (permission denied): %v", err)
	}
	file.Close()

	return nil
}

// ValidateOutputPathBasic performs basic validation during input prompt
func ValidateOutputPathBasic(input string) error {
	if strings.TrimSpace(input) == "" {
		return fmt.Errorf("output path cannot be empty")
	}

	path := strings.TrimSpace(input)

	// Security: Check for directory traversal attempts
	if strings.Contains(path, "..") {
		return fmt.Errorf("path cannot contain '..' (directory traversal)")
	}

	// Convert to absolute path
	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("invalid path format: %v", err)
	}

	// Clean the path
	cleanPath := filepath.Clean(absPath)

	// Ensure it will have .gif extension
	if !strings.HasSuffix(strings.ToLower(cleanPath), ".gif") {
		cleanPath += ".gif"
	}

	// Check if parent directory exists
	parentDir := filepath.Dir(cleanPath)
	if parentInfo, err := os.Stat(parentDir); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("output directory does not exist: %s", parentDir)
		}
		return fmt.Errorf("cannot access output directory: %v", err)
	} else if !parentInfo.IsDir() {
		return fmt.Errorf("output parent path is not a directory: %s", parentDir)
	}

	return nil
}

// ValidateOutputPath performs comprehensive validation of output path
func ValidateOutputPath(outputPath string) error {
	// First run basic validation
	if err := ValidateOutputPathBasic(outputPath); err != nil {
		return err
	}

	cleanPath := filepath.Clean(outputPath)

	// Ensure .gif extension
	if !strings.HasSuffix(strings.ToLower(cleanPath), ".gif") {
		cleanPath += ".gif"
	}

	// Check if file already exists
	if fileInfo, err := os.Stat(cleanPath); err == nil {
		if fileInfo.IsDir() {
			return fmt.Errorf("output path points to an existing directory: %s", cleanPath)
		}
		// File exists - this is okay, we'll overwrite it
		// But check if we can write to it
		if err := checkWritePermission(cleanPath); err != nil {
			return fmt.Errorf("cannot write to existing file: %v", err)
		}
	}

	// Check write permission in parent directory
	parentDir := filepath.Dir(cleanPath)
	if err := checkWritePermission(parentDir); err != nil {
		return fmt.Errorf("cannot write to output directory: %v", err)
	}

	// Security: Ensure the final path is within expected bounds
	if err := validatePathSecurity(cleanPath); err != nil {
		return fmt.Errorf("security validation failed: %v", err)
	}

	return nil
}

// checkWritePermission tests if we can write to a file or directory
func checkWritePermission(path string) error {
	// Try to create a temporary file to test write permissions
	tempFile := filepath.Join(path, ".videotogif_write_test")
	if _, err := os.Stat(path); err != nil {
		// If path is a file, check its directory
		if !os.IsNotExist(err) {
			return err
		}
		tempFile = filepath.Join(filepath.Dir(path), ".videotogif_write_test")
	}

	file, err := os.Create(tempFile)
	if err != nil {
		return fmt.Errorf("no write permission: %v", err)
	}
	file.Close()
	os.Remove(tempFile) // Clean up

	return nil
}

// validatePathSecurity performs additional security checks
func validatePathSecurity(path string) error {
	// Get absolute path
	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("cannot resolve absolute path: %v", err)
	}

	// Check for null bytes (security vulnerability)
	if strings.Contains(absPath, "\x00") {
		return fmt.Errorf("path contains null bytes")
	}

	// Check path length (avoid extremely long paths that could cause issues)
	if len(absPath) > 4096 {
		return fmt.Errorf("path too long (max 4096 characters)")
	}

	// Ensure path doesn't try to escape to system directories
	systemDirs := []string{"/etc", "/usr", "/bin", "/sbin", "/boot", "/sys", "/proc"}
	for _, sysDir := range systemDirs {
		if strings.HasPrefix(absPath, sysDir) {
			return fmt.Errorf("cannot write to system directory: %s", sysDir)
		}
	}

	return nil
}

// ValidateVideoFile performs additional validation specifically for video files
func ValidateVideoFile(path string) error {
	// Check if file can be opened for reading
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("cannot open video file: %v", err)
	}
	defer file.Close()

	// Read first few bytes to check if it looks like a valid MP4
	header := make([]byte, 12)
	n, err := file.Read(header)
	if err != nil || n < 12 {
		return fmt.Errorf("cannot read file header")
	}

	// Basic MP4 signature check (ftyp box)
	if n >= 8 && string(header[4:8]) != "ftyp" {
		return fmt.Errorf("file does not appear to be a valid MP4 video")
	}

	return nil
}

// SanitizePath cleans and validates a file path
func SanitizePath(path string) (string, error) {
	// Trim whitespace
	cleaned := strings.TrimSpace(path)

	// Remove any null bytes
	cleaned = strings.ReplaceAll(cleaned, "\x00", "")

	// Convert to absolute path
	absPath, err := filepath.Abs(cleaned)
	if err != nil {
		return "", fmt.Errorf("invalid path: %v", err)
	}

	// Clean the path (remove . and .. elements)
	cleanPath := filepath.Clean(absPath)

	return cleanPath, nil
}
