// internal/validation/validation.go
package validation

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	// Maximum file size in bytes (500MB)
	MaxFileSizeBytes = 500 * 1024 * 1024
)

// Add SupportedInputFormats to validation package
var SupportedInputFormats = []string{".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv"}

// getSystemDirectories returns platform-specific system directories to protect
func getSystemDirectories() []string {
	switch runtime.GOOS {
	case "windows":
		return []string{
			"C:\\Windows",
			"C:\\Program Files",
			"C:\\Program Files (x86)",
			"C:\\System Volume Information",
			"C:\\ProgramData",
		}
	case "darwin": // macOS
		return []string{
			"/System",
			"/usr",
			"/bin",
			"/sbin",
			"/etc",
			"/private",
			"/Applications",
		}
	case "linux":
		return []string{
			"/etc",
			"/usr",
			"/bin",
			"/sbin",
			"/boot",
			"/sys",
			"/proc",
			"/root",
		}
	default:
		// Generic Unix-like fallback
		return []string{
			"/etc",
			"/usr",
			"/bin",
			"/sbin",
		}
	}
}

// getMaxPathLength returns platform-specific maximum path length
func getMaxPathLength() int {
	switch runtime.GOOS {
	case "windows":
		return 260 // Windows MAX_PATH (can be 32767 with long path support, but 260 is safer)
	case "darwin":
		return 1024 // macOS typical limit
	case "linux":
		return 4096 // Linux PATH_MAX
	default:
		return 1024 // Conservative default
	}
}

// normalizePathForComparison normalizes paths for cross-platform comparison
func normalizePathForComparison(path string) string {
	// Convert to lowercase on Windows for case-insensitive comparison
	if runtime.GOOS == "windows" {
		return strings.ToLower(filepath.Clean(path))
	}
	return filepath.Clean(path)
}

// ValidateInputPath performs comprehensive validation of input video file paths
func ValidateInputPath(input string) error {
	if strings.TrimSpace(input) == "" {
		return fmt.Errorf("path cannot be empty")
	}

	// Clean the input: trim whitespace and remove surrounding quotes
	cleanedPath := strings.TrimSpace(input)

	// Remove surrounding quotes (single or double) that Finder might add
	if len(cleanedPath) >= 2 {
		if (cleanedPath[0] == '\'' && cleanedPath[len(cleanedPath)-1] == '\'') ||
			(cleanedPath[0] == '"' && cleanedPath[len(cleanedPath)-1] == '"') {
			cleanedPath = cleanedPath[1 : len(cleanedPath)-1]
		}
	}

	// Trim again after removing quotes
	cleanedPath = strings.TrimSpace(cleanedPath)

	if cleanedPath == "" {
		return fmt.Errorf("path cannot be empty after removing quotes")
	}

	// Security: Check for directory traversal attempts
	if strings.Contains(cleanedPath, "..") {
		return fmt.Errorf("path cannot contain '..' (directory traversal)")
	}

	// Convert to absolute path for consistent handling
	absPath, err := filepath.Abs(cleanedPath)
	if err != nil {
		return fmt.Errorf("invalid path format: %v", err)
	}

	// Clean the path to remove any redundant elements
	cleanPath := filepath.Clean(absPath)

	// Check for invalid characters (platform-specific)
	if err := validatePathCharacters(cleanPath); err != nil {
		return err
	}

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

	// Check file extension (support multiple formats)
	ext := strings.ToLower(filepath.Ext(cleanPath))
	validExtension := false
	for _, supportedExt := range SupportedInputFormats {
		if ext == supportedExt {
			validExtension = true
			break
		}
	}

	if !validExtension {
		return fmt.Errorf("unsupported file format: %s. Supported formats: %s",
			ext, strings.Join(SupportedInputFormats, ", "))
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

	// Clean the input: trim whitespace and remove surrounding quotes
	cleanedPath := strings.TrimSpace(input)

	// Remove surrounding quotes (single or double) that Finder might add
	if len(cleanedPath) >= 2 {
		if (cleanedPath[0] == '\'' && cleanedPath[len(cleanedPath)-1] == '\'') ||
			(cleanedPath[0] == '"' && cleanedPath[len(cleanedPath)-1] == '"') {
			cleanedPath = cleanedPath[1 : len(cleanedPath)-1]
		}
	}

	// Trim again after removing quotes
	cleanedPath = strings.TrimSpace(cleanedPath)

	if cleanedPath == "" {
		return fmt.Errorf("path cannot be empty after removing quotes")
	}

	// Security: Check for directory traversal attempts
	if strings.Contains(cleanedPath, "..") {
		return fmt.Errorf("path cannot contain '..' (directory traversal)")
	}

	// Convert to absolute path
	absPath, err := filepath.Abs(cleanedPath)
	if err != nil {
		return fmt.Errorf("invalid path format: %v", err)
	}

	// Clean the path
	cleanPath := filepath.Clean(absPath)

	// Check for invalid characters (platform-specific)
	if err := validatePathCharacters(cleanPath); err != nil {
		return err
	}

	// Check if it's an existing directory - this is valid, we'll append output.gif
	if stat, err := os.Stat(cleanPath); err == nil && stat.IsDir() {
		return nil // Directory is valid, we'll handle appending filename later
	}

	// If it's not an existing directory, treat it as a file path
	// Ensure it will have .gif extension for validation purposes
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
	// Clean the path first (handle quotes like in basic validation)
	cleanedPath := strings.TrimSpace(outputPath)

	// Remove surrounding quotes (single or double) that Finder might add
	if len(cleanedPath) >= 2 {
		if (cleanedPath[0] == '\'' && cleanedPath[len(cleanedPath)-1] == '\'') ||
			(cleanedPath[0] == '"' && cleanedPath[len(cleanedPath)-1] == '"') {
			cleanedPath = cleanedPath[1 : len(cleanedPath)-1]
		}
	}

	// Trim again after removing quotes
	cleanedPath = strings.TrimSpace(cleanedPath)

	// First run basic validation on the cleaned path
	if err := ValidateOutputPathBasic(cleanedPath); err != nil {
		return err
	}

	// Convert to absolute path
	absPath, err := filepath.Abs(cleanedPath)
	if err != nil {
		return fmt.Errorf("invalid path format: %v", err)
	}

	cleanPath := filepath.Clean(absPath)

	// Check for invalid characters (platform-specific)
	if err := validatePathCharacters(cleanPath); err != nil {
		return err
	}

	// Special check: if path looks like a directory but doesn't exist
	if err := checkForNonExistentDirectory(outputPath, cleanPath); err != nil {
		return err
	}

	// Check if it's an existing directory - append output.gif
	if stat, err := os.Stat(cleanPath); err == nil && stat.IsDir() {
		cleanPath = filepath.Join(cleanPath, "output.gif")
	} else if !strings.HasSuffix(strings.ToLower(cleanPath), ".gif") {
		// If it's not a directory and doesn't end with .gif, add .gif extension
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

	// Check for invalid characters (platform-specific)
	if err := validatePathCharacters(absPath); err != nil {
		return err
	}

	// Check path length (platform-specific limits)
	maxLen := getMaxPathLength()
	if len(absPath) > maxLen {
		return fmt.Errorf("path too long (max %d characters)", maxLen)
	}

	// Ensure path doesn't try to escape to system directories (platform-specific)
	systemDirs := getSystemDirectories()
	normalizedPath := normalizePathForComparison(absPath)

	for _, sysDir := range systemDirs {
		normalizedSysDir := normalizePathForComparison(sysDir)
		if strings.HasPrefix(normalizedPath, normalizedSysDir) {
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

// validatePathCharacters checks for invalid characters based on OS
func validatePathCharacters(path string) error {
	if runtime.GOOS == "windows" {
		// Windows reserved characters
		invalidChars := []string{"<", ">", ":", "\"", "|", "?", "*"}
		for _, char := range invalidChars {
			if strings.Contains(path, char) {
				return fmt.Errorf("path contains invalid character: %s", char)
			}
		}

		// Windows reserved names (case-insensitive)
		baseName := strings.ToUpper(filepath.Base(path))
		// Remove extension for comparison
		if idx := strings.LastIndex(baseName, "."); idx != -1 {
			baseName = baseName[:idx]
		}

		reservedNames := []string{
			"CON", "PRN", "AUX", "NUL",
			"COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
			"LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
		}

		for _, reserved := range reservedNames {
			if baseName == reserved {
				return fmt.Errorf("path uses reserved Windows name: %s", reserved)
			}
		}
	}

	// Universal checks for all platforms
	if strings.Contains(path, "\x00") {
		return fmt.Errorf("path contains null bytes")
	}

	return nil
}

// checkForNonExistentDirectory checks if a path looks like a directory but doesn't exist
func checkForNonExistentDirectory(originalPath, cleanedPath string) error {
	// First check if the path exists
	if _, err := os.Stat(cleanedPath); err == nil {
		// Path exists, no issue here
		return nil
	} else if !os.IsNotExist(err) {
		// Some other error accessing the path
		return fmt.Errorf("cannot access path: %v", err)
	}

	// Path doesn't exist - check if it looks like the user intended it to be a directory
	if looksLikeDirectory(originalPath) {
		return fmt.Errorf("path '%s' looks like a directory but doesn't exist. Please either:\n  • Create the directory first, or\n  • Provide a filename (e.g., 'my_video.gif')", originalPath)
	}

	return nil
}

// looksLikeDirectory determines if a path looks like it's intended to be a directory
func looksLikeDirectory(path string) bool {
	// Remove leading ./ if present
	cleanPath := strings.TrimPrefix(path, "./")
	cleanPath = strings.TrimPrefix(cleanPath, ".\\") // Windows

	// Check for directory-like indicators:
	// 1. Ends with a path separator
	if strings.HasSuffix(path, "/") || strings.HasSuffix(path, "\\") {
		return true
	}

	// 2. Contains path separators (likely a subdirectory)
	if strings.Contains(cleanPath, "/") || strings.Contains(cleanPath, "\\") {
		return true
	}

	// 3. No file extension and looks like a folder name
	if !strings.Contains(filepath.Base(cleanPath), ".") {
		// Common directory-like names
		lowerBase := strings.ToLower(filepath.Base(cleanPath))
		dirWords := []string{"gifs", "videos", "output", "exports", "files", "media", "images", "tmp", "temp", "data", "assets"}
		for _, word := range dirWords {
			if lowerBase == word || strings.Contains(lowerBase, word) {
				return true
			}
		}
	}

	return false
}
