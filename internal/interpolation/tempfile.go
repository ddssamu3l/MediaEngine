// internal/interpolation/tempfile.go
package interpolation

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"time"
)

// TempFileManager handles sophisticated temporary file management for frame interpolation
type TempFileManager struct {
	baseDir         string
	sessionID       string
	maxDiskUsageGB  float64
	cleanupInterval time.Duration
	frameBatches    map[string]*FrameBatch
	totalFrames     int
}

// FrameBatch represents a batch of frames for processing
type FrameBatch struct {
	ID           string
	InputDir     string
	OutputDir    string
	FrameFiles   []string
	ProcessedAt  time.Time
	SizeBytes    int64
	IsProcessed  bool
}

// DiskUsage represents disk space information
type DiskUsage struct {
	TotalGB     float64
	UsedGB      float64
	AvailableGB float64
	UsagePercent float64
}

// NewTempFileManager creates a new temporary file manager
func NewTempFileManager(baseDir string, maxDiskUsageGB float64) *TempFileManager {
	sessionID := fmt.Sprintf("interpolation_%d", time.Now().Unix())
	
	return &TempFileManager{
		baseDir:         baseDir,
		sessionID:       sessionID,
		maxDiskUsageGB:  maxDiskUsageGB,
		cleanupInterval: 30 * time.Second,
		frameBatches:    make(map[string]*FrameBatch),
		totalFrames:     0,
	}
}

// CreateSessionDir creates a session-specific temporary directory
func (tfm *TempFileManager) CreateSessionDir() (string, error) {
	sessionDir := filepath.Join(tfm.baseDir, tfm.sessionID)
	
	if err := os.MkdirAll(sessionDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create session directory: %v", err)
	}
	
	// Create subdirectories for different processing stages
	subdirs := []string{"input_frames", "interpolated_frames", "output_frames", "audio", "temp"}
	for _, subdir := range subdirs {
		if err := os.MkdirAll(filepath.Join(sessionDir, subdir), 0755); err != nil {
			return "", fmt.Errorf("failed to create subdirectory %s: %v", subdir, err)
		}
	}
	
	return sessionDir, nil
}

// CreateFrameBatch creates a new frame batch for processing
func (tfm *TempFileManager) CreateFrameBatch(batchID string, sessionDir string) (*FrameBatch, error) {
	inputDir := filepath.Join(sessionDir, "input_frames", batchID)
	outputDir := filepath.Join(sessionDir, "interpolated_frames", batchID)
	
	// Create batch directories
	if err := os.MkdirAll(inputDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create input batch directory: %v", err)
	}
	
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output batch directory: %v", err)
	}
	
	batch := &FrameBatch{
		ID:          batchID,
		InputDir:    inputDir,
		OutputDir:   outputDir,
		FrameFiles:  make([]string, 0),
		ProcessedAt: time.Now(),
		SizeBytes:   0,
		IsProcessed: false,
	}
	
	tfm.frameBatches[batchID] = batch
	return batch, nil
}

// AddFramesToBatch adds frame files to a processing batch
func (tfm *TempFileManager) AddFramesToBatch(batchID string, frameFiles []string) error {
	batch, exists := tfm.frameBatches[batchID]
	if !exists {
		return fmt.Errorf("batch %s not found", batchID)
	}
	
	// Calculate batch size
	var batchSize int64
	for _, frameFile := range frameFiles {
		if stat, err := os.Stat(frameFile); err == nil {
			batchSize += stat.Size()
		}
	}
	
	batch.FrameFiles = append(batch.FrameFiles, frameFiles...)
	batch.SizeBytes += batchSize
	tfm.totalFrames += len(frameFiles)
	
	return nil
}

// MarkBatchProcessed marks a batch as processed and eligible for cleanup
func (tfm *TempFileManager) MarkBatchProcessed(batchID string) error {
	batch, exists := tfm.frameBatches[batchID]
	if !exists {
		return fmt.Errorf("batch %s not found", batchID)
	}
	
	batch.IsProcessed = true
	batch.ProcessedAt = time.Now()
	
	return nil
}

// GetDiskUsage returns current disk usage information
func (tfm *TempFileManager) GetDiskUsage() (*DiskUsage, error) {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(tfm.baseDir, &stat); err != nil {
		return nil, fmt.Errorf("failed to get disk usage: %v", err)
	}
	
	totalBytes := stat.Blocks * uint64(stat.Bsize)
	freeBytes := stat.Bavail * uint64(stat.Bsize)
	usedBytes := totalBytes - freeBytes
	
	totalGB := float64(totalBytes) / (1024 * 1024 * 1024)
	usedGB := float64(usedBytes) / (1024 * 1024 * 1024)
	availableGB := float64(freeBytes) / (1024 * 1024 * 1024)
	usagePercent := (usedGB / totalGB) * 100
	
	return &DiskUsage{
		TotalGB:      totalGB,
		UsedGB:       usedGB,
		AvailableGB:  availableGB,
		UsagePercent: usagePercent,
	}, nil
}

// CheckDiskSpace verifies if there's enough disk space for processing
func (tfm *TempFileManager) CheckDiskSpace(estimatedUsageGB float64) (bool, string, error) {
	diskUsage, err := tfm.GetDiskUsage()
	if err != nil {
		return false, "", err
	}
	
	if estimatedUsageGB > diskUsage.AvailableGB {
		return false, fmt.Sprintf("Insufficient disk space: need %.1fGB, available %.1fGB", 
			estimatedUsageGB, diskUsage.AvailableGB), nil
	}
	
	// Check if usage would exceed the configured limit
	projectedUsage := diskUsage.UsedGB + estimatedUsageGB
	if projectedUsage > tfm.maxDiskUsageGB && tfm.maxDiskUsageGB > 0 {
		return false, fmt.Sprintf("Would exceed disk usage limit: %.1fGB (limit: %.1fGB)", 
			projectedUsage, tfm.maxDiskUsageGB), nil
	}
	
	return true, fmt.Sprintf("Sufficient disk space: %.1fGB available", diskUsage.AvailableGB), nil
}

// EstimateFrameStorageNeeds estimates disk space needed for frame processing
func (tfm *TempFileManager) EstimateFrameStorageNeeds(width, height, frameCount, multiplier int) float64 {
	// Estimate frame size based on resolution
	// PNG frames are typically larger than compressed video frames
	bytesPerPixel := 3 // RGB
	avgCompressionRatio := 0.7 // PNG compression
	
	frameSize := float64(width * height * bytesPerPixel) * avgCompressionRatio
	
	// Calculate storage needs:
	// - Input frames (extracted from video)
	// - Interpolated frames (multiplier * input frames)
	// - Temporary processing overhead (50% extra)
	inputFramesGB := (frameSize * float64(frameCount)) / (1024 * 1024 * 1024)
	interpolatedFramesGB := (frameSize * float64(frameCount) * float64(multiplier)) / (1024 * 1024 * 1024)
	overheadGB := (inputFramesGB + interpolatedFramesGB) * 0.5
	
	totalGB := inputFramesGB + interpolatedFramesGB + overheadGB
	
	return totalGB
}

// CleanupProcessedBatches removes processed batches to free disk space
func (tfm *TempFileManager) CleanupProcessedBatches(olderThan time.Duration) error {
	cleanupTime := time.Now().Add(-olderThan)
	
	for batchID, batch := range tfm.frameBatches {
		if batch.IsProcessed && batch.ProcessedAt.Before(cleanupTime) {
			// Remove input frames (keep output frames until final cleanup)
			if err := os.RemoveAll(batch.InputDir); err != nil {
				fmt.Printf("Warning: failed to cleanup input batch %s: %v\n", batchID, err)
			}
			
			// Update batch to reflect cleanup
			batch.FrameFiles = nil
			fmt.Printf("Cleaned up processed batch %s (freed space)\n", batchID)
		}
	}
	
	return nil
}

// StartPeriodicCleanup starts a background cleanup routine
func (tfm *TempFileManager) StartPeriodicCleanup() chan bool {
	stopChan := make(chan bool)
	
	go func() {
		ticker := time.NewTicker(tfm.cleanupInterval)
		defer ticker.Stop()
		
		for {
			select {
			case <-ticker.C:
				// Cleanup batches older than 5 minutes
				if err := tfm.CleanupProcessedBatches(5 * time.Minute); err != nil {
					fmt.Printf("Periodic cleanup error: %v\n", err)
				}
				
				// Check disk usage and warn if high
				if diskUsage, err := tfm.GetDiskUsage(); err == nil {
					if diskUsage.UsagePercent > 90 {
						fmt.Printf("Warning: High disk usage %.1f%%\n", diskUsage.UsagePercent)
					}
				}
				
			case <-stopChan:
				return
			}
		}
	}()
	
	return stopChan
}

// GetBatchStatus returns status information about all batches
func (tfm *TempFileManager) GetBatchStatus() map[string]interface{} {
	status := make(map[string]interface{})
	
	totalBatches := len(tfm.frameBatches)
	processedBatches := 0
	totalSize := int64(0)
	
	for _, batch := range tfm.frameBatches {
		if batch.IsProcessed {
			processedBatches++
		}
		totalSize += batch.SizeBytes
	}
	
	status["total_batches"] = totalBatches
	status["processed_batches"] = processedBatches
	status["pending_batches"] = totalBatches - processedBatches
	status["total_frames"] = tfm.totalFrames
	status["total_size_gb"] = float64(totalSize) / (1024 * 1024 * 1024)
	status["session_id"] = tfm.sessionID
	
	return status
}

// CleanupSession removes all session data
func (tfm *TempFileManager) CleanupSession() error {
	sessionDir := filepath.Join(tfm.baseDir, tfm.sessionID)
	
	if err := os.RemoveAll(sessionDir); err != nil {
		return fmt.Errorf("failed to cleanup session %s: %v", tfm.sessionID, err)
	}
	
	// Clear internal state
	tfm.frameBatches = make(map[string]*FrameBatch)
	tfm.totalFrames = 0
	
	return nil
}

// OptimizeBatchSize calculates optimal batch size based on available resources
func (tfm *TempFileManager) OptimizeBatchSize(frameCount, width, height int, availableMemoryGB float64) int {
	// Calculate memory per frame (in GB)
	bytesPerPixel := 4 // RGBA for processing
	frameMemoryGB := float64(width*height*bytesPerPixel) / (1024 * 1024 * 1024)
	
	// Reserve memory for model and processing overhead
	reservedMemoryGB := 2.0 // Reserve 2GB for model and overhead
	usableMemoryGB := availableMemoryGB - reservedMemoryGB
	
	if usableMemoryGB <= 0 {
		return 1 // Minimum batch size
	}
	
	// Calculate how many frames fit in memory
	maxFramesInMemory := int(usableMemoryGB / frameMemoryGB)
	
	// For interpolation, we need space for input frames + interpolated frames
	// Assume 2x interpolation as baseline
	maxBatchSize := maxFramesInMemory / 4 // Conservative estimate
	
	// Ensure reasonable bounds
	if maxBatchSize < 1 {
		maxBatchSize = 1
	} else if maxBatchSize > 32 {
		maxBatchSize = 32 // Don't make batches too large
	}
	
	// Don't exceed total frame count
	if maxBatchSize > frameCount {
		maxBatchSize = frameCount
	}
	
	return maxBatchSize
}

// CreateFrameSymlinks creates symbolic links to frames for memory-efficient batch processing
func (tfm *TempFileManager) CreateFrameSymlinks(sourceFrames []string, targetDir string) error {
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("failed to create target directory: %v", err)
	}
	
	for i, sourceFrame := range sourceFrames {
		targetFrame := filepath.Join(targetDir, fmt.Sprintf("frame_%06d.png", i))
		
		// Create symbolic link instead of copying for memory efficiency
		if err := os.Symlink(sourceFrame, targetFrame); err != nil {
			// Fall back to copying if symlinks aren't supported
			return tfm.copyFile(sourceFrame, targetFrame)
		}
	}
	
	return nil
}

// copyFile copies a file (fallback when symlinks aren't available)
func (tfm *TempFileManager) copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()
	
	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()
	
	_, err = destFile.ReadFrom(sourceFile)
	return err
}

// ListOldSessions finds and lists old interpolation sessions for cleanup
func ListOldSessions(baseDir string, olderThan time.Duration) ([]string, error) {
	cutoffTime := time.Now().Add(-olderThan)
	var oldSessions []string
	
	entries, err := os.ReadDir(baseDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read base directory: %v", err)
	}
	
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "interpolation_") {
			info, err := entry.Info()
			if err != nil {
				continue
			}
			
			if info.ModTime().Before(cutoffTime) {
				oldSessions = append(oldSessions, filepath.Join(baseDir, entry.Name()))
			}
		}
	}
	
	sort.Strings(oldSessions)
	return oldSessions, nil
}

// CleanupOldSessions removes old interpolation sessions
func CleanupOldSessions(baseDir string, olderThan time.Duration) error {
	oldSessions, err := ListOldSessions(baseDir, olderThan)
	if err != nil {
		return err
	}
	
	for _, sessionDir := range oldSessions {
		if err := os.RemoveAll(sessionDir); err != nil {
			fmt.Printf("Warning: failed to remove old session %s: %v\n", sessionDir, err)
		} else {
			fmt.Printf("Cleaned up old session: %s\n", sessionDir)
		}
	}
	
	return nil
}

// GetMemoryUsage returns current memory usage information
func GetMemoryUsage() map[string]interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	return map[string]interface{}{
		"alloc_gb":      float64(m.Alloc) / (1024 * 1024 * 1024),
		"total_alloc_gb": float64(m.TotalAlloc) / (1024 * 1024 * 1024),
		"sys_gb":        float64(m.Sys) / (1024 * 1024 * 1024),
		"num_gc":        m.NumGC,
		"gc_cpu_fraction": m.GCCPUFraction,
	}
}