// Package mocks provides mock implementations for testing
package mocks

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// MockFileSystem provides a mock file system for testing
type MockFileSystem struct {
	Files         map[string][]byte
	Dirs          map[string]bool
	Permissions   map[string]os.FileMode
	FailOperations map[string]error
	FileInfos     map[string]MockFileInfo
}

// MockFileInfo implements os.FileInfo for testing
type MockFileInfo struct {
	FileName    string
	FileSize    int64
	FileMode    os.FileMode
	FileModTime time.Time
	FileIsDir   bool
}

func (m MockFileInfo) Name() string       { return m.FileName }
func (m MockFileInfo) Size() int64        { return m.FileSize }
func (m MockFileInfo) Mode() os.FileMode  { return m.FileMode }
func (m MockFileInfo) ModTime() time.Time { return m.FileModTime }
func (m MockFileInfo) IsDir() bool        { return m.FileIsDir }
func (m MockFileInfo) Sys() interface{}   { return nil }

// NewMockFileSystem creates a new mock file system
func NewMockFileSystem() *MockFileSystem {
	return &MockFileSystem{
		Files:         make(map[string][]byte),
		Dirs:          make(map[string]bool),
		Permissions:   make(map[string]os.FileMode),
		FailOperations: make(map[string]error),
		FileInfos:     make(map[string]MockFileInfo),
	}
}

func (m *MockFileSystem) Stat(name string) (os.FileInfo, error) {
	if err, exists := m.FailOperations["stat:"+name]; exists {
		return nil, err
	}
	
	if info, exists := m.FileInfos[name]; exists {
		return info, nil
	}
	
	if _, exists := m.Files[name]; exists {
		return MockFileInfo{
			FileName:    filepath.Base(name),
			FileSize:    int64(len(m.Files[name])),
			FileMode:    0644,
			FileModTime: time.Now(),
			FileIsDir:   false,
		}, nil
	}
	
	if m.Dirs[name] {
		return MockFileInfo{
			FileName:    filepath.Base(name),
			FileSize:    0,
			FileMode:    0755,
			FileModTime: time.Now(),
			FileIsDir:   true,
		}, nil
	}
	
	return nil, os.ErrNotExist
}

func (m *MockFileSystem) Open(name string) (*os.File, error) {
	if err, exists := m.FailOperations["open:"+name]; exists {
		return nil, err
	}
	return nil, errors.New("mock file system: open not implemented for real files")
}

func (m *MockFileSystem) Create(name string) (*os.File, error) {
	if err, exists := m.FailOperations["create:"+name]; exists {
		return nil, err
	}
	return nil, errors.New("mock file system: create not implemented for real files")
}

func (m *MockFileSystem) Remove(name string) error {
	if err, exists := m.FailOperations["remove:"+name]; exists {
		return err
	}
	delete(m.Files, name)
	delete(m.Dirs, name)
	return nil
}

func (m *MockFileSystem) MkdirAll(path string, perm os.FileMode) error {
	if err, exists := m.FailOperations["mkdirall:"+path]; exists {
		return err
	}
	m.Dirs[path] = true
	return nil
}

func (m *MockFileSystem) WriteFile(filename string, data []byte, perm os.FileMode) error {
	if err, exists := m.FailOperations["writefile:"+filename]; exists {
		return err
	}
	m.Files[filename] = data
	return nil
}

func (m *MockFileSystem) ReadFile(filename string) ([]byte, error) {
	if err, exists := m.FailOperations["readfile:"+filename]; exists {
		return nil, err
	}
	if data, exists := m.Files[filename]; exists {
		return data, nil
	}
	return nil, os.ErrNotExist
}

func (m *MockFileSystem) TempDir(dir, prefix string) (string, error) {
	if err, exists := m.FailOperations["tempdir"]; exists {
		return "", err
	}
	tempPath := filepath.Join(dir, prefix+"123456")
	m.Dirs[tempPath] = true
	return tempPath, nil
}

func (m *MockFileSystem) RemoveAll(path string) error {
	if err, exists := m.FailOperations["removeall:"+path]; exists {
		return err
	}
	for file := range m.Files {
		if strings.HasPrefix(file, path) {
			delete(m.Files, file)
		}
	}
	for dir := range m.Dirs {
		if strings.HasPrefix(dir, path) {
			delete(m.Dirs, dir)
		}
	}
	return nil
}

// MockCommandExecutor provides a mock command executor for testing
type MockCommandExecutor struct {
	Responses      map[string][]byte
	Errors         map[string]error
	AvailableCommands map[string]bool
	CallLog        []string
}

func NewMockCommandExecutor() *MockCommandExecutor {
	return &MockCommandExecutor{
		Responses:      make(map[string][]byte),
		Errors:         make(map[string]error),
		AvailableCommands: make(map[string]bool),
		CallLog:        make([]string, 0),
	}
}

func (m *MockCommandExecutor) Execute(name string, args ...string) ([]byte, error) {
	cmd := fmt.Sprintf("%s %s", name, strings.Join(args, " "))
	m.CallLog = append(m.CallLog, cmd)
	
	// Check for command-specific errors first
	if err, exists := m.Errors[cmd]; exists {
		return nil, err
	}
	
	// Check for general command errors (e.g., "ffmpeg")
	if err, exists := m.Errors[name]; exists {
		return nil, err
	}
	
	if response, exists := m.Responses[cmd]; exists {
		return response, nil
	}
	
	return []byte("mock response"), nil
}

func (m *MockCommandExecutor) ExecuteWithStdin(name string, stdin io.Reader, args ...string) ([]byte, error) {
	cmd := fmt.Sprintf("%s %s", name, strings.Join(args, " "))
	m.CallLog = append(m.CallLog, cmd+" (with stdin)")
	
	if err, exists := m.Errors[cmd]; exists {
		return nil, err
	}
	
	if response, exists := m.Responses[cmd]; exists {
		return response, nil
	}
	
	return []byte("mock response with stdin"), nil
}

func (m *MockCommandExecutor) IsAvailable(command string) bool {
	if available, exists := m.AvailableCommands[command]; exists {
		return available
	}
	return true // Default to available
}

// MockUserInteraction provides a mock user interaction for testing
type MockUserInteraction struct {
	FloatResponses   map[string]float64
	StringResponses  map[string]string
	SelectResponses  map[string]string
	ConfirmResponses map[string]bool
	Errors          map[string]error
	CallLog         []string
}

func NewMockUserInteraction() *MockUserInteraction {
	return &MockUserInteraction{
		FloatResponses:   make(map[string]float64),
		StringResponses:  make(map[string]string),
		SelectResponses:  make(map[string]string),
		ConfirmResponses: make(map[string]bool),
		Errors:          make(map[string]error),
		CallLog:         make([]string, 0),
	}
}

func (m *MockUserInteraction) PromptForFloat(label string, defaultValue float64, validator func(float64) error) (float64, error) {
	m.CallLog = append(m.CallLog, fmt.Sprintf("PromptForFloat: %s (default: %f)", label, defaultValue))
	
	if err, exists := m.Errors[label]; exists {
		return 0, err
	}
	
	if response, exists := m.FloatResponses[label]; exists {
		if validator != nil {
			if err := validator(response); err != nil {
				return 0, err
			}
		}
		return response, nil
	}
	
	return defaultValue, nil
}

func (m *MockUserInteraction) PromptForString(label string, defaultValue string, validator func(string) error) (string, error) {
	m.CallLog = append(m.CallLog, fmt.Sprintf("PromptForString: %s (default: %s)", label, defaultValue))
	
	if err, exists := m.Errors[label]; exists {
		return "", err
	}
	
	if response, exists := m.StringResponses[label]; exists {
		if validator != nil {
			if err := validator(response); err != nil {
				return "", err
			}
		}
		return response, nil
	}
	
	return defaultValue, nil
}

func (m *MockUserInteraction) PromptForSelect(label string, items []string) (string, error) {
	m.CallLog = append(m.CallLog, fmt.Sprintf("PromptForSelect: %s", label))
	
	if err, exists := m.Errors[label]; exists {
		return "", err
	}
	
	if response, exists := m.SelectResponses[label]; exists {
		return response, nil
	}
	
	if len(items) > 0 {
		return items[0], nil
	}
	
	return "", errors.New("no items provided")
}

func (m *MockUserInteraction) PromptForConfirm(label string) (bool, error) {
	m.CallLog = append(m.CallLog, fmt.Sprintf("PromptForConfirm: %s", label))
	
	if err, exists := m.Errors[label]; exists {
		return false, err
	}
	
	if response, exists := m.ConfirmResponses[label]; exists {
		return response, nil
	}
	
	return false, nil
}

// MockTime provides a mock time interface for testing
type MockTime struct {
	CurrentTime time.Time
	SleepCalls  []time.Duration
}

func NewMockTime() *MockTime {
	return &MockTime{
		CurrentTime: time.Now(),
		SleepCalls:  make([]time.Duration, 0),
	}
}

func (m *MockTime) Now() time.Time {
	return m.CurrentTime
}

func (m *MockTime) Sleep(d time.Duration) {
	m.SleepCalls = append(m.SleepCalls, d)
}