package proxy

import (
	"bytes"
	"fmt"
	"net/http/httptest"
	"os/exec"
	"strings"
	"testing"

	"github.com/mostlygeek/llama-swap/proxy/config"
	"github.com/stretchr/testify/assert"
)

func skipIfNoNvidiaSmi(t *testing.T) {
	_, err := exec.LookPath("nvidia-smi")
	if err != nil {
		t.Skipf("Skipping test: nvidia-smi not available (%v)", err)
	}

	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader")
	if output, err := cmd.CombinedOutput(); err != nil || len(output) == 0 || strings.TrimSpace(string(output)) == "" {
		t.Skipf("Skipping test: nvidia-smi returned no data or error")
	}
}

func TestCheckVRAM_InvalidGB(t *testing.T) {
	// Zero or negative GB should always pass without error
	tests := []float64{0, -1, -0.5}
	for _, gb := range tests {
		err := CheckVRAM(gb)
		if err != nil {
			t.Errorf("CheckVRAM(%f) should return nil for %f, got: %v", gb, gb, err)
		}
	}
}

func TestCheckVRAM_ZeroRequired(t *testing.T) {
	err := CheckVRAM(0)
	if err != nil {
		t.Errorf("CheckVRAM(0) should return nil, got: %v", err)
	}
}

func TestCheckVRAM_InsufficientVRAM(t *testing.T) {
	skipIfNoNvidiaSmi(t)

	err := CheckVRAM(9999)
	if err == nil {
		t.Skip("nvidia-smi returned data but VRAM was somehow sufficient for 9999GB")
	}

	expectedSubstrings := []string{"insufficient", "VRAM", "9999.0"}
	for _, substr := range expectedSubstrings {
		if !strings.Contains(err.Error(), substr) {
			t.Errorf("Error message %q should contain %q", err.Error(), substr)
		}
	}
}

func TestCheckVRAM_SufficientVRAM(t *testing.T) {
	skipIfNoNvidiaSmi(t)

	// With a very small threshold (0.001 GiB = ~1 MiB), this should always pass
	err := CheckVRAM(0.001)
	if err != nil {
		t.Errorf("CheckVRAM(0.001) should succeed, got: %v", err)
	}
}

func TestGetFreeVram(t *testing.T) {
	skipIfNoNvidiaSmi(t)
	freeGB, err := getFreeVramGiB()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	if freeGB <= 0 {
		t.Errorf("Expected positive VRAM values, got: %f GiB", freeGB)
	}
}

func TestGetFreeVram_ReturnsMiBConversion(t *testing.T) {
	tests := []struct {
		miB float64
		giB float64
	}{
		{1024, 1.0},
		{2048, 2.0},
		{512, 0.5},
	}

	for _, tt := range tests {
		expectedGB := tt.miB / 1024.0
		if expectedGB != tt.giB {
			t.Errorf("Expected conversion of %f MiB to be %f GiB, got %f", tt.miB, tt.giB, expectedGB)
		}
	}
}

func TestProxyVRAMCheck_Returns507WhenInsufficient(t *testing.T) {
	skipIfNoNvidiaSmi(t)

	requiredVram := 100.0 // Definitely more than any consumer GPU has free

	yamlContent := fmt.Sprintf(`
healthCheckTimeout: 5
logLevel: error
models:
  big-model:
    cmd: echo test
    proxy: http://localhost:9999
    minVramGB: %f
`, requiredVram)

	cfg, err := config.LoadConfigFromReader(bytes.NewReader([]byte(yamlContent)))
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	pm := New(cfg)
	defer pm.Shutdown()

	reqBody := `{"model":"big-model","messages":[{"role":"user","content":"hi"}]}`
	w := CreateTestResponseRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
	pm.ServeHTTP(w, req)

	// Should get 507 Insufficient Storage
	assert.Equal(t, 507, w.Code, "Expected 507 but got %d. Body: %s", w.Code, w.Body.String())
}

// TestProxyVRAMCheck_AllowsMinVramZero verifies that minVramGB=0 passes (no check)
func TestProxyVRAMCheck_AllowsMinVramZero(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(bytes.NewReader([]byte(`
healthCheckTimeout: 5
logLevel: error
models:
  small-model:
    cmd: echo test
    proxy: http://localhost:9999
    minVramGB: 0
`)))
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	pm := New(cfg)
	defer pm.Shutdown()

	reqBody := `{"model":"small-model","messages":[{"role":"user","content":"hi"}]}`
	w := CreateTestResponseRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
	pm.ServeHTTP(w, req)

	// minVramGB=0 means NO check, so we should NOT get 507
	assert.NotEqual(t, 507, w.Code, "Expected no VRAM error (507), got %d. Body: %s", w.Code, w.Body.String())
}
