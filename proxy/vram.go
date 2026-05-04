package proxy

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// CheckVRAM checks if the available VRAM on all GPUs is sufficient for the model.
// Returns nil if VRAM is sufficient or if GPU checking is not available/needed.
func CheckVRAM(requiredGB float64) error {
	if requiredGB <= 0 {
		return nil
	}

	freeGB, err := getFreeVramGiB()
	if err != nil {
		// If we can't query VRAM, skip the check to avoid blocking requests
		// This allows llama-swap to run on systems without nvidia GPUs
		return nil
	}

	if freeGB < requiredGB {
		return fmt.Errorf("insufficient VRAM: %.1f GiB free, but model requires %.1f GiB", freeGB, requiredGB)
	}

	return nil
}

// getFreeVramGiB queries nvidia-smi for total memory across all available GPUs
// and returns the amount of free memory in GiB (base 2).
func getFreeVramGiB() (float64, error) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader")
	output, err := cmd.Output()
	if err != nil {
		return 0, fmt.Errorf("failed to run nvidia-smi: %w", err)
	}

	var totalFreeMiB float64
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Parse output like "12345 MiB"
		parts := strings.Split(line, " ")
		if len(parts) < 2 {
			continue
		}

		miB, err := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
		if err != nil {
			continue
		}

		totalFreeMiB += miB
	}

	if totalFreeMiB == 0 {
		return 0, fmt.Errorf("no GPUs found or nvidia-smi returned zero memory")
	}

	// Convert MiB to GiB (base 2: 1024 MiB = 1 GiB)
	freeGiB := totalFreeMiB / 1024.0
	return freeGiB, nil
}
