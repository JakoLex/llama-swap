package proxy

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/fxamacker/cbor/v2"
	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"
	"github.com/mostlygeek/llama-swap/event"
	"github.com/mostlygeek/llama-swap/proxy/cache"
	"github.com/tidwall/gjson"
)

// zstdEncOptions are the shared zstd encoder options for maximum compression.
var zstdEncOptions = []zstd.EOption{
	zstd.WithEncoderLevel(zstd.SpeedBetterCompression),
}

// zstdDecOptions are the shared zstd decoder options.
var zstdDecOptions = []zstd.DOption{}

// zstdEncPool pools zstd.Encoder instances to reduce allocations.
var zstdEncPool = &sync.Pool{
	New: func() interface{} {
		enc, _ := zstd.NewWriter(nil, zstdEncOptions...)
		return enc
	},
}

// zstdDecPool pools zstd.Decoder instances to reduce allocations.
var zstdDecPool = &sync.Pool{
	New: func() interface{} {
		dec, _ := zstd.NewReader(nil, zstdDecOptions...)
		return dec
	},
}

// compressCapture marshals a ReqRespCapture to CBOR and compresses it with zstd.
// Returns compressed bytes and the original CBOR byte count for logging.
func compressCapture(c *ReqRespCapture) ([]byte, int, error) {
	cborBytes, err := cbor.Marshal(c)
	if err != nil {
		return nil, 0, fmt.Errorf("marshal capture: %w", err)
	}
	zenc := zstdEncPool.Get().(*zstd.Encoder)
	defer zstdEncPool.Put(zenc)
	return zenc.EncodeAll(cborBytes, nil), len(cborBytes), nil
}

// decompressCapture decompresses zstd-compressed CBOR and unmarshals it into a ReqRespCapture.
func decompressCapture(data []byte) (*ReqRespCapture, error) {
	dec := zstdDecPool.Get().(*zstd.Decoder)
	defer zstdDecPool.Put(dec)
	cborBytes, err := dec.DecodeAll(data, nil)
	if err != nil {
		return nil, fmt.Errorf("decompress capture: %w", err)
	}
	var capture ReqRespCapture
	if err := cbor.Unmarshal(cborBytes, &capture); err != nil {
		return nil, fmt.Errorf("unmarshal capture: %w", err)
	}
	return &capture, nil
}

// TokenMetrics holds token usage and performance metrics
type TokenMetrics struct {
	CachedTokens    int     `json:"cache_tokens"`
	InputTokens     int     `json:"input_tokens"`
	OutputTokens    int     `json:"output_tokens"`
	PromptPerSecond float64 `json:"prompt_per_second"`
	TokensPerSecond float64 `json:"tokens_per_second"`
}

// ActivityLogEntry represents parsed token statistics from llama-server logs
type ActivityLogEntry struct {
	ID              int          `json:"id"`
	Timestamp       time.Time    `json:"timestamp"`
	Model           string       `json:"model"`
	ReqPath         string       `json:"req_path"`
	RespContentType string       `json:"resp_content_type"`
	RespStatusCode  int          `json:"resp_status_code"`
	Tokens          TokenMetrics `json:"tokens"`
	DurationMs      int          `json:"duration_ms"`
	HasCapture      bool         `json:"has_capture"`
}

type ReqRespCapture struct {
	ID          int               `json:"id"`
	ReqPath     string            `json:"req_path"`
	ReqHeaders  map[string]string `json:"req_headers"`
	ReqBody     []byte            `json:"req_body"`
	RespHeaders map[string]string `json:"resp_headers"`
	RespBody    []byte            `json:"resp_body"`
}

// ActivityLogEvent represents a token metrics event
type ActivityLogEvent struct {
	Metrics ActivityLogEntry
}

func (e ActivityLogEvent) Type() uint32 {
	return ActivityLogEventID // defined in events.go
}

// metricsMonitor parses llama-server output for token statistics
type metricsMonitor struct {
	mu            sync.RWMutex
	metrics       []ActivityLogEntry
	maxMetrics    int
	nextID        int
	logger        *LogMonitor
	rollup        metricsRollup
	rollupByModel map[string]*metricsRollup

	// capture fields
	enableCaptures bool
	captureCache   *cache.Cache // zstd-compressed CBOR of ReqRespCapture
}

// metricsRollup holds aggregated metrics data from prior requests
type metricsRollup struct {
	RequestsTotal     uint64
	InputTokensTotal  uint64
	OutputTokensTotal uint64
	CachedTokensTotal uint64
}

// newMetricsMonitor creates a new metricsMonitor. captureBufferMB is the
// capture buffer size in megabytes; 0 disables captures.
func newMetricsMonitor(logger *LogMonitor, maxMetrics int, captureBufferMB int) *metricsMonitor {
	mm := &metricsMonitor{
		logger:         logger,
		maxMetrics:     maxMetrics,
		rollupByModel:  make(map[string]*metricsRollup),
		enableCaptures: captureBufferMB > 0,
	}
	if captureBufferMB > 0 {
		mm.captureCache = cache.New(captureBufferMB * 1024 * 1024)
	}
	return mm
}

// queueMetrics adds a new metric to the collection without emitting an event.
// Returns the assigned metric ID. Call emitMetric after capture setup.
func (mp *metricsMonitor) queueMetrics(metric ActivityLogEntry) int {
	mp.mu.Lock()
	defer mp.mu.Unlock()

	metric.ID = mp.nextID
	mp.nextID++
	mp.metrics = append(mp.metrics, metric)
	if len(mp.metrics) > mp.maxMetrics {
		mp.metrics = mp.metrics[len(mp.metrics)-mp.maxMetrics:]
	}
	mp.updateTokenRollupCounters(metric)
	return metric.ID
}

// emitMetric publishes an ActivityLogEvent for the given metric.
func (mp *metricsMonitor) emitMetric(metric ActivityLogEntry) {
	event.Emit(ActivityLogEvent{Metrics: metric})
}

// addCapture compresses and stores a capture in the cache.
// Returns true if the capture was stored, false otherwise.
func (mp *metricsMonitor) addCapture(capture ReqRespCapture) bool {
	if !mp.enableCaptures {
		return false
	}

	compressed, uncompressedBytes, err := compressCapture(&capture)
	if err != nil {
		mp.logger.Warnf("failed to compress capture: %v, skipping", err)
		return false
	}

	if err := mp.captureCache.Add(capture.ID, compressed); err != nil {
		mp.logger.Warnf("capture %d too large (%d bytes), skipping: %v", capture.ID, len(compressed), err)
		return false
	}

	compressionRatio := (1 - float64(len(compressed))/float64(uncompressedBytes)) * 100
	mp.logger.Debugf("Capture %d compressed and saved: %d bytes -> %d bytes (%.1f%% compression)", capture.ID, uncompressedBytes, len(compressed), compressionRatio)
	return true
}

// getCompressedBytes returns the raw compressed bytes for a capture by ID.
func (mp *metricsMonitor) getCompressedBytes(id int) ([]byte, bool) {
	if mp.captureCache == nil {
		return nil, false
	}
	data, err := mp.captureCache.Get(id)
	if err != nil {
		return nil, false
	}
	return data, true
}

// getCaptureByID decompresses and unmarshals a capture by ID.
// Returns nil if the capture is not found or decompression fails.
func (mp *metricsMonitor) getCaptureByID(id int) *ReqRespCapture {
	if mp.captureCache == nil {
		return nil
	}
	data, exists := mp.getCompressedBytes(id)
	if !exists {
		return nil
	}

	capture, err := decompressCapture(data)
	if err != nil {
		mp.logger.Warnf("failed to decompress capture %d: %v", id, err)
		return nil
	}

	return capture
}

// getMetrics returns a copy of the current metrics
func (mp *metricsMonitor) getMetrics() []ActivityLogEntry {
	mp.mu.RLock()
	defer mp.mu.RUnlock()

	result := make([]ActivityLogEntry, len(mp.metrics))
	copy(result, mp.metrics)
	return result
}

// getMetricsJSON returns metrics as JSON
func (mp *metricsMonitor) getMetricsJSON() ([]byte, error) {
	mp.mu.RLock()
	defer mp.mu.RUnlock()

	if mp.captureCache == nil {
		return json.Marshal(mp.metrics)
	}

	// Make a copy with up-to-date has_capture from cache
	result := make([]ActivityLogEntry, len(mp.metrics))
	for i, m := range mp.metrics {
		m.HasCapture = mp.captureCache.Has(m.ID)
		result[i] = m
	}
	return json.Marshal(result)
}

// Capture field flags for controlling what is saved in ReqRespCapture.
type captureFields uint

const (
	captureNone captureFields = 1 << iota
	captureReqHeaders
	captureReqBody
	captureRespHeaders
	captureRespBody
)

const (
	captureReqAll  = captureReqHeaders | captureReqBody
	captureRespAll = captureRespHeaders | captureRespBody
	captureAll     = captureReqAll | captureRespAll
)

// update token counters at model level and global level.
// Must be called with mp.mu write lock held.
func (mp *metricsMonitor) updateTokenRollupCounters(metric ActivityLogEntry) {
	updateRollup(&mp.rollup, metric)

	modelRollup, ok := mp.rollupByModel[metric.Model]
	if !ok {
		modelRollup = &metricsRollup{}
		mp.rollupByModel[metric.Model] = modelRollup
	}
	updateRollup(modelRollup, metric)
}

// updateRollup increases counters based on the given metric object
func updateRollup(rollup *metricsRollup, metric ActivityLogEntry) {
	rollup.RequestsTotal++
	if metric.Tokens.InputTokens >= 0 {
		rollup.InputTokensTotal += uint64(metric.Tokens.InputTokens)
	}
	if metric.Tokens.OutputTokens >= 0 {
		rollup.OutputTokensTotal += uint64(metric.Tokens.OutputTokens)
	}
	if metric.Tokens.CachedTokens >= 0 {
		rollup.CachedTokensTotal += uint64(metric.Tokens.CachedTokens)
	}
}

// getPrometheusText returns the metrics for the current monitor in Prometheus text format
func (mp *metricsMonitor) getPrometheusText() []byte {
	mp.mu.RLock()
	overall := mp.rollup
	perModel := make(map[string]metricsRollup, len(mp.rollupByModel))
	for model, rollup := range mp.rollupByModel {
		if rollup != nil {
			perModel[model] = *rollup
		}
	}
	metricsCopy := make([]ActivityLogEntry, len(mp.metrics))
	copy(metricsCopy, mp.metrics)
	mp.mu.RUnlock()

	models := make([]string, 0, len(perModel))
	for model := range perModel {
		models = append(models, model)
	}
	sort.Strings(models)

	var b strings.Builder
	writeCounterWithModel(&b, "llama_swap_requests_total", "Total number of requests with recorded metrics.", overall.RequestsTotal, models, perModel, func(r metricsRollup) uint64 {
		return r.RequestsTotal
	})
	writeCounterWithModel(&b, "llama_swap_input_tokens_total", "Total input tokens recorded.", overall.InputTokensTotal, models, perModel, func(r metricsRollup) uint64 {
		return r.InputTokensTotal
	})
	writeCounterWithModel(&b, "llama_swap_output_tokens_total", "Total output tokens recorded.", overall.OutputTokensTotal, models, perModel, func(r metricsRollup) uint64 {
		return r.OutputTokensTotal
	})
	writeCounterWithModel(&b, "llama_swap_cached_tokens_total", "Total cached tokens recorded.", overall.CachedTokensTotal, models, perModel, func(r metricsRollup) uint64 {
		return r.CachedTokensTotal
	})

	windowSizes := []int{1, 5, 15}
	overallGen, overallPrompt, perModelGen, perModelPrompt := computeTokensPerSecondLastN(metricsCopy, windowSizes)
	for _, windowSize := range windowSizes {
		writeGaugeWithModel(&b, fmt.Sprintf("llama_swap_generate_tokens_per_second_last_%d", windowSize), fmt.Sprintf("Average generation tokens per second over last %d requests.", windowSize), overallGen[windowSize], models, func(model string) float64 {
			if modelValues, ok := perModelGen[model]; ok {
				return modelValues[windowSize]
			}
			return 0
		})
		writeGaugeWithModel(&b, fmt.Sprintf("llama_swap_prompt_tokens_per_second_last_%d", windowSize), fmt.Sprintf("Average prompt tokens per second over last %d requests.", windowSize), overallPrompt[windowSize], models, func(model string) float64 {
			if modelValues, ok := perModelPrompt[model]; ok {
				return modelValues[windowSize]
			}
			return 0
		})
	}

	return []byte(b.String())
}

// writeCounterWithModel writes a Prometheus counter metric with per-model breakdown
func writeCounterWithModel(
	b *strings.Builder,
	name string,
	help string,
	overall uint64,
	models []string,
	perModel map[string]metricsRollup,
	getValue func(metricsRollup) uint64,
) {
	fmt.Fprintf(b, "# HELP %s %s\n", name, help)
	fmt.Fprintf(b, "# TYPE %s counter\n", name)
	fmt.Fprintf(b, "%s %d\n", name, overall)
	for _, model := range models {
		value := getValue(perModel[model])
		fmt.Fprintf(b, "%s{model=\"%s\"} %d\n", name, promLabelValue(model), value)
	}
}

// writeGaugeWithModel writes a Prometheus gauge metric with per-model breakdown
func writeGaugeWithModel(
	b *strings.Builder,
	name string,
	help string,
	overall float64,
	models []string,
	getValue func(model string) float64,
) {
	fmt.Fprintf(b, "# HELP %s %s\n", name, help)
	fmt.Fprintf(b, "# TYPE %s gauge\n", name)
	fmt.Fprintf(b, "%s %s\n", name, formatFloat(overall))
	for _, model := range models {
		fmt.Fprintf(b, "%s{model=\"%s\"} %s\n", name, promLabelValue(model), formatFloat(getValue(model)))
	}
}

// computeTokensPerSecondLastN looks at a window size of the last N metrics and calculates the average tokens per second
func computeTokensPerSecondLastN(metrics []ActivityLogEntry, windowSizes []int) (map[int]float64, map[int]float64, map[string]map[int]float64, map[string]map[int]float64) {
	overallGenSum := make(map[int]float64)
	overallGenCount := make(map[int]int)
	overallPromptSum := make(map[int]float64)
	overallPromptCount := make(map[int]int)
	overallSeen := make(map[int]int)

	perModelSeen := make(map[string]map[int]int)
	perModelGenSum := make(map[string]map[int]float64)
	perModelGenCount := make(map[string]map[int]int)
	perModelPromptSum := make(map[string]map[int]float64)
	perModelPromptCount := make(map[string]map[int]int)

	// iterate over metrics in reverse order to get the most recent first
	for i := len(metrics) - 1; i >= 0; i-- {
		metric := metrics[i]
		model := metric.Model

		if _, ok := perModelSeen[model]; !ok {
			perModelSeen[model] = make(map[int]int)
			perModelGenSum[model] = make(map[int]float64)
			perModelGenCount[model] = make(map[int]int)
			perModelPromptSum[model] = make(map[int]float64)
			perModelPromptCount[model] = make(map[int]int)
		}

		// calculate for each window size at the same time
		for _, windowSize := range windowSizes {
			if overallSeen[windowSize] < windowSize {
				overallSeen[windowSize]++
				if metric.Tokens.TokensPerSecond >= 0 {
					overallGenSum[windowSize] += metric.Tokens.TokensPerSecond
					overallGenCount[windowSize]++
				}
				if metric.Tokens.PromptPerSecond >= 0 {
					overallPromptSum[windowSize] += metric.Tokens.PromptPerSecond
					overallPromptCount[windowSize]++
				}
			}

			if perModelSeen[model][windowSize] < windowSize {
				perModelSeen[model][windowSize]++
				if metric.Tokens.TokensPerSecond >= 0 {
					perModelGenSum[model][windowSize] += metric.Tokens.TokensPerSecond
					perModelGenCount[model][windowSize]++
				}
				if metric.Tokens.PromptPerSecond >= 0 {
					perModelPromptSum[model][windowSize] += metric.Tokens.PromptPerSecond
					perModelPromptCount[model][windowSize]++
				}
			}
		}
	}

	// calculate averages for each window size
	overallGen := make(map[int]float64)
	overallPrompt := make(map[int]float64)
	perModelGen := make(map[string]map[int]float64)
	perModelPrompt := make(map[string]map[int]float64)

	for _, windowSize := range windowSizes {
		if overallGenCount[windowSize] > 0 {
			overallGen[windowSize] = overallGenSum[windowSize] / float64(overallGenCount[windowSize])
		} else {
			overallGen[windowSize] = 0
		}
		if overallPromptCount[windowSize] > 0 {
			overallPrompt[windowSize] = overallPromptSum[windowSize] / float64(overallPromptCount[windowSize])
		} else {
			overallPrompt[windowSize] = 0
		}
	}

	for model := range perModelSeen {
		perModelGen[model] = make(map[int]float64)
		perModelPrompt[model] = make(map[int]float64)
		for _, windowSize := range windowSizes {
			if perModelGenCount[model][windowSize] > 0 {
				perModelGen[model][windowSize] = perModelGenSum[model][windowSize] / float64(perModelGenCount[model][windowSize])
			} else {
				perModelGen[model][windowSize] = 0
			}
			if perModelPromptCount[model][windowSize] > 0 {
				perModelPrompt[model][windowSize] = perModelPromptSum[model][windowSize] / float64(perModelPromptCount[model][windowSize])
			} else {
				perModelPrompt[model][windowSize] = 0
			}
		}
	}

	return overallGen, overallPrompt, perModelGen, perModelPrompt
}

func formatFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}

func promLabelValue(value string) string {
	replacer := strings.NewReplacer("\\", "\\\\", "\n", "\\n", "\"", "\\\"")
	return replacer.Replace(value)
}

// wrapHandler wraps the proxy handler to extract token metrics.
// captureFields controls what is saved in the ReqRespCapture using bitwise flags.
// if wrapHandler returns an error it is safe to assume that no
// data was sent to the client
func (mp *metricsMonitor) wrapHandler(
	modelID string,
	writer gin.ResponseWriter,
	request *http.Request,
	captureFields captureFields,
	next func(modelID string, w http.ResponseWriter, r *http.Request) error,
) error {
	// Capture request body and headers if captures enabled
	var reqBody []byte
	var reqHeaders map[string]string
	if mp.enableCaptures && (captureFields&captureReqBody) != 0 {
		if request.Body != nil {
			var err error
			reqBody, err = io.ReadAll(request.Body)
			if err != nil {
				return fmt.Errorf("failed to read request body for capture: %w", err)
			}
			request.Body.Close()
			request.Body = io.NopCloser(bytes.NewBuffer(reqBody))
		}
	}
	if mp.enableCaptures && (captureFields&captureReqHeaders) != 0 {
		reqHeaders = make(map[string]string)
		for key, values := range request.Header {
			if len(values) > 0 {
				reqHeaders[key] = values[0]
			}
		}
		redactHeaders(reqHeaders)
	}

	recorder := newBodyCopier(writer)

	// Filter Accept-Encoding to only include encodings we can decompress for metrics
	if ae := request.Header.Get("Accept-Encoding"); ae != "" {
		request.Header.Set("Accept-Encoding", filterAcceptEncoding(ae))
	}

	if err := next(modelID, recorder, request); err != nil {
		return err
	}

	// after this point we have to assume that data was sent to the client
	// and we can only log errors but not send them to clients

	// Initialize default metrics - recorded for every request
	tm := ActivityLogEntry{
		Timestamp:       time.Now(),
		Model:           modelID,
		ReqPath:         request.URL.Path,
		RespContentType: recorder.Header().Get("Content-Type"),
		RespStatusCode:  recorder.Status(),
		DurationMs:      int(time.Since(recorder.StartTime()).Milliseconds()),
	}

	if recorder.Status() != http.StatusOK {
		mp.logger.Warnf("non-200 response, recording partial metrics: status=%d, path=%s", recorder.Status(), request.URL.Path)
		tm.ID = mp.queueMetrics(tm)
		mp.emitMetric(tm)
		return nil
	}

	body := recorder.body.Bytes()
	if len(body) == 0 {
		mp.logger.Warn("metrics: empty body, recording minimal metrics")
		tm.ID = mp.queueMetrics(tm)
		mp.emitMetric(tm)
		return nil
	}

	// Decompress if needed
	if encoding := recorder.Header().Get("Content-Encoding"); encoding != "" {
		var err error
		body, err = decompressBody(body, encoding)
		if err != nil {
			mp.logger.Warnf("metrics: decompression failed: %v, path=%s, recording minimal metrics", err, request.URL.Path)
			tm.ID = mp.queueMetrics(tm)
			mp.emitMetric(tm)
			return nil
		}
	}
	if strings.Contains(recorder.Header().Get("Content-Type"), "text/event-stream") {
		if parsed, err := processStreamingResponse(modelID, recorder.StartTime(), body); err != nil {
			mp.logger.Warnf("error processing streaming response: %v, path=%s, recording minimal metrics", err, request.URL.Path)
		} else {
			tm.Tokens = parsed.Tokens
			tm.DurationMs = parsed.DurationMs
		}
	} else {
		if gjson.ValidBytes(body) {
			parsed := gjson.ParseBytes(body)
			usage := parsed.Get("usage")
			timings := parsed.Get("timings")

			// extract timings for infill - response is an array, timings are in the last element
			// see #463
			if strings.HasPrefix(request.URL.Path, "/infill") {
				if arr := parsed.Array(); len(arr) > 0 {
					timings = arr[len(arr)-1].Get("timings")
				}
			}

			if usage.Exists() || timings.Exists() {
				if parsedMetrics, err := parseMetrics(modelID, recorder.StartTime(), usage, timings); err != nil {
					mp.logger.Warnf("error parsing metrics: %v, path=%s, recording minimal metrics", err, request.URL.Path)
				} else {
					tm.Tokens = parsedMetrics.Tokens
					tm.DurationMs = parsedMetrics.DurationMs
				}
			}
		} else {
			mp.logger.Warnf("metrics: invalid JSON in response body path=%s, recording minimal metrics", request.URL.Path)
		}
	}

	// Build capture if enabled and determine if it will be stored
	var capture *ReqRespCapture
	if mp.enableCaptures {
		var respHeaders map[string]string
		var respBody []byte
		if (captureFields & captureRespHeaders) != 0 {
			respHeaders = make(map[string]string)
			for key, values := range recorder.Header() {
				if len(values) > 0 {
					respHeaders[key] = values[0]
				}
			}
			redactHeaders(respHeaders)
			delete(respHeaders, "Content-Encoding")
		}
		if (captureFields & captureRespBody) != 0 {
			respBody = body
		}
		capture = &ReqRespCapture{
			ReqPath:     request.URL.Path,
			ReqHeaders:  reqHeaders,
			ReqBody:     reqBody,
			RespHeaders: respHeaders,
			RespBody:    respBody,
		}
	}

	metricID := mp.queueMetrics(tm)
	tm.ID = metricID

	// Store capture if enabled
	if capture != nil {
		capture.ID = metricID
		if mp.addCapture(*capture) {
			tm.HasCapture = true
			mp.mu.Lock()
			mp.metrics[len(mp.metrics)-1].HasCapture = true
			mp.mu.Unlock()
		}
	}

	mp.emitMetric(tm)

	return nil
}

func processStreamingResponse(modelID string, start time.Time, body []byte) (ActivityLogEntry, error) {
	// Iterate **backwards** through the body looking for the data payload with
	// usage data. This avoids allocating a slice of all lines via bytes.Split.

	// Start from the end of the body and scan backwards for newlines
	pos := len(body)
	for pos > 0 {
		// Find the previous newline (or start of body)
		lineStart := bytes.LastIndexByte(body[:pos], '\n')
		if lineStart == -1 {
			lineStart = 0
		} else {
			lineStart++ // Move past the newline
		}

		line := bytes.TrimSpace(body[lineStart:pos])
		pos = lineStart - 1 // Move position before the newline for next iteration

		if len(line) == 0 {
			continue
		}

		// SSE payload always follows "data:"
		prefix := []byte("data:")
		if !bytes.HasPrefix(line, prefix) {
			continue
		}
		data := bytes.TrimSpace(line[len(prefix):])

		if len(data) == 0 {
			continue
		}

		if bytes.Equal(data, []byte("[DONE]")) {
			// [DONE] line itself contains nothing of interest.
			continue
		}

		if gjson.ValidBytes(data) {
			parsed := gjson.ParseBytes(data)
			usage := parsed.Get("usage")
			timings := parsed.Get("timings")

			// v1/responses format nests usage under response.usage
			if !usage.Exists() {
				usage = parsed.Get("response.usage")
			}

			if usage.Exists() || timings.Exists() {
				return parseMetrics(modelID, start, usage, timings)
			}
		}
	}

	return ActivityLogEntry{}, fmt.Errorf("no valid JSON data found in stream")
}

func parseMetrics(modelID string, start time.Time, usage, timings gjson.Result) (ActivityLogEntry, error) {
	wallDurationMs := int(time.Since(start).Milliseconds())

	// default values
	cachedTokens := -1 // unknown or missing data
	outputTokens := 0
	inputTokens := 0

	// timings data
	tokensPerSecond := -1.0
	promptPerSecond := -1.0
	durationMs := wallDurationMs

	if usage.Exists() {
		if pt := usage.Get("prompt_tokens"); pt.Exists() {
			// v1/chat/completions
			inputTokens = int(pt.Int())
		} else if it := usage.Get("input_tokens"); it.Exists() {
			// v1/messages
			inputTokens = int(it.Int())
		}

		if ct := usage.Get("completion_tokens"); ct.Exists() {
			// v1/chat/completions
			outputTokens = int(ct.Int())
		} else if ot := usage.Get("output_tokens"); ot.Exists() {
			outputTokens = int(ot.Int())
		}

		if ct := usage.Get("cache_read_input_tokens"); ct.Exists() {
			cachedTokens = int(ct.Int())
		}
	}

	// use llama-server's timing data for tok/sec and duration as it is more accurate
	if timings.Exists() {
		inputTokens = int(timings.Get("prompt_n").Int())
		outputTokens = int(timings.Get("predicted_n").Int())
		promptPerSecond = timings.Get("prompt_per_second").Float()
		tokensPerSecond = timings.Get("predicted_per_second").Float()
		timingsDurationMs := int(timings.Get("prompt_ms").Float() + timings.Get("predicted_ms").Float())
		if timingsDurationMs > durationMs {
			durationMs = timingsDurationMs
		}

		if cachedValue := timings.Get("cache_n"); cachedValue.Exists() {
			cachedTokens = int(cachedValue.Int())
		}
	}

	return ActivityLogEntry{
		Timestamp: time.Now(),
		Model:     modelID,
		Tokens: TokenMetrics{
			CachedTokens:    cachedTokens,
			InputTokens:     inputTokens,
			OutputTokens:    outputTokens,
			PromptPerSecond: promptPerSecond,
			TokensPerSecond: tokensPerSecond,
		},
		DurationMs: durationMs,
	}, nil
}

// decompressBody decompresses the body based on Content-Encoding header
func decompressBody(body []byte, encoding string) ([]byte, error) {
	switch strings.ToLower(strings.TrimSpace(encoding)) {
	case "gzip":
		reader, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		defer reader.Close()
		return io.ReadAll(reader)
	case "deflate":
		reader := flate.NewReader(bytes.NewReader(body))
		defer reader.Close()
		return io.ReadAll(reader)
	default:
		return body, nil // Return as-is for unknown/no encoding
	}
}

// responseBodyCopier records the response body and writes to the original response writer
// while also capturing it in a buffer for later processing
type responseBodyCopier struct {
	gin.ResponseWriter
	body  *bytes.Buffer
	tee   io.Writer
	start time.Time
}

func newBodyCopier(w gin.ResponseWriter) *responseBodyCopier {
	bodyBuffer := &bytes.Buffer{}
	return &responseBodyCopier{
		ResponseWriter: w,
		body:           bodyBuffer,
		tee:            io.MultiWriter(w, bodyBuffer),
		start:          time.Now(),
	}
}

func (w *responseBodyCopier) Write(b []byte) (int, error) {
	return w.tee.Write(b)
}

func (w *responseBodyCopier) WriteHeader(statusCode int) {
	w.ResponseWriter.WriteHeader(statusCode)
}

func (w *responseBodyCopier) Header() http.Header {
	return w.ResponseWriter.Header()
}

func (w *responseBodyCopier) StartTime() time.Time {
	return w.start
}

// sensitiveHeaders lists headers that should be redacted in captures
var sensitiveHeaders = map[string]bool{
	"authorization":       true,
	"proxy-authorization": true,
	"cookie":              true,
	"set-cookie":          true,
	"x-api-key":           true,
}

// redactHeaders replaces sensitive header values in-place with "[REDACTED]"
func redactHeaders(headers map[string]string) {
	for key := range headers {
		if sensitiveHeaders[strings.ToLower(key)] {
			headers[key] = "[REDACTED]"
		}
	}
}

// filterAcceptEncoding filters the Accept-Encoding header to only include
// encodings we can decompress (gzip, deflate). This respects the client's
// preferences while ensuring we can parse response bodies for metrics.
func filterAcceptEncoding(acceptEncoding string) string {
	if acceptEncoding == "" {
		return ""
	}

	supported := map[string]bool{"gzip": true, "deflate": true}
	var filtered []string

	for part := range strings.SplitSeq(acceptEncoding, ",") {
		// Parse encoding and optional quality value (e.g., "gzip;q=1.0")
		encoding, _, _ := strings.Cut(strings.TrimSpace(part), ";")
		if supported[strings.ToLower(encoding)] {
			filtered = append(filtered, strings.TrimSpace(part))
		}
	}

	return strings.Join(filtered, ", ")
}
