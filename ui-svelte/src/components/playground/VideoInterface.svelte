<script lang="ts">
  import { models } from "../../stores/api";
  import { persistentStore } from "../../stores/persistent";
  import { generateVideo } from "../../lib/videoApi";
  import { playgroundStores } from "../../stores/playgroundActivity";
  import ModelSelector from "./ModelSelector.svelte";
  import ExpandableTextarea from "./ExpandableTextarea.svelte";
  import type { VideoGenerationResponse } from "../../lib/types";

  const selectedModelStore = persistentStore<string>("playground-video-model", "");
  const widthStore = persistentStore<number>("playground-video-width", 512);
  const heightStore = persistentStore<number>("playground-video-height", 512);
  const numFramesStore = persistentStore<number>("playground-video-num-frames", 14);
  const numStepsStore = persistentStore<number>("playground-video-num-steps", 25);
  const guidanceStore = persistentStore<number>("playground-video-guidance", 1.0);
  const fpsStore = persistentStore<number>("playground-video-fps", 6);
  const seedStore = persistentStore<number>("playground-video-seed", -1);
  const negativePromptStore = persistentStore<string>("playground-video-negative-prompt", "");
  const useAdvancedStore = persistentStore<boolean>("playground-video-advanced", false);
  const formatStore = persistentStore<"url" | "b64_json">("playground-video-format", "url");

  let prompt = $state("");
  let isGenerating = $state(false);
  let generatedVideos = $state<Array<{ url: string; b64: string | null; num_frames: number; fps: number }>>([]);
  let error = $state<string | null>(null);
  let abortController = $state<AbortController | null>(null);
  let showSettings = $state(false);

  let hasModels = $derived($models.some((m) => !m.unlisted));

  $effect(() => {
    playgroundStores.videoGenerating.set(isGenerating);
  });

  async function generate() {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt || !$selectedModelStore || isGenerating) return;

    isGenerating = true;
    error = null;
    abortController = new AbortController();

    const actualSeed = $seedStore === -1 ? undefined : $seedStore;

    try {
      const response = await generateVideo(
        $selectedModelStore,
        trimmedPrompt,
        {
          width: $widthStore,
          height: $heightStore,
          num_frames: $numFramesStore,
          num_inference_steps: $numStepsStore,
          guidance_scale: $guidanceStore,
          fps: $fpsStore,
          seed: actualSeed,
          negative_prompt: $negativePromptStore || undefined,
          response_format: $formatStore,
        },
        abortController.signal
      );

      if (response.data && response.data.length > 0) {
        generatedVideos = response.data
          .filter((v) => v.url || v.b64_json)
          .map((v) => ({
            url: v.url ?? "",
            b64: v.b64_json ? `data:video/mp4;base64,${v.b64_json}` : null,
            num_frames: v.num_frames ?? $numFramesStore,
            fps: v.fps ?? $fpsStore,
          }));
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        // User cancelled
      } else {
        error = err instanceof Error ? err.message : "An error occurred";
      }
    } finally {
      isGenerating = false;
      abortController = null;
    }
  }

  function cancelGeneration() {
    abortController?.abort();
  }

  function clearVideos() {
    generatedVideos = [];
    error = null;
    prompt = "";
  }

  function downloadVideo(index: number = 0) {
    const video = generatedVideos[index];
    if (!video) return;

    const link = document.createElement("a");
    if (video.b64) {
      link.href = video.b64;
    } else {
      link.href = video.url;
      link.target = "_blank";
    }
    link.download = `generated-video-${Date.now()}-${index}.mp4`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      generate();
    }
  }
</script>

<div class="flex flex-col h-full">
  <!-- Model selector and controls -->
  <div class="shrink-0 flex flex-wrap gap-2 mb-4">
    <ModelSelector bind:value={$selectedModelStore} placeholder="Select a video model..." disabled={isGenerating} />

    <select
      class="px-3 py-2 rounded border border-gray-200 dark:border-white/10 bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
      bind:value={$formatStore}
      disabled={isGenerating}
    >
      <option value="url">URL Format</option>
      <option value="b64_json">Base64 Format</option>
    </select>

    <button
      class="px-3 py-2 rounded border border-gray-200 dark:border-white/10 bg-surface hover:bg-secondary-hover transition-colors"
      onclick={() => (showSettings = !showSettings)}
    >
      {showSettings ? "Hide Settings" : "Settings"}
    </button>
  </div>

  <!-- Settings Panel -->
  {#if showSettings}
    <div class="shrink-0 mb-4 p-4 rounded border border-gray-200 dark:border-white/10 bg-surface">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
        <label class="flex flex-col gap-1">
          <span class="text-xs text-txtsecondary">Width</span>
          <input
            type="number"
            class="px-2 py-1 rounded border border-gray-200 dark:border-white/10 bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
            bind:value={$widthStore}
            min="48"
            max="1024"
            step="16"
            disabled={isGenerating}
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-xs text-txtsecondary">Height</span>
          <input
            type="number"
            class="px-2 py-1 rounded border border-gray-200 dark:border-white/10 bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
            bind:value={$heightStore}
            min="48"
            max="1024"
            step="16"
            disabled={isGenerating}
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-xs text-txtsecondary">Frames</span>
          <input
            type="number"
            class="px-2 py-1 rounded border border-gray-200 dark:border-white/10 bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
            bind:value={$numFramesStore}
            min="2"
            max="64"
            step="2"
            disabled={isGenerating}
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-xs text-txtsecondary">Steps</span>
          <input
            type="number"
            class="px-2 py-1 rounded border border-gray-200 dark:border-white/10 bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
            bind:value={$numStepsStore}
            min="1"
            max="100"
            disabled={isGenerating}
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-xs text-txtsecondary">CFG Scale</span>
          <input
            type="number"
            class="px-2 py-1 rounded border border-gray-200 dark:border-white/10 bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
            bind:value={$guidanceStore}
            min="0"
            max="20"
            step="0.1"
            disabled={isGenerating}
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-xs text-txtsecondary">FPS</span>
          <input
            type="number"
            class="px-2 py-1 rounded border border-gray-200 dark:border-white/10 bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
            bind:value={$fpsStore}
            min="1"
            max="120"
            disabled={isGenerating}
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-xs text-txtsecondary">Seed (-1 = random)</span>
          <input
            type="number"
            class="px-2 py-1 rounded border border-gray-200 dark:border-white/10 bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
            bind:value={$seedStore}
            min="-1"
            disabled={isGenerating}
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-xs text-txtsecondary">Negative Prompt</span>
          <input
            type="text"
            class="px-2 py-1 rounded border border-gray-200 dark:border-white/10 bg-surface focus:outline-none focus:ring-2 focus:ring-primary text-sm"
            bind:value={$negativePromptStore}
            placeholder="blur, low quality, distorted..."
            disabled={isGenerating}
          />
        </label>
      </div>
    </div>
  {/if}

  <!-- Empty state for no models configured -->
  {#if !hasModels}
    <div class="flex-1 flex items-center justify-center text-txtsecondary">
      <p>No video models configured. Add a video model to your configuration.</p>
    </div>
  {:else}
    <!-- Video display area -->
    <div class="flex-1 overflow-auto mb-4 flex items-center justify-center bg-surface border border-gray-200 dark:border-white/10 rounded">
      {#if isGenerating}
        <div class="text-center text-txtsecondary">
          <div class="inline-block w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mb-2"></div>
          <p>Generating video...</p>
          <p class="text-sm mt-1 text-txtsecondary">This may take a while</p>
        </div>
      {:else if error}
        <div class="text-center text-red-500 p-4">
          <p class="font-medium">Error</p>
          <p class="text-sm mt-1">{error}</p>
        </div>
      {:else if generatedVideos.length > 0}
        <div class="w-full max-w-full p-4">
          {#each generatedVideos as video, i}
            <div class="flex flex-col items-center gap-2 mb-4">
              <video
                controls
                class="max-w-full max-h-[60vh] rounded object-contain"
              >
                {#if video.b64}
                  <source src={video.b64} type="video/mp4" />
                {:else}
                  <source src={video.url} type="video/mp4" />
                {/if}
                Your browser does not support the video tag.
              </video>
              <div class="flex items-center gap-2 text-sm text-txtsecondary">
                <span>{video.num_frames} frames @ {video.fps} fps</span>
                <button
                  class="px-2 py-0.5 rounded border border-gray-200 dark:border-white/10 bg-surface hover:bg-secondary-hover transition-colors text-xs"
                  onclick={() => downloadVideo(i)}
                  aria-label="Download video"
                >
                  Download
                </button>
              </div>
            </div>
          {/each}
        </div>
      {:else}
        <div class="text-center text-txtsecondary p-4">
          <p>Enter a prompt below to generate a video</p>
          <p class="text-sm mt-1 text-txtsecondary/70">Video generation may take a few minutes depending on settings</p>
        </div>
      {/if}
    </div>

    <!-- Prompt input area -->
    <div class="shrink-0 flex flex-col md:flex-row gap-2">
      <ExpandableTextarea
        bind:value={prompt}
        placeholder="Describe the video you want to generate..."
        rows={3}
        onkeydown={handleKeyDown}
        disabled={isGenerating || !$selectedModelStore}
      />
      <div class="flex flex-row md:flex-col gap-2">
        {#if isGenerating}
          <button class="btn bg-red-500 hover:bg-red-600 text-white flex-1 md:flex-none" onclick={cancelGeneration}>
            Cancel
          </button>
        {:else}
          <button
            class="btn bg-primary text-btn-primary-text hover:opacity-90 flex-1 md:flex-none"
            onclick={generate}
            disabled={!prompt.trim() || !$selectedModelStore}
          >
            Generate
          </button>
          <button
            class="btn flex-1 md:flex-none"
            onclick={clearVideos}
            disabled={generatedVideos.length === 0 && !error && !prompt.trim()}
          >
            Clear
          </button>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  video::-webkit-media-controls-panel {
    display: flex !important;
    align-items: center;
  }
</style>
