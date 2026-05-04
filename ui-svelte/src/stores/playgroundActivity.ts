import { writable, derived } from "svelte/store";

const chatStreaming = writable(false);
const imageGenerating = writable(false);
const videoGenerating = writable(false);
const speechGenerating = writable(false);
const audioTranscribing = writable(false);
const rerankLoading = writable(false);

export const playgroundActivity = derived(
  [chatStreaming, imageGenerating, videoGenerating, speechGenerating, audioTranscribing, rerankLoading],
  ([$chat, $image, $video, $speech, $audio, $rerank]) => $chat || $image || $video || $speech || $audio || $rerank
);

export const playgroundStores = {
  chatStreaming,
  imageGenerating,
  videoGenerating,
  speechGenerating,
  audioTranscribing,
  rerankLoading,
};
