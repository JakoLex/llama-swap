import type { VideoGenerationRequest, VideoGenerationResponse } from "./types";

export async function generateVideo(
  model: string,
  prompt: string,
  options?: Partial<Omit<VideoGenerationRequest, "model" | "prompt" | "n">>,
  signal?: AbortSignal
): Promise<VideoGenerationResponse> {
  const request: VideoGenerationRequest = {
    model,
    prompt,
    n: 1,
    response_format: options?.response_format || "url",
    ...options,
  };

  const response = await fetch("/v1/videos/generations", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Video API error: ${response.status} - ${errorText}`);
  }

  return response.json();
}
