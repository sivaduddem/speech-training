/** Channel order shared with backend/src/companion/config.py live exports. */

export const LIVE_AUDIO_CHANNELS = [
  "log_rms",
  "zcr",
  "centroid",
  "flux",
  "low_band",
  "autocorr",
  "voiced",
  "delta_rms",
] as const;

export const LIVE_VISUAL_CHANNELS = [
  "lip_aperture",
  "jaw_open",
  "lip_compression_shape",
  "mouth_width",
  "lip_velocity",
  "landmark_motion",
  "pose_speed",
  "video_present",
] as const;

export const LIVE_TEXT_CHANNELS = ["repetition_cue", "restart_cue", "prolongation_cue", "text_present"] as const;

export interface ResearchFrame {
  t: number;
  audio: number[];
  visual: number[];
  text: number[];
}

export function researchDocument(frames: ResearchFrame[], events: unknown[]) {
  return {
    frame_sec: 0.05,
    audio_channels: LIVE_AUDIO_CHANNELS,
    visual_channels: LIVE_VISUAL_CHANNELS,
    text_channels: LIVE_TEXT_CHANNELS,
    frames,
    events,
    note: "Personal feature log. A model trained on synthetic channels cannot score this file.",
  };
}
