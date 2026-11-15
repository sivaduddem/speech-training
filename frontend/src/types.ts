export type EventType = "repetition" | "prolongation" | "block" | "restart" | "pause";
export type EventState = "provisional" | "final";
export type TimingQuality = "frame" | "approximate";
export type FeedbackMode = "quiet" | "gentle" | "reminders";

export interface SpeechEvent {
  id: number;
  type: EventType;
  start: number;
  end: number;
  confidence: number;
  state: EventState;
  evidence: string[];
  timing: TimingQuality;
}

export interface Frame {
  t: number;
  rms: number;
  voiced: number;
  flux: number;
  autocorr: number;
  videoPresent: boolean;
  lipAperture: number;
  jawOpen: number;
  lipCompression: number;
  lipVelocity: number;
  landmarkMotion: number;
  textPresent: boolean;
  repetitionCue: number;
  restartCue: number;
  prolongationCue: number;
}

export interface FeatureSample {
  t: number;
  rms: number;
  lip: number;
  jaw: number;
  compression: number;
  silentMove: number;
}

export const EVENT_TYPES: EventType[] = ["repetition", "prolongation", "block", "restart", "pause"];

export const GOALS = [
  { id: "company", label: "Company while I talk" },
  { id: "review", label: "Review moments afterward" },
  { id: "pace", label: "Practice at my own pace" },
  { id: "notice-blocks", label: "Look back at possible blocks" },
  { id: "notice-repetitions", label: "Look back at repetitions" },
] as const;

export function clamp(value: number, min = 0, max = 1): number {
  return Math.min(max, Math.max(min, value));
}

export function formatClock(seconds: number): string {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const remain = Math.floor(safe % 60);
  return `${minutes}:${remain.toString().padStart(2, "0")}`;
}
