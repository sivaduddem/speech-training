import type { EventType, Frame, SpeechEvent } from "../types";
import { clamp } from "../types";
import type { LinguisticCue } from "./linguistic";

const TYPES: EventType[] = ["repetition", "prolongation", "block", "restart", "pause"];
const ONSET = 0.6;
const OFFSET = 0.4;
const ONSET_FRAMES = 3;
const OFFSET_FRAMES = 3;
const HORIZON = 0.4;
const MIN_DURATION: Record<EventType, number> = {
  repetition: 0.18,
  prolongation: 0.45,
  block: 0.28,
  restart: 0.2,
  pause: 0.35,
};

interface Part {
  value: number;
  weight: number;
  available: boolean;
  evidence: string;
}

interface Active {
  id: number;
  peak: number;
}

export interface DetState {
  smooth: Record<EventType, number>;
  highRun: Record<EventType, number>;
  lowRun: Record<EventType, number>;
  active: Partial<Record<EventType, Active>>;
  events: SpeechEvent[];
  recentSpeech: number;
  sustain: number;
  idCounter: number;
  lastT: number | null;
}

export function emptyDetector(): DetState {
  return {
    smooth: blank(),
    highRun: blank(),
    lowRun: blank(),
    active: {},
    events: [],
    recentSpeech: 0,
    sustain: 0,
    idCounter: 0,
    lastT: null,
  };
}

function blank(): Record<EventType, number> {
  return { repetition: 0, prolongation: 0, block: 0, restart: 0, pause: 0 };
}

function attend(parts: Part[]): { score: number; support: number; evidence: string[] } {
  const available = parts.filter((part) => part.available && part.value > 0);
  if (available.length === 0) return { score: 0, support: 0, evidence: [] };
  const logits = available.map((part) => part.weight * (0.25 + part.value));
  const peak = Math.max(...logits);
  const weights = logits.map((logit) => Math.exp((logit - peak) / 0.7));
  const sum = weights.reduce((total, weight) => total + weight, 0);
  const score = available.reduce((total, part, index) => total + (weights[index] / sum) * part.value, 0);
  const evidence = available.filter((part) => part.value >= 0.45).map((part) => part.evidence);
  return { score: clamp(score), support: evidence.length, evidence };
}

function scoresFor(state: DetState, frame: Frame): Record<EventType, { score: number; support: number; evidence: string[] }> {
  const quiet = frame.rms < 0.02 && frame.voiced < 0.4;
  const speaking = frame.voiced > 0.5 && frame.rms > 0.02;
  state.recentSpeech = speaking ? 1 : state.recentSpeech * 0.96;
  const flat = (1 - clamp(frame.flux)) * (1 - clamp(frame.autocorr)) * (frame.voiced > 0.5 ? 1 : 0);
  const steady = flat > 0.55 && frame.rms > 0.02;
  state.sustain = steady ? state.sustain + 1 : Math.max(0, state.sustain - 2);
  const prolongAudio = state.sustain >= 10 ? flat * Math.min(1, state.sustain / 10) : 0;

  const held =
    frame.videoPresent &&
    quiet &&
    frame.lipVelocity < 0.05 &&
    frame.landmarkMotion < 0.05 &&
    (frame.lipCompression > 0.5 || frame.jawOpen > 0.4 || frame.lipAperture > 0.32);
  const relaxed =
    frame.videoPresent &&
    frame.lipCompression < 0.28 &&
    frame.jawOpen < 0.22 &&
    frame.lipAperture < 0.24 &&
    frame.landmarkMotion < 0.06;

  const repetitionAudio = frame.autocorr > 0.45 && speaking ? frame.autocorr : 0;
  const raw = {
    repetition: attend([
      { value: repetitionAudio, weight: 1.15, available: true, evidence: "A repeating rhythm in the sound" },
      {
        value: frame.videoPresent && speaking && frame.lipVelocity > 0.08 && frame.autocorr > 0.4 ? 0.45 : 0,
        weight: 0.4,
        available: frame.videoPresent,
        evidence: "Lip opening is pulsing with that rhythm",
      },
      {
        value: frame.textPresent ? frame.repetitionCue : 0,
        weight: 1.45,
        available: frame.textPresent,
        evidence: "The transcript repeats a word or phrase",
      },
    ]),
    prolongation: attend([
      { value: prolongAudio, weight: 1.2, available: true, evidence: "A sound stayed steady" },
      {
        value: frame.textPresent ? frame.prolongationCue : 0,
        weight: 0.9,
        available: frame.textPresent,
        evidence: "The words held still while the sound continued",
      },
    ]),
    block: attend([
      {
        value: quiet && state.recentSpeech > 0.35 ? 0.34 : 0,
        weight: 0.5,
        available: true,
        evidence: "Sound dropped after recent speech",
      },
      {
        value: held ? 0.9 : 0,
        weight: 1.8,
        available: frame.videoPresent,
        evidence: "The mouth held a shape while the sound dropped",
      },
    ]),
    restart: attend([
      {
        value: frame.textPresent ? frame.restartCue : 0,
        weight: 1.4,
        available: frame.textPresent,
        evidence: "The transcript looks like a revision or a restart",
      },
    ]),
    pause: attend([
      {
        value: quiet && state.recentSpeech > 0.2 ? 0.75 : 0,
        weight: 1,
        available: true,
        evidence: "A quiet stretch",
      },
      {
        value: relaxed && quiet ? 0.85 : 0,
        weight: 1.3,
        available: frame.videoPresent,
        evidence: "The mouth looks at rest during the quiet",
      },
    ]),
  };

  if (raw.block.score >= raw.pause.score) raw.pause.score *= 0.25;
  else raw.block.score *= 0.25;
  return raw;
}

export function stepDetector(state: DetState, frame: Frame): DetState {
  const next: DetState = {
    ...state,
    smooth: { ...state.smooth },
    highRun: { ...state.highRun },
    lowRun: { ...state.lowRun },
    active: { ...state.active },
    events: state.events.map((event) => ({ ...event, evidence: [...event.evidence] })),
    recentSpeech: state.recentSpeech,
    sustain: state.sustain,
  };
  const raw = scoresFor(next, frame);
  for (const type of TYPES) {
    const reading = raw[type];
    next.smooth[type] = 0.55 * next.smooth[type] + 0.45 * reading.score;
    const active = next.active[type];
    if (!active) {
      next.highRun[type] = next.smooth[type] > ONSET ? next.highRun[type] + 1 : 0;
      if (next.highRun[type] >= ONSET_FRAMES) {
        const dt = next.lastT === null ? 0.05 : Math.max(0.02, frame.t - next.lastT);
        const start = Math.max(0, frame.t - (ONSET_FRAMES - 1) * dt);
        const reusable = [...next.events].reverse().find((event) => event.type === type && frame.t - event.end < 0.15);
        if (reusable) {
          next.active[type] = { id: reusable.id, peak: next.smooth[type] };
        } else {
          next.idCounter += 1;
          next.events.push({
            id: next.idCounter,
            type,
            start,
            end: frame.t,
            confidence: confidenceFor(next.smooth[type], reading.support),
            state: "provisional",
            evidence: reading.evidence,
            timing: "frame",
          });
          next.active[type] = { id: next.idCounter, peak: next.smooth[type] };
        }
        next.lowRun[type] = 0;
      }
    } else {
      const event = next.events.find((item) => item.id === active.id);
      if (event) {
        event.end = Math.max(event.end, frame.t);
        event.confidence = Math.max(event.confidence, confidenceFor(next.smooth[type], reading.support));
        for (const note of reading.evidence) {
          if (!event.evidence.includes(note)) event.evidence.push(note);
        }
      }
      next.lowRun[type] = next.smooth[type] < OFFSET ? next.lowRun[type] + 1 : 0;
      if (next.lowRun[type] >= OFFSET_FRAMES) {
        if (event && event.end - event.start < MIN_DURATION[type]) {
          next.events = next.events.filter((item) => item.id !== event.id);
        }
        delete next.active[type];
        next.highRun[type] = 0;
      }
    }
  }
  for (const event of next.events) {
    const open = Object.values(next.active).some((item) => item?.id === event.id);
    event.state = open || event.end > frame.t - HORIZON ? "provisional" : "final";
  }
  next.lastT = frame.t;
  return next;
}

export function runDetector(frames: Frame[]): SpeechEvent[] {
  return frames.reduce(stepDetector, emptyDetector()).events;
}

export function applyDelayedCue(state: DetState, cue: LinguisticCue, now: number): DetState {
  const events = state.events.map((event) => ({ ...event, evidence: [...event.evidence] }));
  const match = events.find((event) => event.type === cue.kind && overlap(event.start, event.end, cue.start, cue.end) > 0.2);
  if (match) {
    match.confidence = Math.min(0.92, Math.max(match.confidence, cue.confidence));
    if (!match.evidence.includes(cue.detail)) match.evidence.push(cue.detail);
    return { ...state, events };
  }
  const end = Math.max(cue.end, cue.start + 0.2);
  return {
    ...state,
    idCounter: state.idCounter + 1,
    events: [
      ...events,
      {
        id: state.idCounter + 1,
        type: cue.kind,
        start: cue.start,
        end,
        confidence: Math.min(0.92, cue.confidence),
        state: now - end > HORIZON ? "final" : "provisional",
        evidence: [cue.detail],
        timing: "approximate",
      },
    ],
  };
}

function confidenceFor(score: number, support: number): number {
  return Math.min(0.92, Math.max(0.3, score * (support >= 2 ? 1 : 0.86)));
}

function overlap(startA: number, endA: number, startB: number, endB: number): number {
  const intersection = Math.max(0, Math.min(endA, endB) - Math.max(startA, startB));
  const union = Math.max(endA, endB) - Math.min(startA, startB);
  return union <= 0 ? 0 : intersection / union;
}
