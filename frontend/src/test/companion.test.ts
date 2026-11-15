import { describe, expect, it } from "vitest";
import { envelopeAutocorr, magnitudeSpectrum, rms, spectralCentroid, zeroCrossingRate } from "../lib/audioFeatures";
import { applyDelayedCue, emptyDetector, runDetector, stepDetector } from "../lib/detector";
import { describeCoupling, lipAperture, lipCompressionShape, measureLipMotion, type Pt } from "../lib/faceFeatures";
import { containsBannedCoaching, insightLines, liveMessage, sessionSummary } from "../lib/feedback";
import { findLinguisticEvents } from "../lib/linguistic";
import type { Frame, SpeechEvent } from "../types";

function frame(partial: Partial<Frame> & Pick<Frame, "t">): Frame {
  return {
    rms: 0.001,
    voiced: 0,
    flux: 0.2,
    autocorr: 0,
    videoPresent: true,
    lipAperture: 0.12,
    jawOpen: 0.08,
    lipCompression: 0.08,
    lipVelocity: 0.01,
    landmarkMotion: 0.01,
    textPresent: false,
    repetitionCue: 0,
    restartCue: 0,
    prolongationCue: 0,
    ...partial,
  };
}

function speech(t: number): Frame {
  return frame({
    t,
    rms: 0.08,
    voiced: 1,
    flux: 0.45,
    autocorr: 0.1,
    lipAperture: 0.42,
    jawOpen: 0.4,
    lipVelocity: 0.12,
    landmarkMotion: 0.08,
  });
}

function sequence(count: number, make: (index: number) => Frame): Frame[] {
  return Array.from({ length: count }, (_, index) => make(index * 0.05));
}

describe("detector", () => {
  it("marks a repeating rhythm and then finalizes it", () => {
    const frames = [
      ...sequence(6, (t) => speech(t)),
      ...sequence(16, (t) =>
        frame({
          t: 0.3 + t,
          rms: 0.07,
          voiced: 1,
          flux: 0.3,
          autocorr: 0.82,
          lipAperture: 0.4,
          lipVelocity: 0.14,
        }),
      ),
      ...sequence(16, (t) => speech(1.1 + t)),
    ];
    const events = runDetector(frames).filter((event) => event.type === "repetition");
    expect(events.length).toBeGreaterThan(0);
    expect(events[0].state).toBe("final");
    expect(events[0].confidence).toBeLessThanOrEqual(0.92);
  });

  it("treats a held mouth during silence as a possible block", () => {
    const frames = [
      ...sequence(20, (t) => speech(t)),
      ...sequence(16, (t) =>
        frame({
          t: 1 + t,
          rms: 0.004,
          voiced: 0,
          lipAperture: 0.4,
          jawOpen: 0.55,
          lipCompression: 0.72,
          lipVelocity: 0.005,
          landmarkMotion: 0.005,
          videoPresent: true,
        }),
      ),
    ];
    const events = runDetector(frames);
    expect(events.some((event) => event.type === "block")).toBe(true);
    expect(events.some((event) => event.type === "pause")).toBe(false);
  });

  it("uses a still, non-resting mouth as a block even when the shape is moderate", () => {
    const frames = [
      ...sequence(20, (t) => speech(t)),
      ...sequence(16, (t) =>
        frame({
          t: 1 + t,
          rms: 0.004,
          voiced: 0,
          lipAperture: 0.22,
          jawOpen: 0.2,
          lipCompression: 0.16,
          lipVelocity: 0.02,
          landmarkMotion: 0.03,
          videoPresent: true,
        }),
      ),
    ];
    const events = runDetector(frames);
    expect(events.some((event) => event.type === "block")).toBe(true);
    expect(events.some((event) => event.type === "pause")).toBe(false);
  });

  it("does not keep a short gap as a pause", () => {
    const frames = [
      ...sequence(20, (t) => speech(t)),
      ...sequence(8, (t) =>
        frame({
          t: 1 + t,
          rms: 0.004,
          voiced: 0,
          lipAperture: 0.1,
          jawOpen: 0.05,
          lipCompression: 0.05,
          lipVelocity: 0.01,
          landmarkMotion: 0.01,
        }),
      ),
      ...sequence(8, (t) => speech(1.4 + t)),
    ];
    expect(runDetector(frames).some((event) => event.type === "pause")).toBe(false);
  });

  it("keeps a resting silence as a pause", () => {
    const frames = [
      ...sequence(20, (t) => speech(t)),
      ...sequence(18, (t) =>
        frame({
          t: 1 + t,
          rms: 0.004,
          voiced: 0,
          lipAperture: 0.12,
          jawOpen: 0.08,
          lipCompression: 0.08,
          lipVelocity: 0.01,
          landmarkMotion: 0.01,
        }),
      ),
    ];
    const events = runDetector(frames);
    expect(events.some((event) => event.type === "pause")).toBe(true);
    expect(events.some((event) => event.type === "block")).toBe(false);
  });

  it("does not call a block from silence when the camera is absent", () => {
    const frames = [
      ...sequence(20, (t) => speech(t)),
      ...sequence(16, (t) =>
        frame({
          t: 1 + t,
          rms: 0.004,
          voiced: 0,
          videoPresent: false,
          lipCompression: 0.8,
          jawOpen: 0.6,
          lipAperture: 0.4,
        }),
      ),
    ];
    expect(runDetector(frames).some((event) => event.type === "block")).toBe(false);
  });

  it("notices a restart from the transcript and a steady sound", () => {
    const restart = runDetector(sequence(10, (t) => frame({ t, textPresent: true, restartCue: 0.9, rms: 0.05, voiced: 0.8 })));
    expect(restart.some((event) => event.type === "restart")).toBe(true);
    const held = runDetector(
      sequence(28, (t) => frame({ t, rms: 0.06, voiced: 1, flux: 0.04, autocorr: 0.02, lipAperture: 0.3, jawOpen: 0.2 })),
    );
    expect(held.some((event) => event.type === "prolongation")).toBe(true);
  });

  it("can attach a late transcript without blocking earlier frames", () => {
    let state = emptyDetector();
    for (let i = 0; i < 10; i += 1) state = stepDetector(state, speech(i * 0.05));
    state = applyDelayedCue(
      state,
      { start: 0.1, end: 0.45, kind: "repetition", confidence: 0.8, detail: "The transcript repeats a word" },
      1.2,
    );
    const added = state.events.find((event) => event.timing === "approximate");
    expect(added?.type).toBe("repetition");
    expect(added?.state).toBe("final");
  });
});

describe("language", () => {
  it("finds repetitions and ignores filled pauses", () => {
    expect(findLinguisticEvents("um uh", 0, 1)).toEqual([]);
    expect(findLinguisticEvents("I I want to go", 0, 2).some((cue) => cue.kind === "repetition")).toBe(true);
    expect(findLinguisticEvents("the um the cat", 0, 2).some((cue) => cue.kind === "repetition")).toBe(true);
    expect(findLinguisticEvents("b-b-ball", 0, 1).some((cue) => cue.detail.includes("part of a word"))).toBe(true);
    expect(findLinguisticEvents("I was I was going", 1, 3).some((cue) => cue.detail.includes("phrase"))).toBe(true);
    expect(findLinguisticEvents("I mean we can leave", 0, 2).some((cue) => cue.kind === "restart")).toBe(true);
  });
});

describe("face measures", () => {
  it("normalizes aperture and ignores a rigid head shift", () => {
    const small = face(1, 0);
    const large = face(2, 0);
    expect(lipAperture(small)).toBeCloseTo(lipAperture(large), 5);

    let state = measureLipMotion(small, { refLeft: null, refRight: null, previous: null }).state;
    const shifted = measureLipMotion(face(1, 0, 0.12), state);
    expect(shifted.motion).toBeLessThan(0.01);
    const moved = measureLipMotion(face(1, 0.08), shifted.state);
    expect(moved.motion).toBeGreaterThan(0.02);
  });

  it("describes movement without guessing at emotion or tension", () => {
    expect(lipCompressionShape({ mouthPressLeft: 1, mouthPressRight: 1, mouthRollLower: 1, mouthRollUpper: 1, mouthClose: 1 })).toBe(1);
    const quietMove = describeCoupling(Array.from({ length: 12 }, () => ({ rms: 0.005, aperture: 0.2, silentMove: 0.08 })));
    expect(quietMove).toMatch(/quiet/i);
    expect(containsBannedCoaching([quietMove])).toBe(false);
  });
});

describe("feedback policy", () => {
  const block: SpeechEvent = {
    id: 1,
    type: "block",
    start: 0,
    end: 0.4,
    confidence: 0.8,
    state: "final",
    evidence: [],
    timing: "frame",
  };

  it("does not turn a detection into an instruction", () => {
    const base = { mode: "reminders" as const, now: 30, lastReminderAt: 0, intervalSec: 20, reminderText: "You can take your time." };
    const withEvent = liveMessage({ ...base, detected: [block] });
    const without = liveMessage({ ...base, detected: [] });
    expect(withEvent).toEqual(without);
    expect(withEvent.text).toBe("You can take your time.");
    expect(liveMessage({ ...base, mode: "gentle", detected: [block] }).text).toBeNull();
    expect(liveMessage({ ...base, mode: "quiet", detected: [block] }).showMarkers).toBe(false);
  });

  it("keeps session and weekly notes descriptive", () => {
    const lines = sessionSummary({ duration: 125, events: [block], videoCoverage: 0.2 });
    const weekly = insightLines({
      sessions: [{ startedAt: 1_000, duration: 125, events: [block] }],
      goals: ["company", "notice-blocks"],
      now: 1_000 + 1000,
    });
    expect(containsBannedCoaching([...lines, ...weekly.lead, ...weekly.details])).toBe(false);
    expect(lines.join(" ")).toMatch(/not a fluency score|camera was missing/i);
    expect(weekly.lead.join(" ")).toMatch(/possible block/i);
  });
});

describe("audio features", () => {
  it("measures loudness, a quiet crossing rate, and a tonal peak", () => {
    expect(rms(new Float32Array(16))).toBe(0);
    expect(zeroCrossingRate(new Float32Array([0.2, 0.2, 0.2]))).toBe(0);
    const rate = 16000;
    const tone = new Float32Array(512);
    for (let i = 0; i < tone.length; i += 1) tone[i] = Math.sin((2 * Math.PI * 1000 * i) / rate);
    const centroid = spectralCentroid(magnitudeSpectrum(tone), rate);
    expect(centroid).toBeGreaterThan(0.08);
    expect(centroid).toBeLessThan(0.22);
    const pulsed = Array.from({ length: 24 }, (_, index) => (index % 4 === 0 ? 1 : 0));
    expect(envelopeAutocorr(pulsed)).toBeGreaterThan(0.4);
  });
});

function face(scale: number, lipDy: number, shift = 0): Pt[] {
  const points: Pt[] = Array.from({ length: 320 }, () => ({ x: shift + 0.5, y: 0.5 }));
  points[33] = { x: shift + 0.5 - 0.15 * scale, y: 0.4 };
  points[263] = { x: shift + 0.5 + 0.15 * scale, y: 0.4 };
  for (const index of [0, 13, 14, 17, 61, 78, 82, 87, 291, 308, 312, 317]) {
    const spread = index === 61 ? -0.08 * scale : index === 291 ? 0.08 * scale : 0;
    const open = index === 14 ? 0.06 * scale : 0;
    points[index] = { x: shift + 0.5 + spread, y: 0.55 + open + lipDy };
  }
  return points;
}
