import { clamp } from "../types";

export interface Pt {
  x: number;
  y: number;
  z?: number;
}

export const IDX = {
  upperInner: 13,
  lowerInner: 14,
  mouthLeft: 61,
  mouthRight: 291,
  leftEye: 33,
  rightEye: 263,
} as const;

export const LIP_INDEXES = [0, 13, 14, 17, 61, 78, 82, 87, 291, 308, 312, 317];

export function distance(a: Pt, b: Pt): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function interocular(landmarks: Pt[]): number {
  return Math.max(distance(landmarks[IDX.leftEye], landmarks[IDX.rightEye]), 1e-4);
}

/** Lip opening divided by eye spacing, so face size and distance from the camera drop out. */
export function lipAperture(landmarks: Pt[]): number {
  return distance(landmarks[IDX.upperInner], landmarks[IDX.lowerInner]) / interocular(landmarks);
}

export function mouthWidth(landmarks: Pt[]): number {
  return distance(landmarks[IDX.mouthLeft], landmarks[IDX.mouthRight]) / interocular(landmarks);
}

export function mapSimilarity(point: Pt, srcA: Pt, srcB: Pt, dstA: Pt, dstB: Pt): Pt {
  const svx = srcB.x - srcA.x;
  const svy = srcB.y - srcA.y;
  const dvx = dstB.x - dstA.x;
  const dvy = dstB.y - dstA.y;
  const srcLen = Math.hypot(svx, svy) || 1e-6;
  const dstLen = Math.hypot(dvx, dvy) || 1e-6;
  const scale = dstLen / srcLen;
  const rotation = Math.atan2(dvy, dvx) - Math.atan2(svy, svx);
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  const px = point.x - srcA.x;
  const py = point.y - srcA.y;
  return {
    x: dstA.x + scale * (cos * px - sin * py),
    y: dstA.y + scale * (sin * px + cos * py),
  };
}

export interface MotionState {
  refLeft: Pt | null;
  refRight: Pt | null;
  previous: Pt[] | null;
}

/**
 * Lip-landmark speed after removing head translation, scale, and roll.
 * The reference is the eye line from the first tracked frame.
 */
export function measureLipMotion(landmarks: Pt[], state: MotionState): { motion: number; state: MotionState } {
  const left = landmarks[IDX.leftEye];
  const right = landmarks[IDX.rightEye];
  if (!left || !right) return { motion: 0, state };
  const lips = LIP_INDEXES.map((index) => landmarks[index]).filter((point): point is Pt => Boolean(point));
  if (!state.refLeft || !state.refRight) {
    return { motion: 0, state: { refLeft: { ...left }, refRight: { ...right }, previous: lips.map((point) => ({ ...point })) } };
  }
  const canonical = lips.map((point) => mapSimilarity(point, left, right, state.refLeft as Pt, state.refRight as Pt));
  let motion = 0;
  if (state.previous && state.previous.length === canonical.length) {
    let total = 0;
    for (let i = 0; i < canonical.length; i += 1) {
      total += distance(canonical[i], state.previous[i]);
    }
    motion = total / canonical.length;
  }
  return { motion, state: { refLeft: state.refLeft, refRight: state.refRight, previous: canonical } };
}

export function poseFromMatrix(matrix: number[] | null, leftEye: Pt, rightEye: Pt): { yaw: number; pitch: number; roll: number } {
  if (matrix && matrix.length >= 16) {
    const r10 = matrix[1];
    const r00 = matrix[0];
    const r20 = matrix[2];
    const r21 = matrix[6];
    const r22 = matrix[10];
    return {
      yaw: Math.atan2(r10, r00),
      pitch: Math.asin(clamp(-r20, -1, 1)),
      roll: Math.atan2(r21, r22),
    };
  }
  return { yaw: 0, pitch: 0, roll: Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) };
}

/** Geometric blendshape score. This is not muscle tension, effort, or emotion. */
export function lipCompressionShape(blend: Record<string, number>): number {
  const names = ["mouthPressLeft", "mouthPressRight", "mouthRollLower", "mouthRollUpper", "mouthClose"];
  const total = names.reduce((sum, name) => sum + (blend[name] ?? 0), 0);
  return clamp(total / names.length);
}

export function jawOpening(blend: Record<string, number>): number {
  return clamp(blend.jawOpen ?? 0);
}

export interface CouplingSample {
  rms: number;
  aperture: number;
  silentMove: number;
}

export function describeCoupling(samples: CouplingSample[]): string {
  if (samples.length < 10) return "Waiting for a little more speech and movement.";
  const rmsValues = samples.map((sample) => sample.rms);
  const aperture = samples.map((sample) => sample.aperture);
  const relation = correlation(rmsValues, aperture);
  const meanRms = average(rmsValues);
  const meanSilent = average(samples.map((sample) => sample.silentMove));
  const apertureSpread = deviation(aperture);
  if (meanRms < 0.02 && meanSilent > 0.04) {
    return "The mouth is moving during a quiet stretch.";
  }
  if (meanRms > 0.03 && apertureSpread < 0.015) {
    return "Sound is present while lip opening is staying fairly still.";
  }
  if (relation > 0.45) return "Lip opening and loudness are rising and falling together.";
  return "No strong link between loudness and lip opening in this stretch.";
}

function correlation(xs: number[], ys: number[]): number {
  const meanX = average(xs);
  const meanY = average(ys);
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < xs.length; i += 1) {
    const a = xs[i] - meanX;
    const b = ys[i] - meanY;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  if (dx < 1e-8 || dy < 1e-8) return 0;
  return num / Math.sqrt(dx * dy);
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function deviation(values: number[]): number {
  const mean = average(values);
  return Math.sqrt(average(values.map((value) => (value - mean) ** 2)));
}
