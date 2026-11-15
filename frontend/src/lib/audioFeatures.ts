import { clamp } from "../types";

export function rms(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let energy = 0;
  for (let i = 0; i < samples.length; i += 1) energy += samples[i] * samples[i];
  return Math.sqrt(energy / samples.length);
}

export function zeroCrossingRate(samples: Float32Array): number {
  if (samples.length < 2) return 0;
  let crossings = 0;
  for (let i = 1; i < samples.length; i += 1) {
    if ((samples[i - 1] >= 0 && samples[i] < 0) || (samples[i - 1] < 0 && samples[i] >= 0)) {
      crossings += 1;
    }
  }
  return crossings / (samples.length - 1);
}

/** Radix-2 magnitude spectrum of the last 512 samples. */
export function magnitudeSpectrum(input: Float32Array): Float32Array {
  const n = 512;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  const start = Math.max(0, input.length - n);
  for (let i = 0; i < n; i += 1) {
    const sample = input[start + i] ?? 0;
    const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    re[i] = sample * window;
  }
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const swap = re[i];
      re[i] = re[j];
      re[j] = swap;
      const swapIm = im[i];
      im[i] = im[j];
      im[j] = swapIm;
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const theta = (-2 * Math.PI) / size;
    const stepRe = Math.cos(theta);
    const stepIm = Math.sin(theta);
    for (let offset = 0; offset < n; offset += size) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < half; k += 1) {
        const even = offset + k;
        const odd = even + half;
        const tRe = curRe * re[odd] - curIm * im[odd];
        const tIm = curRe * im[odd] + curIm * re[odd];
        re[odd] = re[even] - tRe;
        im[odd] = im[even] - tIm;
        re[even] += tRe;
        im[even] += tIm;
        const nextRe = curRe * stepRe - curIm * stepIm;
        curIm = curRe * stepIm + curIm * stepRe;
        curRe = nextRe;
      }
    }
  }
  const mag = new Float32Array(n / 2);
  for (let i = 0; i < mag.length; i += 1) mag[i] = Math.hypot(re[i], im[i]);
  return mag;
}

export function spectralCentroid(mag: Float32Array, sampleRate: number): number {
  let weighted = 0;
  let total = 0;
  const bins = mag.length * 2;
  for (let i = 1; i < mag.length; i += 1) {
    const freq = (i * sampleRate) / bins;
    weighted += freq * mag[i];
    total += mag[i];
  }
  if (total <= 1e-8) return 0;
  return clamp(weighted / total / (sampleRate / 2));
}

export function spectralFlux(mag: Float32Array, previous: Float32Array | null): number {
  if (!previous || previous.length !== mag.length) return 0;
  let rise = 0;
  let total = 0;
  for (let i = 0; i < mag.length; i += 1) {
    rise += Math.max(0, mag[i] - previous[i]);
    total += mag[i];
  }
  if (total <= 1e-8) return 0;
  return clamp(rise / total);
}

export function lowBandRatio(mag: Float32Array, sampleRate: number): number {
  const bins = mag.length * 2;
  const cutoff = Math.max(1, Math.floor((1000 * bins) / sampleRate));
  let low = 0;
  let total = 0;
  for (let i = 1; i < mag.length; i += 1) {
    total += mag[i];
    if (i <= cutoff) low += mag[i];
  }
  if (total <= 1e-8) return 0;
  return clamp(low / total);
}

/** Peak normalized autocorrelation of a loudness envelope at syllable-sized lags. */
export function envelopeAutocorr(envelope: number[]): number {
  if (envelope.length < 12) return 0;
  const mean = envelope.reduce((sum, value) => sum + value, 0) / envelope.length;
  let energy = 0;
  const centered = envelope.map((value) => {
    const shifted = value - mean;
    energy += shifted * shifted;
    return shifted;
  });
  if (energy < 1e-6) return 0;
  let best = 0;
  const maxLag = Math.min(10, envelope.length - 2);
  for (let lag = 2; lag <= maxLag; lag += 1) {
    let dot = 0;
    for (let i = lag; i < centered.length; i += 1) dot += centered[i] * centered[i - lag];
    best = Math.max(best, dot / energy);
  }
  return clamp(best);
}

export function voicedScore(rmsValue: number, zcr: number, lowBand: number): number {
  if (rmsValue > 0.02 && zcr < 0.18 && lowBand > 0.35) return 1;
  if (rmsValue > 0.03 && zcr < 0.28) return 0.55;
  return 0;
}
