export interface LinguisticCue {
  start: number;
  end: number;
  kind: "repetition" | "restart";
  confidence: number;
  detail: string;
}

const FILLED_PAUSE = /^(um+|uh+|er+|ah+|hm+)$/i;

/**
 * Repetitions and restarts from a transcript span.
 * Filled pauses (um, uh) are removed and are not events. MMSD-Net argues
 * those ordinary disfluencies are not the same as stuttering; this companion
 * follows that distinction.
 * Timing is spread across the span because streaming ASR often has no word times.
 */
export function findLinguisticEvents(text: string, start: number, end: number): LinguisticCue[] {
  const tokens = tokenize(text);
  if (tokens.length === 0 || end <= start) return [];
  const cues: LinguisticCue[] = [];
  const span = Math.max(0.25, end - start);

  for (let i = 0; i < tokens.length; i += 1) {
    if (partWord(tokens[i])) {
      cues.push(place(i, tokens.length, start, span, "repetition", 0.78, "The transcript repeats part of a word"));
    }
  }

  for (let i = 0; i < tokens.length - 1; i += 1) {
    if (tokens[i].includes("-")) continue;
    let run = 1;
    while (i + run < tokens.length && tokens[i + run] === tokens[i] && !tokens[i].includes("-")) run += 1;
    if (run >= 2) {
      cues.push(place(i, tokens.length, start, span, "repetition", 0.8, "The transcript repeats a word"));
      i += run - 1;
    }
  }

  for (const size of [3, 2]) {
    for (let i = 0; i + size * 2 <= tokens.length; i += 1) {
      const left = tokens.slice(i, i + size).join(" ");
      const right = tokens.slice(i + size, i + size * 2).join(" ");
      if (left === right && !cues.some((cue) => cue.kind === "repetition" && Math.abs(cue.start - start) < span)) {
        cues.push(place(i, tokens.length, start, span, "repetition", 0.74, "The transcript repeats a phrase"));
      }
    }
  }

  const joined = tokens.join(" ");
  if (joined.includes("i mean") || joined.includes("or rather")) {
    cues.push({
      start,
      end: start + span,
      kind: "restart",
      confidence: 0.58,
      detail: "The transcript includes a revision phrase",
    });
  }
  return dedupe(cues);
}

function tokenize(text: string): string[] {
  const raw = text.match(/[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*/gu) ?? [];
  return raw.map((token) => token.toLowerCase()).filter((token) => !FILLED_PAUSE.test(token));
}

function partWord(token: string): boolean {
  const bits = token.split("-").filter(Boolean);
  if (bits.length < 2) return false;
  const counts = new Map<string, number>();
  for (const bit of bits) {
    if (bit.length > 3) continue;
    counts.set(bit, (counts.get(bit) ?? 0) + 1);
  }
  return [...counts.values()].some((count) => count >= 2);
}

function place(
  index: number,
  count: number,
  start: number,
  span: number,
  kind: "repetition" | "restart",
  confidence: number,
  detail: string,
): LinguisticCue {
  const cueStart = start + (index / count) * span;
  const cueEnd = Math.max(cueStart + 0.2, start + ((index + 1) / count) * span);
  return { start: cueStart, end: cueEnd, kind, confidence, detail };
}

function dedupe(cues: LinguisticCue[]): LinguisticCue[] {
  const kept: LinguisticCue[] = [];
  for (const cue of cues) {
    const clash = kept.some(
      (other) => other.kind === cue.kind && other.detail === cue.detail && Math.abs(other.start - cue.start) < 0.15,
    );
    if (!clash) kept.push(cue);
  }
  return kept;
}
