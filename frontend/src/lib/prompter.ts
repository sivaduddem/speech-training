export interface ScriptWord {
  text: string;
  norm: string;
}

const FILLER = /^(um+|uh+|er+|ah+|hm+)$/;

export function normalizeWord(word: string): string {
  return word.toLowerCase().replace(/[’]/g, "'").replace(/[^a-z0-9']+/g, "");
}

export function spokenWords(text: string): string[] {
  const raw = text.match(/[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*/gu) ?? [];
  return raw.map(normalizeWord).filter((word) => word.length > 0 && !FILLER.test(word));
}

export interface PassageSentence {
  words: ScriptWord[];
  start: number;
}

export function parseScript(body: string): ScriptWord[][] {
  return body
    .split(/\n+/)
    .map((line) =>
      line
        .split(/\s+/)
        .filter(Boolean)
        .map((text) => ({ text, norm: normalizeWord(text) }))
        .filter((word) => word.norm.length > 0),
    )
    .filter((line) => line.length > 0);
}

/** Break a passage into sentences so the reader can move a line at a time. */
export function sentencesOf(lines: ScriptWord[][]): PassageSentence[] {
  const sentences: PassageSentence[] = [];
  let words: ScriptWord[] = [];
  let start = 0;
  let flat = 0;
  const flush = () => {
    if (words.length === 0) return;
    sentences.push({ words, start });
    words = [];
  };
  for (const line of lines) {
    for (const word of line) {
      if (words.length === 0) start = flat;
      words.push(word);
      flat += 1;
      if (/[.!?]["']?$/.test(word.text)) flush();
    }
    flush();
  }
  return sentences;
}

/**
 * Which sentence to show. A sentence gives way once about half of its words
 * were heard, so a few missed words do not hold the passage.
 */
export function sentencePage(sentences: PassageSentence[], cursor: number): number {
  if (sentences.length === 0) return 0;
  for (let index = 0; index < sentences.length; index += 1) {
    const sentence = sentences[index];
    const length = sentence.words.length;
    const heard = Math.max(0, Math.min(cursor, sentence.start + length) - sentence.start);
    const needed = length <= 4 ? length : Math.max(3, Math.ceil(length * 0.5));
    if (heard < needed) return index;
  }
  return sentences.length - 1;
}

function close(heard: string, expected: string): boolean {
  if (heard === expected) return true;
  if (heard.length < 3 || expected.length < 3) return false;
  const [shorter, longer] = heard.length < expected.length ? [heard, expected] : [expected, heard];
  return longer.startsWith(shorter) && shorter.length / longer.length >= 0.6;
}

/**
 * How many script words the spoken words have already covered.
 * Repeating a word does not skip ahead. A later word can catch the cursor up
 * when the words around it agree.
 */
export function spokenCursor(script: string[], spoken: string[]): number {
  let index = 0;
  for (let heard = 0; heard < spoken.length && index < script.length; heard += 1) {
    const word = spoken[heard];
    if (close(word, script[index])) {
      index += 1;
      continue;
    }
    if (index > 0 && close(word, script[index - 1])) continue;

    let ahead = -1;
    for (let step = 1; step <= 8 && index + step < script.length; step += 1) {
      if (close(word, script[index + step])) {
        ahead = step;
        break;
      }
    }
    if (ahead < 0) continue;
    const next = spoken[heard + 1];
    const landed = index + ahead;
    const confirmed =
      ahead <= 2 ||
      (next !== undefined &&
        (close(next, script[landed]) || (landed + 1 < script.length && close(next, script[landed + 1]))));
    if (confirmed) index = landed + 1;
  }
  return index;
}
