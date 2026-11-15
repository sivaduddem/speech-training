import { describe, expect, it } from "vitest";
import { parseScript, sentencePage, sentencesOf, spokenCursor, spokenWords } from "./prompter";

const SCRIPT = spokenWords("I was thinking we could meet on Thursday around four");

describe("spokenCursor", () => {
  it("follows an exact reading", () => {
    expect(spokenCursor(SCRIPT, spokenWords("I was thinking we could meet on Thursday"))).toBe(8);
  });

  it("does not skip ahead when a word is repeated", () => {
    expect(spokenCursor(SCRIPT, spokenWords("I was thinking thinking we could"))).toBe(5);
  });

  it("ignores fillers and words that are not in the passage", () => {
    expect(spokenCursor(SCRIPT, spokenWords("um I was uh going to thinking we could"))).toBe(5);
  });

  it("catches up when a missed word is followed by the next ones", () => {
    expect(spokenCursor(SCRIPT, spokenWords("I was thinking we meet on Thursday around"))).toBe(9);
  });

  it("stays put until the passage is actually reached", () => {
    expect(spokenCursor(SCRIPT, spokenWords("hello there"))).toBe(0);
  });

  it("moves to the next sentence once half the words were heard", () => {
    const lines = parseScript("I was thinking we could meet on Thursday. The cafe opens at eight.");
    const sentences = sentencesOf(lines);
    expect(sentences.map((sentence) => sentence.words.length)).toEqual([8, 5]);
    expect(sentencePage(sentences, 0)).toBe(0);
    expect(sentencePage(sentences, 3)).toBe(0);
    expect(sentencePage(sentences, 4)).toBe(1);
    expect(sentencePage(sentences, 8)).toBe(1);
  });

  it("keeps punctuation on the displayed word and out of the match", () => {
    const lines = parseScript("Meet on Thursday.\nThe cafe is open.");
    expect(lines[0].map((word) => word.text)).toEqual(["Meet", "on", "Thursday."]);
    expect(lines[0].map((word) => word.norm)).toEqual(["meet", "on", "thursday"]);
    expect(lines).toHaveLength(2);
  });
});
