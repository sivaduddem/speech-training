"""Synthetic clips for the harness, and a loader for licensed local data.

Synthetic clips are built so silent blocks and natural pauses share a weak
audio cue and differ in the visual channels. That lets an ablation ask
whether the visual encoder is being used. It does not imitate a real
speaker, and it is not a substitute for SEP-28k or FluencyBank.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from companion.config import CLASS_INDEX, CLASS_NAMES


@dataclass
class Example:
    speaker: str
    audio: np.ndarray
    visual: np.ndarray
    text: np.ndarray
    labels: np.ndarray
    video_missing: bool = False
    text_missing: bool = False
    frame_sec: float = 0.05

    @property
    def clip_label(self) -> int:
        disfluency = {
            CLASS_INDEX["repetition"],
            CLASS_INDEX["prolongation"],
            CLASS_INDEX["block"],
            CLASS_INDEX["restart"],
        }
        return int(any(int(label) in disfluency for label in self.labels))


def _paint(target: np.ndarray, start: int, end: int, channel: int, value: float, rng: np.random.Generator) -> None:
    noise = rng.normal(0.0, 0.08, size=end - start)
    target[start:end, channel] = value + noise


def make_example(rng: np.random.Generator, speaker: str, speaker_bias: float, length: int = 80) -> Example:
    audio = rng.normal(0.0, 0.12, size=(length, 8)).astype(np.float32)
    visual = rng.normal(0.0, 0.12, size=(length, 6)).astype(np.float32)
    text = rng.normal(0.0, 0.08, size=(length, 4)).astype(np.float32)
    labels = np.zeros(length, dtype=np.int64)
    audio[:, 5] += 0.4
    audio[:, 6] += speaker_bias

    roll = rng.random()
    events: list[tuple[int, int, str]] = []
    if roll < 0.46:
        kind = None
    elif roll < 0.62:
        kind = "pause"
    elif roll < 0.74:
        kind = "block"
    elif roll < 0.84:
        kind = "repetition"
    elif roll < 0.93:
        kind = "prolongation"
    else:
        kind = "restart"
    if kind is not None and length >= 24:
        width = int(rng.integers(12, 20))
        start = int(rng.integers(4, length - width))
        events.append((start, start + width, kind))

    for start, end, kind in events:
        labels[start:end] = CLASS_INDEX[kind]
        if kind == "repetition":
            _paint(audio, start, end, 0, 1.4, rng)
            _paint(text, start, end, 0, 1.5, rng)
            _paint(visual, start, end, 0, 0.5, rng)
        elif kind == "prolongation":
            _paint(audio, start, end, 1, 1.5, rng)
            _paint(text, start, end, 1, 0.8, rng)
        elif kind == "block":
            # Same silence channel as a pause, plus a small audio hint and a
            # strong held-posture visual channel.
            _paint(audio, start, end, 2, 0.55, rng)
            _paint(audio, start, end, 3, 0.25, rng)
            _paint(visual, start, end, 1, 1.7, rng)
            visual[start:end, 4] = rng.normal(0.0, 0.05, size=end - start)
        elif kind == "pause":
            _paint(audio, start, end, 2, 0.55, rng)
            _paint(visual, start, end, 2, 1.6, rng)
            visual[start:end, 4] = rng.normal(0.0, 0.05, size=end - start)
        elif kind == "restart":
            _paint(text, start, end, 2, 1.7, rng)
            _paint(audio, start, end, 4, 0.2, rng)

    fluent = labels == 0
    visual[fluent, 3] += 0.3
    return Example(speaker=speaker, audio=audio, visual=visual, text=text, labels=labels)


def make_split(
    n_train_speakers: int = 8,
    n_test_speakers: int = 4,
    clips_per_speaker: int = 24,
    seed: int = 0,
) -> tuple[list[Example], list[Example]]:
    rng = np.random.default_rng(seed)
    train: list[Example] = []
    test: list[Example] = []
    for speaker_index in range(n_train_speakers):
        bias = -0.4 + 0.1 * speaker_index
        for _ in range(clips_per_speaker):
            train.append(make_example(rng, f"train-{speaker_index}", bias))
    for speaker_index in range(n_test_speakers):
        bias = 0.7 + 0.15 * speaker_index
        for _ in range(clips_per_speaker):
            example = make_example(rng, f"heldout-{speaker_index}", bias)
            example.audio = (
                example.audio + rng.normal(0.0, 0.05, size=example.audio.shape)
            ).astype(np.float32)
            test.append(example)
    return train, test


def load_manifest(path: str | Path) -> list[Example]:
    """Load a local JSONL manifest of precomputed, licensed features.

    Each line points at arrays stored next to the manifest. This loader does
    not download SEP-28k, FluencyBank, or any other dataset.

    {"speaker": "s1", "audio": "a.npy", "visual": "v.npy", "text": "t.npy",
     "labels": "y.npy", "frame_sec": 0.05}

    `visual` or `text` may be null. Label ids follow CLASS_NAMES.
    """

    manifest = Path(path)
    examples: list[Example] = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            root = manifest.parent

            def resolve(name: str) -> np.ndarray | None:
                value = row.get(name)
                if value is None:
                    return None
                return np.load(root / value)

            audio = resolve("audio")
            labels = resolve("labels")
            if audio is None or labels is None:
                raise ValueError(f"{manifest}:{line_number} needs audio and labels")
            visual = resolve("visual")
            text = resolve("text")
            video_missing = visual is None
            text_missing = text is None
            if visual is None:
                visual = np.zeros((audio.shape[0], 6), dtype=np.float32)
            if text is None:
                text = np.zeros((audio.shape[0], 4), dtype=np.float32)
            if labels.ndim != 1 or labels.shape[0] != audio.shape[0]:
                raise ValueError(f"{manifest}:{line_number} labels must align with audio frames")
            unknown = sorted(set(int(item) for item in labels) - set(range(len(CLASS_NAMES))))
            if unknown:
                raise ValueError(f"{manifest}:{line_number} has unknown label ids {unknown}")
            examples.append(
                Example(
                    speaker=str(row.get("speaker", "unknown")),
                    audio=audio.astype(np.float32),
                    visual=visual.astype(np.float32),
                    text=text.astype(np.float32),
                    labels=labels.astype(np.int64),
                    video_missing=video_missing,
                    text_missing=text_missing,
                    frame_sec=float(row.get("frame_sec", 0.05)),
                )
            )
    return examples
