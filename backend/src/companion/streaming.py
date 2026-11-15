"""Sliding-window buffers and overlap smoothing for research inference.

The browser runs its own causal detector. This helper is the server-side
counterpart: features that share a clock go into one window, the model may
see the whole window, and only frames older than the hop are finalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyncedFrame:
    time: float
    audio: np.ndarray
    visual: np.ndarray | None
    text: np.ndarray | None


@dataclass
class Window:
    start: float
    times: np.ndarray
    audio: np.ndarray
    visual: np.ndarray
    text: np.ndarray
    video_present: np.ndarray
    text_present: np.ndarray


@dataclass
class SlidingBuffer:
    window_sec: float = 2.0
    hop_sec: float = 0.4
    frame_sec: float = 0.05
    frames: list[SyncedFrame] = field(default_factory=list)
    emitted_until: float = 0.0

    def push(self, frame: SyncedFrame) -> None:
        self.frames.append(frame)
        horizon = frame.time - self.window_sec - self.hop_sec
        if horizon > 0:
            self.frames = [item for item in self.frames if item.time >= horizon]

    def ready(self) -> bool:
        if not self.frames:
            return False
        span = self.frames[-1].time - self.frames[0].time
        pending = self.frames[-1].time - self.emitted_until
        return span >= min(self.window_sec, 0.5) and pending >= self.hop_sec

    def window(self, visual_dim: int | None = None, text_dim: int | None = None) -> Window:
        if not self.frames:
            raise RuntimeError("buffer is empty")
        end = self.frames[-1].time
        start = max(self.frames[0].time, end - self.window_sec)
        chosen = [frame for frame in self.frames if frame.time >= start - 1e-6]
        audio_dim = chosen[0].audio.shape[-1]
        seen_visual = [frame.visual.shape[-1] for frame in chosen if frame.visual is not None]
        seen_text = [frame.text.shape[-1] for frame in chosen if frame.text is not None]
        visual_dim = seen_visual[0] if seen_visual else (visual_dim or 1)
        text_dim = seen_text[0] if seen_text else (text_dim or 1)
        count = len(chosen)
        audio = np.zeros((count, audio_dim), dtype=np.float32)
        visual = np.zeros((count, visual_dim), dtype=np.float32)
        text = np.zeros((count, text_dim), dtype=np.float32)
        video_present = np.zeros(count, dtype=bool)
        text_present = np.zeros(count, dtype=bool)
        times = np.zeros(count, dtype=np.float32)
        for index, frame in enumerate(chosen):
            audio[index] = frame.audio
            times[index] = frame.time
            if frame.visual is not None:
                visual[index] = frame.visual
                video_present[index] = True
            if frame.text is not None:
                width = min(text_dim, frame.text.shape[-1])
                text[index, :width] = frame.text[:width]
                text_present[index] = True
        return Window(start, times, audio, visual, text, video_present, text_present)

    def mark_emitted(self, until: float) -> None:
        self.emitted_until = until


@dataclass
class OverlapSmoother:
    """Average class logits that land in the same time bin across hops."""

    frame_sec: float = 0.05
    n_classes: int = 6
    bins: dict[int, list[np.ndarray]] = field(default_factory=dict)

    def add(self, times: np.ndarray, logits: np.ndarray) -> None:
        if len(times) != len(logits):
            raise ValueError("times and logits must align")
        for time, row in zip(times, logits):
            key = int(round(float(time) / self.frame_sec))
            self.bins.setdefault(key, []).append(np.asarray(row, dtype=np.float32))

    def labels_at(self, now: float, finalize_before: float) -> list[tuple[float, int, str, float]]:
        rows: list[tuple[float, int, str, float]] = []
        for key in sorted(self.bins):
            time = key * self.frame_sec
            if time > now + 1e-6:
                continue
            stacked = np.stack(self.bins[key], axis=0)
            mean_logits = stacked.mean(axis=0)
            shifted = mean_logits - mean_logits.max()
            probs = np.exp(shifted)
            probs = probs / probs.sum()
            class_id = int(probs.argmax())
            state = "final" if time < finalize_before else "provisional"
            rows.append((time, class_id, state, float(probs[class_id])))
        return rows
