"""Turn frame labels into timestamped events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from companion.config import CLASS_NAMES


@dataclass(frozen=True)
class SpeechEvent:
    start: float
    end: float
    label: str
    confidence: float
    state: str


def decode_frame_labels(
    labels: np.ndarray,
    frame_sec: float = 0.05,
    min_duration: dict[str, float] | None = None,
    confidences: np.ndarray | None = None,
    provisional_after: float | None = None,
) -> list[SpeechEvent]:
    """Run-length decode frame ids.

    `provisional_after` marks events that touch the open edge of a window.
    Fluent frames close an event and are not emitted.
    """

    if labels.ndim != 1:
        raise ValueError("labels must be a 1-D array of class ids")
    floors = {
        "repetition": 0.15,
        "prolongation": 0.30,
        "block": 0.20,
        "restart": 0.15,
        "pause": 0.25,
    }
    if min_duration:
        floors.update(min_duration)

    events: list[SpeechEvent] = []
    index = 0
    total = int(labels.shape[0])
    while index < total:
        class_id = int(labels[index])
        if class_id <= 0 or class_id >= len(CLASS_NAMES):
            index += 1
            continue
        end = index + 1
        while end < total and int(labels[end]) == class_id:
            end += 1
        name = CLASS_NAMES[class_id]
        start_t = index * frame_sec
        end_t = end * frame_sec
        if end_t - start_t >= floors.get(name, 0.15):
            if confidences is None:
                confidence = 0.7
            else:
                confidence = float(np.clip(confidences[index:end].mean(), 0.0, 1.0))
            state = "provisional" if provisional_after is not None and end_t >= provisional_after else "final"
            events.append(
                SpeechEvent(
                    start=start_t,
                    end=end_t,
                    label=name,
                    confidence=confidence,
                    state=state,
                )
            )
        index = end
    return events


def iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return intersection / union
