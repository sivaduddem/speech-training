"""Event-level precision, recall, localization, and calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from companion.config import CLASS_NAMES, DISFLUENCY_LABELS
from companion.events import SpeechEvent, iou


@dataclass
class ClassScore:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvalReport:
    per_class: dict[str, ClassScore]
    micro_f1: float
    onset_mae: float | None
    offset_mae: float | None
    pause_false_positive_rate: float
    silent_block_f1: float
    clip_f1: float
    latency_ms: float | None = None
    calibration_error: float | None = None


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def match_events(
    reference: list[SpeechEvent],
    hypothesis: list[SpeechEvent],
    iou_threshold: float = 0.3,
) -> tuple[list[tuple[SpeechEvent, SpeechEvent]], list[SpeechEvent], list[SpeechEvent]]:
    """Greedy one-to-one match within each label."""

    pairs: list[tuple[SpeechEvent, SpeechEvent]] = []
    used_ref: set[int] = set()
    used_hyp: set[int] = set()
    candidates: list[tuple[float, int, int]] = []
    for ref_index, ref in enumerate(reference):
        for hyp_index, hyp in enumerate(hypothesis):
            if ref.label != hyp.label:
                continue
            overlap = iou(ref.start, ref.end, hyp.start, hyp.end)
            if overlap >= iou_threshold:
                candidates.append((overlap, ref_index, hyp_index))
    candidates.sort(reverse=True)
    for _, ref_index, hyp_index in candidates:
        if ref_index in used_ref or hyp_index in used_hyp:
            continue
        used_ref.add(ref_index)
        used_hyp.add(hyp_index)
        pairs.append((reference[ref_index], hypothesis[hyp_index]))
    missed = [event for index, event in enumerate(reference) if index not in used_ref]
    extra = [event for index, event in enumerate(hypothesis) if index not in used_hyp]
    return pairs, extra, missed


def score_events(
    reference: list[SpeechEvent],
    hypothesis: list[SpeechEvent],
    iou_threshold: float = 0.3,
    labels: tuple[str, ...] = tuple(name for name in CLASS_NAMES if name != "fluent"),
) -> dict[str, ClassScore]:
    scores: dict[str, ClassScore] = {}
    for label in labels:
        ref = [event for event in reference if event.label == label]
        hyp = [event for event in hypothesis if event.label == label]
        pairs, extra, missed = match_events(ref, hyp, iou_threshold=iou_threshold)
        tp = len(pairs)
        fp = len(extra)
        fn = len(missed)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores[label] = ClassScore(precision, recall, _safe_f1(precision, recall), support=len(ref))
    return scores


def localization_error(
    reference: list[SpeechEvent],
    hypothesis: list[SpeechEvent],
    iou_threshold: float = 0.3,
) -> tuple[float | None, float | None]:
    pairs, _, _ = match_events(reference, hypothesis, iou_threshold=iou_threshold)
    interesting = [(ref, hyp) for ref, hyp in pairs if ref.label in DISFLUENCY_LABELS]
    if not interesting:
        return None, None
    onset = float(np.mean([abs(ref.start - hyp.start) for ref, hyp in interesting]))
    offset = float(np.mean([abs(ref.end - hyp.end) for ref, hyp in interesting]))
    return onset, offset


def pause_false_positive_rate(reference_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Share of true pause frames labeled as a disfluency."""

    pause_id = CLASS_NAMES.index("pause")
    disfluency_ids = {CLASS_NAMES.index(name) for name in DISFLUENCY_LABELS}
    mask = reference_labels == pause_id
    total = int(mask.sum())
    if total == 0:
        return 0.0
    predicted = predicted_labels[mask]
    mistakes = sum(int(class_id) in disfluency_ids for class_id in predicted)
    return mistakes / total


def binary_f1(reference: np.ndarray, predicted: np.ndarray) -> float:
    reference = reference.astype(bool)
    predicted = predicted.astype(bool)
    tp = int(np.logical_and(reference, predicted).sum())
    fp = int(np.logical_and(~reference, predicted).sum())
    fn = int(np.logical_and(reference, ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return _safe_f1(precision, recall)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE of the predicted class confidence. Lower is better calibrated."""

    if probabilities.size == 0:
        return 0.0
    chosen = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(labels)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (chosen >= left) & (chosen < right if right < 1 else chosen <= right)
        count = int(in_bin.sum())
        if count == 0:
            continue
        accuracy = float(correct[in_bin].mean())
        confidence = float(chosen[in_bin].mean())
        error += (count / total) * abs(accuracy - confidence)
    return float(error)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, steps: int = 80) -> float:
    """Fit a scalar temperature on validation logits. Returns T >= 0.05."""

    import torch

    logit_tensor = torch.tensor(logits, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    log_temp = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.2, max_iter=steps)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temp.exp().clamp(min=0.05)
        loss = torch.nn.functional.cross_entropy(logit_tensor / temperature, label_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temp.detach().exp().clamp(min=0.05))
