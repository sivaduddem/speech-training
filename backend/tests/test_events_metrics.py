import numpy as np
import pytest

from companion.config import CLASS_INDEX
from companion.events import SpeechEvent, decode_frame_labels
from companion.metrics import (
    expected_calibration_error,
    localization_error,
    pause_false_positive_rate,
    score_events,
)


def test_decode_and_perfect_match():
    labels = np.array([0, 0, 1, 1, 1, 1, 0, 5, 5, 5, 5, 5, 5, 0])
    events = decode_frame_labels(labels, frame_sec=0.05, provisional_after=10)
    assert [event.label for event in events] == ["repetition", "pause"]
    scores = score_events(events, events)
    assert scores["repetition"].f1 == 1
    assert scores["pause"].f1 == 1


def test_pause_frames_labeled_as_blocks_count_as_false_positives():
    pause = CLASS_INDEX["pause"]
    block = CLASS_INDEX["block"]
    reference = np.array([pause, pause, pause, pause])
    predicted = np.array([block, pause, block, pause])
    assert pause_false_positive_rate(reference, predicted) == 0.5


def test_localization_error_on_shifted_event():
    reference = [SpeechEvent(1.0, 1.5, "block", 0.8, "final")]
    hypothesis = [SpeechEvent(1.1, 1.55, "block", 0.8, "final")]
    onset, offset = localization_error(reference, hypothesis)
    assert onset == pytest.approx(0.1)
    assert offset == pytest.approx(0.05)


def test_calibration_of_certain_correct_predictions():
    probabilities = np.eye(2)
    labels = np.array([0, 1])
    assert expected_calibration_error(probabilities, labels) == 0
