"""Train the harness and write an honest ablation report.

The numbers describe synthetic or locally supplied features. They are not a
reproduction of MMSD-Net's published F1, and they are not a claim about the
live companion.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from companion.baselines import AudioConvLSTM, AudioTDNN
from companion.config import CLASS_NAMES, DISFLUENCY_LABELS, ModelConfig
from companion.data import Example, load_manifest, make_split
from companion.events import decode_frame_labels
from companion.metrics import (
    binary_f1,
    expected_calibration_error,
    fit_temperature,
    localization_error,
    pause_false_positive_rate,
    score_events,
)
from companion.model import CompanionModel
from companion.train import collate, predict_example, train_model

MODALITY = {
    "tdnn-audio": (False, False),
    "convlstm-audio": (False, False),
    "companion-audio": (False, False),
    "companion-audio+video": (True, False),
    "companion-audio+video+text": (True, True),
}


def _score_split(model, examples: list[Example], device: str, video: bool, text: bool, frame_sec: float) -> dict:
    reference_events = []
    hypothesis_events = []
    pause_rates = []
    clip_ref = []
    clip_hyp = []
    prob_rows = []
    label_rows = []
    block_ref = []
    block_hyp = []
    for example in examples:
        probabilities, predicted, clip_probability, _logits = predict_example(
            model, example, device=device, video=video, text=text
        )
        reference = decode_frame_labels(example.labels, frame_sec=frame_sec)
        hypothesis = decode_frame_labels(predicted, frame_sec=frame_sec, confidences=probabilities.max(axis=1))
        reference_events.extend(reference)
        hypothesis_events.extend(hypothesis)
        pause_rates.append(pause_false_positive_rate(example.labels, predicted))
        clip_ref.append(example.clip_label)
        clip_hyp.append(int(clip_probability >= 0.5))
        prob_rows.append(probabilities)
        label_rows.append(example.labels)
        block_ref.extend([event for event in reference if event.label == "block"])
        block_hyp.extend([event for event in hypothesis if event.label == "block"])

    per_class = score_events(reference_events, hypothesis_events)
    onset, offset = localization_error(reference_events, hypothesis_events)
    block_scores = score_events(block_ref, block_hyp, labels=("block",))
    probs = np.concatenate(prob_rows, axis=0)
    labels = np.concatenate(label_rows, axis=0)
    return {
        "per_class": {
            name: {
                "precision": round(score.precision, 4),
                "recall": round(score.recall, 4),
                "f1": round(score.f1, 4),
                "support": score.support,
            }
            for name, score in per_class.items()
        },
        "block_f1": round(block_scores["block"].f1, 4),
        "pause_false_positive_rate": round(float(np.mean(pause_rates)), 4),
        "onset_mae_sec": None if onset is None else round(onset, 4),
        "offset_mae_sec": None if offset is None else round(offset, 4),
        "clip_f1": round(binary_f1(np.array(clip_ref), np.array(clip_hyp)), 4),
        "calibration_error": round(expected_calibration_error(probs, labels), 4),
        "probabilities": probs,
        "labels": labels,
    }


def _latency_ms(model, example: Example, device: str, video: bool, text: bool) -> float:
    """Time the model forward only. Capture and drawing delay are measured in the browser."""

    torch_device = torch.device(device)
    batch = collate([example], torch_device)
    video_missing = torch.tensor([not video], dtype=torch.bool, device=torch_device)
    text_missing = torch.tensor([not text], dtype=torch.bool, device=torch_device)
    model.eval()

    def step() -> None:
        model(
            batch.audio,
            batch.visual,
            batch.text,
            video_missing=video_missing,
            text_missing=text_missing,
            audio_padding=batch.padding,
            visual_padding=batch.padding,
            text_padding=batch.padding,
        )

    with torch.no_grad():
        step()
        start = time.perf_counter()
        repeats = 6
        for _ in range(repeats):
            step()
    return (time.perf_counter() - start) * 1000 / repeats


def _fit_temperature(model: CompanionModel, examples: list[Example], device: str, video: bool, text: bool) -> float:
    logit_rows = []
    label_rows = []
    for example in examples:
        _probs, _pred, _clip, logits = predict_example(model, example, device=device, video=video, text=text)
        logit_rows.append(logits)
        label_rows.append(example.labels)
    temperature = fit_temperature(np.concatenate(logit_rows), np.concatenate(label_rows))
    with torch.no_grad():
        model.temperature.fill_(temperature)
    return temperature


def run_ablation(
    seed: int = 0,
    epochs: int = 5,
    clips_per_speaker: int = 20,
    device: str = "cpu",
    manifest: str | None = None,
    save_path: str | None = None,
) -> dict:
    if manifest:
        examples = load_manifest(manifest)
        speakers = sorted({example.speaker for example in examples})
        if len(speakers) < 2:
            train, test = examples, examples
            split_note = "A single speaker was provided, so train and test are the same list. This is not a generalization estimate."
        else:
            heldout = set(speakers[-max(1, len(speakers) // 5) :])
            train = [example for example in examples if example.speaker not in heldout]
            test = [example for example in examples if example.speaker in heldout]
            split_note = f"Held out speakers: {', '.join(sorted(heldout))}."
        origin = "local-manifest"
    else:
        train, test = make_split(clips_per_speaker=clips_per_speaker, seed=seed)
        split_note = "Synthetic speakers 0-7 are used for training and speakers 8-11 are held out."
        origin = "synthetic"

    config = ModelConfig(dropout=0.0 if origin == "synthetic" else 0.1)
    variants: list[tuple[str, torch.nn.Module]] = [
        ("tdnn-audio", AudioTDNN(config)),
        ("convlstm-audio", AudioConvLSTM(config)),
        ("companion-audio", CompanionModel(config)),
        ("companion-audio+video", CompanionModel(config)),
        ("companion-audio+video+text", CompanionModel(config)),
    ]
    # Masks during training match the variant so an audio-only model cannot
    # memorize visual channels that will be hidden at test time.
    reports = []
    for name, model in variants:
        video, text = MODALITY[name]
        losses = train_model(
            model,
            train,
            epochs=epochs,
            batch_size=16,
            device=device,
            seed=seed,
            drop_video=0.0 if video else 1.0,
            drop_text=0.0 if text else 1.0,
        )
        temperature = None
        if isinstance(model, CompanionModel) and train:
            temperature = _fit_temperature(model, train[:12], device, video, text)
        scored = _score_split(model, test, device, video=video, text=text, frame_sec=config.frame_sec)
        latency = _latency_ms(model, test[0], device, video, text)
        public = {key: value for key, value in scored.items() if key not in {"probabilities", "labels"}}
        public["train_loss"] = [round(value, 4) for value in losses]
        public["temperature"] = None if temperature is None else round(temperature, 4)
        public["model_forward_ms"] = round(latency, 2)
        public["uses_video"] = video
        public["uses_text"] = text
        reports.append({"model": name, **public})
        if save_path and name == "companion-audio+video+text":
            destination = Path(save_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state": model.state_dict(), "config": asdict(config), "data_origin": origin},
                destination,
            )

    with_video = next(item for item in reports if item["model"] == "companion-audio+video")
    full = next(item for item in reports if item["model"] == "companion-audio+video+text")
    audio_only = next(item for item in reports if item["model"] == "companion-audio")
    return {
        "disclaimer": (
            "This report scores the training harness. Synthetic results are a "
            "constructed check that visual features can separate silent blocks "
            "from pauses when the labels were generated that way. They are not "
            "MMSD-Net's published scores and not a clinical accuracy claim. "
            "A local manifest is limited by whatever that dataset annotated."
        ),
        "data_origin": origin,
        "split": split_note,
        "epochs": epochs,
        "clips_per_speaker": None if manifest else clips_per_speaker,
        "train_clips": len(train),
        "test_clips": len(test),
        "labels": list(CLASS_NAMES),
        "disfluency_labels": sorted(DISFLUENCY_LABELS),
        "variants": reports,
        "visual_block_f1_gain": round(with_video["block_f1"] - audio_only["block_f1"], 4),
        "text_restart_f1_gain": round(
            full["per_class"]["restart"]["f1"] - with_video["per_class"]["restart"]["f1"],
            4,
        ),
        "note_on_paper": (
            "MMSD-Net reported clip-level precision, recall, and F1 on a mix of "
            "SEP-28k, FluencyBank, and Adults Who Stutter. This harness reports "
            "event overlap at IoU 0.3, pause false positives, localization error, "
            "and latency. The tasks are not the same."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multimodal ablation harness")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--clips-per-speaker", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--manifest", default=None, help="JSONL of locally licensed features")
    parser.add_argument("--smoke", action="store_true", help="Tiny run used to check the pipeline")
    parser.add_argument("--out", default=None, help="Where to write the JSON report")
    parser.add_argument("--save", default=None, help="Optional path for the full multimodal checkpoint")
    args = parser.parse_args()
    epochs = 1 if args.smoke else args.epochs
    clips = 2 if args.smoke else args.clips_per_speaker
    report = run_ablation(
        seed=args.seed,
        epochs=epochs,
        clips_per_speaker=clips,
        device=args.device,
        manifest=args.manifest,
        save_path=args.save,
    )
    text = json.dumps(report, indent=2)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        print(f"Wrote {destination}")
    else:
        print(text)


if __name__ == "__main__":
    main()
