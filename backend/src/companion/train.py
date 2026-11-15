"""Training loop shared by the multimodal model and the audio baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from companion.config import CLASS_INDEX, ModelConfig
from companion.data import Example
from companion.model import CompanionModel, class_weights, loss_on_batch


@dataclass
class Batch:
    audio: torch.Tensor
    visual: torch.Tensor
    text: torch.Tensor
    labels: torch.Tensor
    clip_labels: torch.Tensor
    video_missing: torch.Tensor
    text_missing: torch.Tensor
    padding: torch.Tensor


def collate(examples: list[Example], device: torch.device, drop_video: float = 0.0, drop_text: float = 0.0, rng: np.random.Generator | None = None) -> Batch:
    length = max(example.labels.shape[0] for example in examples)
    audio_dim = examples[0].audio.shape[1]
    visual_dim = examples[0].visual.shape[1]
    text_dim = examples[0].text.shape[1]
    batch = len(examples)
    audio = torch.zeros(batch, length, audio_dim)
    visual = torch.zeros(batch, length, visual_dim)
    text = torch.zeros(batch, length, text_dim)
    labels = torch.zeros(batch, length, dtype=torch.long)
    padding = torch.ones(batch, length, dtype=torch.bool)
    video_missing = []
    text_missing = []
    clip_labels = []
    generator = rng or np.random.default_rng(0)
    for index, example in enumerate(examples):
        count = example.labels.shape[0]
        audio[index, :count] = torch.from_numpy(example.audio)
        visual[index, :count] = torch.from_numpy(example.visual)
        text[index, :count] = torch.from_numpy(example.text)
        labels[index, :count] = torch.from_numpy(example.labels.astype(np.int64))
        padding[index, :count] = False
        hide_video = example.video_missing or generator.random() < drop_video
        hide_text = example.text_missing or generator.random() < drop_text
        video_missing.append(hide_video)
        text_missing.append(hide_text)
        clip_labels.append(example.clip_label)
    return Batch(
        audio=audio.to(device),
        visual=visual.to(device),
        text=text.to(device),
        labels=labels.to(device),
        clip_labels=torch.tensor(clip_labels, dtype=torch.float32, device=device),
        video_missing=torch.tensor(video_missing, dtype=torch.bool, device=device),
        text_missing=torch.tensor(text_missing, dtype=torch.bool, device=device),
        padding=padding.to(device),
    )


def iterate_batches(examples: list[Example], batch_size: int, seed: int):
    # Clips that contain a non-fluent stretch are repeated so a batch is not
    # almost entirely fluent frames.
    boosted: list[Example] = []
    for example in examples:
        boosted.append(example)
        if np.any(example.labels != 0):
            boosted.append(example)
    order = np.arange(len(boosted))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chosen = [boosted[int(index)] for index in order[start : start + batch_size]]
        yield chosen


def frame_class_weights(examples: list[Example], device: torch.device) -> torch.Tensor:
    counts = np.ones(6, dtype=np.float64)
    for example in examples:
        counts += np.bincount(example.labels, minlength=6)
    return class_weights(device, torch.tensor(counts, dtype=torch.float32, device=device))


def train_model(  # noqa: PLR0913 — training entry point, callers pass this explicitly
    model: nn.Module,
    examples: list[Example],
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    drop_video: float = 0.2,
    drop_text: float = 0.35,
) -> list[float]:
    torch.manual_seed(seed)
    torch_device = torch.device(device)
    model.to(torch_device)
    weights = frame_class_weights(examples, torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for group in iterate_batches(examples, batch_size, seed + epoch):
            batch = collate(group, torch_device, drop_video=drop_video, drop_text=drop_text)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch.audio,
                batch.visual,
                batch.text,
                video_missing=batch.video_missing,
                text_missing=batch.text_missing,
                audio_padding=batch.padding,
                visual_padding=batch.padding,
                text_padding=batch.padding,
            )
            loss = loss_on_batch(output, batch.labels, batch.padding, batch.clip_labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses) if epoch_losses else 0.0))
    return losses


def predict_example(
    model: nn.Module,
    example: Example,
    device: str = "cpu",
    video: bool = True,
    text: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return probabilities, frame labels, clip probability, and raw logits."""

    torch_device = torch.device(device)
    model.eval()
    batch = collate([example], torch_device)
    video_missing = torch.tensor([not video or example.video_missing], dtype=torch.bool, device=torch_device)
    text_missing = torch.tensor([not text or example.text_missing], dtype=torch.bool, device=torch_device)
    with torch.no_grad():
        output = model(
            batch.audio,
            batch.visual,
            batch.text,
            video_missing=video_missing,
            text_missing=text_missing,
            audio_padding=batch.padding,
            visual_padding=batch.padding,
            text_padding=batch.padding,
        )
    logits = output.frame_logits[0, : example.labels.shape[0]].detach().cpu().numpy()
    temperature = 1.0
    if isinstance(model, CompanionModel):
        temperature = float(model.temperature.detach().clamp(min=0.05))
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = probs.argmax(axis=1).astype(np.int64)
    clip = float(torch.sigmoid(output.clip_logit)[0].cpu())
    return probs, labels, clip, logits


def disfluency_frame_count(labels: np.ndarray) -> int:
    ids = {
        CLASS_INDEX["repetition"],
        CLASS_INDEX["prolongation"],
        CLASS_INDEX["block"],
        CLASS_INDEX["restart"],
    }
    return int(sum(int(label) in ids for label in labels))
