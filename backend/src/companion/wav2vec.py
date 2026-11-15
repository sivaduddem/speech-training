"""Optional Wav2Vec 2.0 frame embeddings.

MMSD-Net-style audio encoders expect a sequence, not a clip embedding.
Wav2Vec 2.0's last hidden state is about 50 Hz, which matches the live
frame clock. Loading it is optional so the rest of the harness runs without
the checkpoint download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def wav2vec_available() -> bool:
    try:
        import transformers  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError:
        return False
    return True


class Wav2VecFrontend:
    """Frozen Wav2Vec 2.0 encoder. Call `embed_file` or `embed_waveforms`."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base", device: str = "cpu") -> None:
        if not wav2vec_available():
            raise RuntimeError(
                'Wav2Vec support is not installed. Use pip install -e ".[wav2vec]" inside backend/.'
            )
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        self.torch = torch
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.hidden_size = int(self.model.config.hidden_size)

    def embed_waveforms(self, waveforms: list[np.ndarray], sample_rate: int = 16000) -> list[np.ndarray]:
        if sample_rate != 16000:
            raise ValueError("Wav2Vec 2.0 frontend expects 16 kHz audio")
        torch = self.torch
        inputs = self.processor(
            [wave.astype(np.float32) for wave in waveforms],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            hidden = self.model(input_values, attention_mask=attention).last_hidden_state
        lengths = attention.sum(dim=1)
        # The feature encoder downsamples by 320 samples.
        frame_lengths = torch.div(lengths - 1, 320, rounding_mode="floor") + 1
        sequences: list[np.ndarray] = []
        for index in range(hidden.size(0)):
            count = int(frame_lengths[index].item())
            count = max(1, min(count, hidden.size(1)))
            sequences.append(hidden[index, :count].cpu().numpy().astype(np.float32))
        return sequences

    def embed_file(self, path: str | Path) -> np.ndarray:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            raise ValueError(f"{path} is {sample_rate} Hz; resample to 16 kHz first")
        return self.embed_waveforms([audio.astype(np.float32)])[0]
