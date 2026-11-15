"""Simplified audio baselines for the same event-label harness.

These are small reference models inspired by published audio-only stuttering
detectors (a TDNN in the spirit of StutterNet, and a ConvLSTM). They are not
reimplementations of those systems and are not expected to match published
scores.
"""

from __future__ import annotations

import torch
from torch import nn

from companion.config import ModelConfig
from companion.model import CompanionOutput


class _AudioFrameModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.clip_head = nn.Linear(config.d_model, 1)
        self.frame_head = nn.Linear(config.d_model, config.n_classes)

    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor | None = None,
        text: torch.Tensor | None = None,
        video_missing: torch.Tensor | None = None,
        text_missing: torch.Tensor | None = None,
        audio_padding: torch.Tensor | None = None,
        visual_padding: torch.Tensor | None = None,
        text_padding: torch.Tensor | None = None,
    ) -> CompanionOutput:
        del visual, text, video_missing, text_missing, visual_padding, text_padding
        hidden = self._encode(audio)
        if audio_padding is not None:
            hidden = hidden.masked_fill(audio_padding.unsqueeze(-1), 0.0)
            denom = (~audio_padding).sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = hidden.sum(dim=1) / denom
        else:
            pooled = hidden.mean(dim=1)
        return CompanionOutput(
            frame_logits=self.frame_head(hidden),
            clip_logit=self.clip_head(pooled).squeeze(-1),
        )


class AudioTDNN(_AudioFrameModel):
    """Dilated temporal convolution stack on audio features."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        dim = config.d_model
        self.net = nn.Sequential(
            nn.Conv1d(config.audio_dim, dim, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        )

    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self.net(audio.transpose(1, 2)).transpose(1, 2)


class AudioConvLSTM(_AudioFrameModel):
    """A short convolution followed by a bidirectional LSTM."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        dim = config.d_model
        self.conv = nn.Sequential(
            nn.Conv1d(config.audio_dim, dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(dim, dim // 2, num_layers=1, batch_first=True, bidirectional=True)

    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        local = self.conv(audio.transpose(1, 2)).transpose(1, 2)
        encoded, _ = self.lstm(local)
        return encoded
