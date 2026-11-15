"""MMSD-Net-inspired encoders, compression, fusion, and event head.

The paper compresses each modality with Conv1D and a linear map, fuses the
soft tokens with Q-Former-style attention, and classifies a whole recording.
This model keeps that structure and adds a frame-level head so a window can
produce timestamped events. A learned null token stands in for missing video
or delayed text so attention stays defined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from companion.config import CLASS_NAMES, ModelConfig


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4000) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:length]


class ModalityEncoder(nn.Module):
    """Transformer encoder plus the paper's Conv1D compression.

    Compression here keeps the frame rate (stride 1) so events can be timed.
    A pooled clip head still sees a short query summary, which is the role
    fixed-length soft tokens play in MMSD-Net.
    """

    def __init__(self, d_in: int, config: ModelConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, config.d_model)
        self.pos = SinusoidalPositionalEncoding(config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers, enable_nested_tensor=False)
        self.compress = nn.Sequential(
            nn.Conv1d(config.d_model, config.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.d_model, config.d_model, kernel_size=3, padding=1),
        )
        self.causal = config.causal

    def forward(self, x: torch.Tensor, key_padding: torch.Tensor | None) -> torch.Tensor:
        length = x.size(1)
        hidden = self.proj(x) + self.pos(length)
        attn_mask = None
        if self.causal and length > 1:
            attn_mask = torch.triu(
                torch.ones(length, length, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        hidden = self.encoder(hidden, mask=attn_mask, src_key_padding_mask=key_padding)
        compressed = self.compress(hidden.transpose(1, 2)).transpose(1, 2)
        return compressed


def cross_attend(
    attn: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    key_padding: torch.Tensor | None,
    null_token: torch.Tensor,
    modality_off: torch.Tensor | None,
    causal: bool,
) -> torch.Tensor:
    """Cross-attention with a null key for a missing or fully padded modality."""

    batch, query_len, _ = query.shape
    _, key_len, dim = key.shape
    keys = torch.cat([key, null_token.expand(batch, 1, dim)], dim=1)
    if key_padding is None:
        padding = torch.zeros(batch, key_len, dtype=torch.bool, device=key.device)
    else:
        padding = key_padding
    if modality_off is not None:
        padding = padding | modality_off.unsqueeze(1)
    any_real = (~padding).any(dim=1, keepdim=True)
    null_padding = any_real
    padding = torch.cat([padding, null_padding], dim=1)
    none_visible = padding.all(dim=1)
    if bool(none_visible.any()):
        padding = padding.clone()
        padding[none_visible, -1] = False

    attn_mask = None
    if causal and query_len > 1:
        attn_mask = torch.zeros(query_len, key_len + 1, dtype=torch.bool, device=key.device)
        causal_real = torch.triu(
            torch.ones(query_len, key_len, dtype=torch.bool, device=key.device),
            diagonal=1,
        )
        usable = min(query_len, key_len)
        attn_mask[:usable, :usable] = causal_real[:usable, :usable]
    context, _ = attn(query, keys, keys, key_padding_mask=padding, attn_mask=attn_mask, need_weights=False)
    return context


class QueryFusion(nn.Module):
    """Fuse modalities with learnable queries and audio-led cross-attention.

    Global queries follow the Q-Former idea: a small set of learned queries
    reads every modality and is pooled for a recording-level decision.
    Frame queries come from the audio stream so each timestamp keeps an
    aligned fused vector for event detection.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.d_model
        heads = config.n_heads
        self.causal = config.causal
        self.global_queries = nn.Parameter(torch.randn(1, config.n_global_queries, dim) * 0.02)
        self.audio_to_visual = nn.MultiheadAttention(dim, heads, dropout=config.dropout, batch_first=True)
        self.audio_to_text = nn.MultiheadAttention(dim, heads, dropout=config.dropout, batch_first=True)
        self.global_attn = nn.MultiheadAttention(dim, heads, dropout=config.dropout, batch_first=True)
        self.null_visual = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.null_text = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.null_global = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.mixer = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.clip_head = nn.Linear(dim, 1)

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
        audio_padding: torch.Tensor | None,
        visual_padding: torch.Tensor | None,
        text_padding: torch.Tensor | None,
        video_missing: torch.Tensor | None,
        text_missing: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual_context = cross_attend(
            self.audio_to_visual,
            audio,
            visual,
            visual_padding,
            self.null_visual,
            video_missing,
            self.causal,
        )
        text_context = cross_attend(
            self.audio_to_text,
            audio,
            text,
            text_padding,
            self.null_text,
            text_missing,
            self.causal,
        )
        fused = self.mixer(torch.cat([audio, visual_context, text_context], dim=-1))

        batch = audio.size(0)
        queries = self.global_queries.expand(batch, -1, -1)
        memory = torch.cat([audio, visual, text], dim=1)
        memory_padding = _concat_padding(
            audio,
            visual,
            text,
            audio_padding,
            visual_padding,
            text_padding,
            video_missing,
            text_missing,
        )
        summary = cross_attend(
            self.global_attn,
            queries,
            memory,
            memory_padding,
            self.null_global,
            modality_off=None,
            causal=False,
        )
        clip_logit = self.clip_head(summary.mean(dim=1)).squeeze(-1)
        return fused, clip_logit


def _concat_padding(
    audio: torch.Tensor,
    visual: torch.Tensor,
    text: torch.Tensor,
    audio_padding: torch.Tensor | None,
    visual_padding: torch.Tensor | None,
    text_padding: torch.Tensor | None,
    video_missing: torch.Tensor | None,
    text_missing: torch.Tensor | None,
) -> torch.Tensor:
    batch = audio.size(0)
    device = audio.device
    if audio_padding is None:
        audio_padding = torch.zeros(batch, audio.size(1), dtype=torch.bool, device=device)
    if visual_padding is None:
        visual_padding = torch.zeros(batch, visual.size(1), dtype=torch.bool, device=device)
    if text_padding is None:
        text_padding = torch.zeros(batch, text.size(1), dtype=torch.bool, device=device)
    if video_missing is not None:
        visual_padding = visual_padding | video_missing.unsqueeze(1)
    if text_missing is not None:
        text_padding = text_padding | text_missing.unsqueeze(1)
    return torch.cat([audio_padding, visual_padding, text_padding], dim=1)


@dataclass
class CompanionOutput:
    frame_logits: torch.Tensor
    clip_logit: torch.Tensor


class CompanionModel(nn.Module):
    """Three encoders, query fusion, temporal classes, and a clip-level head."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.audio_encoder = ModalityEncoder(self.config.audio_dim, self.config)
        self.visual_encoder = ModalityEncoder(self.config.visual_dim, self.config)
        self.text_encoder = ModalityEncoder(self.config.text_dim, self.config)
        self.fusion = QueryFusion(self.config)
        self.event_head = nn.Linear(self.config.d_model, self.config.n_classes)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
        video_missing: torch.Tensor | None = None,
        text_missing: torch.Tensor | None = None,
        audio_padding: torch.Tensor | None = None,
        visual_padding: torch.Tensor | None = None,
        text_padding: torch.Tensor | None = None,
    ) -> CompanionOutput:
        audio_h = self.audio_encoder(audio, audio_padding)
        visual_h = self.visual_encoder(visual, visual_padding)
        text_h = self.text_encoder(text, text_padding)
        if video_missing is not None:
            visual_h = visual_h.masked_fill(video_missing[:, None, None], 0.0)
        if text_missing is not None:
            text_h = text_h.masked_fill(text_missing[:, None, None], 0.0)
        fused, clip_logit = self.fusion(
            audio_h,
            visual_h,
            text_h,
            audio_padding,
            visual_padding,
            text_padding,
            video_missing,
            text_missing,
        )
        # Temperature is fit after training and applied by the caller so the
        # loss sees unscaled logits.
        frame_logits = self.event_head(fused)
        return CompanionOutput(frame_logits=frame_logits, clip_logit=clip_logit)


def class_weights(device: torch.device, counts: torch.Tensor | None = None) -> torch.Tensor:
    """Inverse-frequency weights when frame counts are known, otherwise a fixed prior."""

    if counts is None:
        weights = torch.tensor([0.35, 2.5, 2.5, 3.0, 2.5, 1.6], dtype=torch.float32, device=device)
    else:
        safe = counts.float().clamp(min=1)
        # Square-root inverse frequency. Full inverse frequency makes the
        # model call the rare classes on almost every frame.
        weights = torch.sqrt(safe.sum() / safe)
        weights = weights / weights.mean()
    if weights.numel() != len(CLASS_NAMES):
        raise RuntimeError("class weight length drifted from CLASS_NAMES")
    return weights


def loss_on_batch(
    output: CompanionOutput,
    frame_labels: torch.Tensor,
    frame_padding: torch.Tensor | None,
    clip_labels: torch.Tensor,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Frame cross-entropy plus a clip-level binary term.

    Clip labels are 1 when the window contains a disfluency other than a
    natural pause. That mirrors the paper's recording-level decision without
    replacing the event head.
    """

    logits = output.frame_logits
    batch, length, classes = logits.shape
    flat_logits = logits.reshape(batch * length, classes)
    flat_labels = frame_labels.reshape(batch * length)
    weight = class_weight if class_weight is not None else class_weights(logits.device)
    if frame_padding is None:
        frame_loss = nn.functional.cross_entropy(flat_logits, flat_labels, weight=weight)
    else:
        valid = ~frame_padding.reshape(batch * length)
        if bool(valid.any()):
            frame_loss = nn.functional.cross_entropy(flat_logits[valid], flat_labels[valid], weight=weight)
        else:
            frame_loss = logits.sum() * 0.0
    clip_loss = nn.functional.binary_cross_entropy_with_logits(output.clip_logit, clip_labels.float())
    return frame_loss + 0.3 * clip_loss
