"""Shared labels and feature layouts.

The live browser pipeline and this package use different feature layouts on
purpose. Synthetic channels exist only to test training and ablation code.
Live channels describe measurements a session can actually log.
"""

from __future__ import annotations

from dataclasses import dataclass

# Frame labels. Pause and fluent are not counted as stuttering observations.
CLASS_NAMES: tuple[str, ...] = (
    "fluent",
    "repetition",
    "prolongation",
    "block",
    "restart",
    "pause",
)
DISFLUENCY_LABELS: frozenset[str] = frozenset(
    {"repetition", "prolongation", "block", "restart"}
)
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

# Live session export (frontend/src/lib/researchSchema.ts keeps the same order).
LIVE_AUDIO_CHANNELS: tuple[str, ...] = (
    "log_rms",
    "zcr",
    "centroid",
    "flux",
    "low_band",
    "autocorr",
    "voiced",
    "delta_rms",
)
LIVE_VISUAL_CHANNELS: tuple[str, ...] = (
    "lip_aperture",
    "jaw_open",
    "lip_compression_shape",
    "mouth_width",
    "lip_velocity",
    "landmark_motion",
    "pose_speed",
    "video_present",
)
LIVE_TEXT_CHANNELS: tuple[str, ...] = (
    "repetition_cue",
    "restart_cue",
    "prolongation_cue",
    "text_present",
)


@dataclass(frozen=True)
class ModelConfig:
    """Sizes for the multimodal event model.

    Defaults are small enough to train on CPU as a harness. They are not the
    MMSD-Net training setup (LoRA, 512 tokens, four A100s).
    """

    audio_dim: int = 8
    visual_dim: int = 6
    text_dim: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    n_global_queries: int = 4
    n_classes: int = len(CLASS_NAMES)
    # Bidirectional windows match the paper's full-clip encoders.
    # Streaming inference commits only frames older than the hop so the
    # newest edge stays provisional.
    causal: bool = False
    frame_sec: float = 0.05

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_classes != len(CLASS_NAMES):
            raise ValueError("n_classes must match CLASS_NAMES")
