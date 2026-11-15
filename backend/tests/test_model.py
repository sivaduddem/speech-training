import torch

from companion.config import ModelConfig
from companion.model import CompanionModel


def _batch(batch: int = 2, length: int = 16):
    config = ModelConfig(d_model=32, n_heads=4, n_layers=1, n_global_queries=2, dropout=0.0)
    model = CompanionModel(config)
    audio = torch.randn(batch, length, config.audio_dim)
    visual = torch.randn(batch, length, config.visual_dim)
    text = torch.randn(batch, length, config.text_dim)
    return model, audio, visual, text


def test_forward_shapes_are_finite():
    model, audio, visual, text = _batch()
    output = model(audio, visual, text, None, None, None, None, None)
    assert output.frame_logits.shape == (2, 16, 6)
    assert output.clip_logit.shape == (2,)
    assert torch.isfinite(output.frame_logits).all()
    assert torch.isfinite(output.clip_logit).all()


def test_missing_video_and_text_stay_finite():
    model, audio, visual, text = _batch()
    video_missing = torch.tensor([True, True])
    text_missing = torch.tensor([True, False])
    padding = torch.zeros(2, 16, dtype=torch.bool)
    padding[:, -3:] = True
    output = model(
        audio,
        visual,
        text,
        video_missing=video_missing,
        text_missing=text_missing,
        audio_padding=padding,
        visual_padding=padding,
        text_padding=padding,
    )
    assert torch.isfinite(output.frame_logits).all()
    assert torch.isfinite(output.clip_logit).all()


def test_single_frame_window():
    model, audio, visual, text = _batch(batch=1, length=1)
    output = model(audio, visual, text, None, None, None, None, None)
    assert output.frame_logits.shape == (1, 1, 6)
    assert torch.isfinite(output.frame_logits).all()
