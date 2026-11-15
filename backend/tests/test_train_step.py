import numpy as np
import torch

from companion.config import ModelConfig
from companion.data import make_split
from companion.model import CompanionModel
from companion.train import train_model


def test_one_epoch_returns_finite_loss():
    train, _test = make_split(n_train_speakers=2, n_test_speakers=1, clips_per_speaker=4, seed=3)
    model = CompanionModel(ModelConfig(d_model=32, n_heads=4, n_layers=1, n_global_queries=2, dropout=0.0))
    losses = train_model(model, train, epochs=1, batch_size=4, device="cpu", seed=3, drop_video=1.0, drop_text=1.0)
    assert len(losses) == 1
    assert np.isfinite(losses[0])
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
