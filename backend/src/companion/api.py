"""Optional research server.

Live sessions do not need this process. The browser keeps audio, video, and
transcripts on device. This API exposes health and, when a checkpoint is
loaded, windowed research inference on the feature schema that checkpoint
was trained with.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from companion import __version__
from companion.config import CLASS_NAMES, ModelConfig
from companion.events import decode_frame_labels
from companion.model import CompanionModel
from companion.streaming import OverlapSmoother, SlidingBuffer, SyncedFrame

app = FastAPI(title="Speech companion research API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: CompanionModel | None = None
_origin = "unloaded"
_buffer = SlidingBuffer()
_smoother = OverlapSmoother()


class FrameIn(BaseModel):
    time: float
    audio: list[float]
    visual: list[float] | None = None
    text: list[float] | None = None


class WindowIn(BaseModel):
    frames: list[FrameIn] = Field(min_length=1)
    reset: bool = False


def _load_from_env() -> None:
    global _model, _origin
    raw = os.environ.get("COMPANION_CHECKPOINT", "").strip()
    if not raw:
        return
    path = Path(raw)
    if not path.exists():
        raise RuntimeError(f"COMPANION_CHECKPOINT does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["config"])
    model = CompanionModel(config)
    model.load_state_dict(payload["state"])
    model.eval()
    _model = model
    _origin = str(payload.get("data_origin", "unknown"))


@app.on_event("startup")
def startup() -> None:
    if os.environ.get("COMPANION_CHECKPOINT"):
        _load_from_env()


@app.get("/v1/health")
def health() -> dict:
    return {
        "version": __version__,
        "live_path": "on-device",
        "research_model_loaded": _model is not None,
        "checkpoint_origin": _origin,
        "labels": list(CLASS_NAMES),
        "note": (
            "The live companion detects events in the browser. A checkpoint "
            "trained on synthetic features must not be read as a measurement "
            "of a person's speech."
        ),
    }


@app.post("/v1/research/infer")
def research_infer(body: WindowIn) -> dict:
    if _model is None:
        raise HTTPException(
            status_code=409,
            detail="No research checkpoint is loaded. The live detector stays in the browser.",
        )
    global _buffer, _smoother
    if body.reset:
        _buffer = SlidingBuffer(frame_sec=_model.config.frame_sec)
        _smoother = OverlapSmoother(frame_sec=_model.config.frame_sec, n_classes=_model.config.n_classes)
    for frame in body.frames:
        _buffer.push(
            SyncedFrame(
                time=frame.time,
                audio=np.asarray(frame.audio, dtype=np.float32),
                visual=None if frame.visual is None else np.asarray(frame.visual, dtype=np.float32),
                text=None if frame.text is None else np.asarray(frame.text, dtype=np.float32),
            )
        )
    if not _buffer.frames:
        return {"events": [], "checkpoint_origin": _origin}
    window = _buffer.window()
    audio = torch.from_numpy(window.audio).unsqueeze(0)
    visual = torch.from_numpy(window.visual).unsqueeze(0)
    text = torch.from_numpy(window.text).unsqueeze(0)
    video_missing = torch.tensor([not bool(window.video_present.any())])
    text_missing = torch.tensor([not bool(window.text_present.any())])
    with torch.no_grad():
        output = _model(audio, visual, text, video_missing=video_missing, text_missing=text_missing)
    logits = output.frame_logits[0].cpu().numpy()
    _smoother.add(window.times, logits)
    now = float(window.times[-1])
    finalize_before = now - _buffer.hop_sec
    rows = _smoother.labels_at(now, finalize_before)
    labels = np.array([row[1] for row in rows], dtype=np.int64)
    times = np.array([row[0] for row in rows], dtype=np.float32)
    confidences = np.array([row[3] for row in rows], dtype=np.float32)
    # decode_frame_labels assumes a uniform grid starting at 0. Map back after.
    events = decode_frame_labels(labels, frame_sec=_buffer.frame_sec, confidences=confidences, provisional_after=finalize_before)
    offset = float(times[0]) if len(times) else 0.0
    payload = [
        {
            "start": round(offset + event.start, 3),
            "end": round(offset + event.end, 3),
            "label": event.label,
            "confidence": round(event.confidence, 3),
            "state": "provisional" if event.end + offset >= finalize_before else "final",
        }
        for event in events
    ]
    _buffer.mark_emitted(finalize_before)
    return {
        "events": payload,
        "checkpoint_origin": _origin,
        "warning": (
            None
            if _origin == "local-manifest"
            else "This checkpoint does not describe real-world accuracy unless it was trained on licensed, synchronized annotations."
        ),
    }
