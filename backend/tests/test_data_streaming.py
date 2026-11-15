import numpy as np

from companion.config import CLASS_INDEX
from companion.data import load_manifest, make_example, make_split
from companion.streaming import OverlapSmoother, SlidingBuffer, SyncedFrame
from companion.wav2vec import Wav2VecFrontend, wav2vec_available


def test_synthetic_blocks_share_silence_with_pauses_and_differ_visually():
    rng = np.random.default_rng(2)
    blocks = []
    pauses = []
    for _ in range(300):
        example = make_example(rng, "speaker", 0.0)
        if np.any(example.labels == CLASS_INDEX["block"]):
            blocks.append(example)
        if np.any(example.labels == CLASS_INDEX["pause"]):
            pauses.append(example)
    assert len(blocks) > 10 and len(pauses) > 10

    def mean_channel(examples, modality: str, channel: int, label: int) -> float:
        values = []
        for example in examples:
            mask = example.labels == label
            values.append(float(getattr(example, modality)[mask, channel].mean()))
        return float(np.mean(values))

    block_silence = mean_channel(blocks, "audio", 2, CLASS_INDEX["block"])
    pause_silence = mean_channel(pauses, "audio", 2, CLASS_INDEX["pause"])
    assert abs(block_silence - pause_silence) < 0.35
    assert mean_channel(blocks, "visual", 1, CLASS_INDEX["block"]) > 1.2
    assert mean_channel(pauses, "visual", 2, CLASS_INDEX["pause"]) > 1.2


def test_split_holds_out_speakers():
    train, test = make_split(n_train_speakers=4, n_test_speakers=2, clips_per_speaker=2, seed=1)
    train_speakers = {example.speaker for example in train}
    test_speakers = {example.speaker for example in test}
    assert train_speakers.isdisjoint(test_speakers)


def test_manifest_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    example = make_example(rng, "ada", 0.1, length=20)
    np.save(tmp_path / "a.npy", example.audio)
    np.save(tmp_path / "y.npy", example.labels)
    (tmp_path / "clips.jsonl").write_text(
        '{"speaker": "ada", "audio": "a.npy", "visual": null, "text": null, "labels": "y.npy"}\n',
        encoding="utf-8",
    )
    loaded = load_manifest(tmp_path / "clips.jsonl")
    assert loaded[0].video_missing and loaded[0].text_missing
    assert loaded[0].audio.shape == example.audio.shape


def test_sliding_window_marks_the_tail_provisional():
    buffer = SlidingBuffer(window_sec=1.0, hop_sec=0.4, frame_sec=0.05)
    smoother = OverlapSmoother(frame_sec=0.05, n_classes=6)
    for index in range(30):
        buffer.push(
            SyncedFrame(
                time=index * 0.05,
                audio=np.ones(8, dtype=np.float32),
                visual=None,
                text=None,
            )
        )
    assert buffer.ready()
    window = buffer.window(visual_dim=6, text_dim=4)
    assert window.video_present.sum() == 0
    logits = np.zeros((len(window.times), 6), dtype=np.float32)
    logits[:, 0] = 2
    smoother.add(window.times, logits)
    now = float(window.times[-1])
    rows = smoother.labels_at(now, finalize_before=now - 0.4)
    assert rows[-1][2] == "provisional"
    assert any(row[2] == "final" for row in rows)


def test_wav2vec_frontend_is_optional():
    if wav2vec_available():
        return
    try:
        Wav2VecFrontend()
    except RuntimeError as exc:
        assert "wav2vec" in str(exc).lower()
    else:
        raise AssertionError("missing transformers should not construct a frontend")
