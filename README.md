# Alongside

Alongside is a speech companion for everyday conversation. It listens with the microphone, can follow the face with the camera, and keeps a transcript when the browser can make one. It is built to help someone understand their own speech. It is not a fluency score and it does not give corrective instructions when it notices a disfluency.

The design takes its cue from [MMSD-Net](https://arxiv.org/html/2407.11492v1): separate encoders for audio, face, and text, a compression step, and attention that puts those streams in one space. The live product is not a copy of that paper. MMSD-Net classifies a whole recording. This companion runs continuously and marks moments in time.

## What you can do

- Start a session from the camera and microphone. Audio is captured without echo cancellation, noise suppression, or automatic gain, so the measurement is closer to the signal in the room.
- Show or hide a facial mesh. Lip opening, jaw opening, and lip shape are drawn as traces. Lip shape is a geometric blendshape score, not muscle tension or emotion.
- Choose quiet listening, gentle timeline markers, or a reminder sentence you wrote. Reminders follow a timer. They are not triggered by a detection.
- Review a session afterward, with the recording when the browser was able to store it, and write a note if you want one.
- Read weekly notes that follow the goals you picked. Pauses are counted separately.

Speech recognition in the browser is provisional and often late. The companion keeps going without it. Filled pauses such as “um” are not treated as stuttering.

## Run the companion

From `frontend` in PowerShell:

```powershell
npm install
npm test
npm run dev
```

Open the URL Vite prints. Chrome or Edge can supply the live transcript. The face model and its runtime are downloaded from the network the first time a session starts. Audio, video, and notes stay in this browser.

## Research harness

From `backend`:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m companion.evaluate --out reports/ablation.json
```

That command trains small models on synthetic clips and writes an ablation. It takes a few minutes on CPU. The synthetic clips are a check that the code can use a visual channel to separate silent blocks from pauses when the labels were built that way. They are not a clinical result and they are not MMSD-Net’s published F1.

Wav2Vec 2.0 embeddings, for real 16 kHz recordings, are optional:

```powershell
python -m pip install -e ".[wav2vec]"
```

`companion.wav2vec.Wav2VecFrontend` returns frame embeddings at about 50 Hz. Point `--manifest` at a local JSONL file of features you are licensed to use. The loader does not download SEP-28k, FluencyBank, or any other corpus. See `docs/limitations.md`.

An optional research server loads a checkpoint only when you set `COMPANION_CHECKPOINT`. The live page does not need it.

```powershell
$env:COMPANION_CHECKPOINT = "checkpoints\model.pt"
python -m uvicorn companion.api:app --app-dir src --port 8000
```

A checkpoint trained on synthetic features is labeled as such and should not be read as a measurement of a person.

## Layout

- `frontend/` is the live companion: capture, mesh, on-device detector, review, and notes.
- `backend/src/companion/` is the multimodal model, audio baselines, metrics, and ablation.
- `docs/architecture.md` describes how the paper was adapted.
- `docs/limitations.md` describes datasets, annotations, and what the numbers do not mean.
