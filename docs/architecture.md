# Architecture

MMSD-Net (Nie, Kadiri, and Agrawal, 2024) is a clip classifier. It runs a transformer encoder on video, audio, and text, shortens each sequence with a 1-D convolution and a linear map, fuses those soft tokens with attention in the style of a Q-Former, and makes one binary decision for the recording.

Alongside keeps the parts that transfer to a live companion and changes the decision.

## What is carried over

- Each modality has its own encoder. Audio does not have to share weights with the face or the transcript.
- A short stack of 1-D convolutions compresses each encoder output. In the paper this also shortens the sequence to a handful of tokens. Here the frame rate stays put, because a moment needs a time.
- Learned queries read the other modalities with cross-attention. A null token is the only visible key when video is missing or the transcript has not arrived, so attention stays defined.
- A second, global set of queries is pooled for a recording-level yes-or-no head. That head is the paper-style baseline. It is not what the live screen shows.

## What is different

- The main head labels every frame as fluent, repetition, prolongation, block, restart, or pause. Frames become events with a start, an end, a confidence, and a provisional or final state.
- Pauses and filled pauses are not positive stuttering classes. The paper argues that ordinary pauses and interjections are not the same thing as stuttering. The live detector follows that distinction. Restarts are kept because a revision is one of the patterns the transcript can actually show.
- Windows used for research inference can look at the whole short window, as the paper looks at a whole clip. Only frames older than the hop are finalized. The newest edge stays provisional.
- Overlapping windows average their logits in the same time bin before the label is chosen.
- A scalar temperature is fit on training logits and applied at scoring time. The live detector uses a simpler cap: confidence is never shown above 0.92, because this on-device model is not a calibrated clinical probability.
- The live path does not wait for Wav2Vec 2.0. It uses loudness, zero-crossing rate, spectral centroid, spectral flux, low-band energy, and the autocorrelation of the loudness envelope, plus facial measurements and whatever transcript text has arrived. Wav2Vec 2.0 remains the embedding frontend for 16 kHz audio when a training run needs it. Those embeddings are about 50 Hz, which matches the frame clock.

## Live clock

Microphone chunks and face frames share the audio clock. Face frames that are more than about a third of a second away from the current sound are treated as missing, and the video mask turns off. A transcript result is applied when it arrives, including to a moment that has already passed. That late cue is marked as approximate timing.

The face mesh is drawn from MediaPipe Face Landmarker landmarks, blendshapes, and the facial transformation matrix. Lip and jaw measurements used for decisions are scaled by the distance between the eyes, and lip motion is estimated after a similarity transform that removes head translation, scale, and roll. Head pose is kept as its own channel so a head turn is not mistaken for mouth movement.

## Feedback

Quiet mode records and does not mark the timeline. Gentle mode draws marks and does not speak. Reminder mode shows a sentence the person wrote, on an interval they chose. Ending a session stores events, a downsampled feature log, the transcript, and the recording when storage allows. Review and the weekly notes are descriptive. They do not set a target to reduce.

## Evaluation

The harness trains three audio references (a dilated convolution stack, a convolution plus LSTM, and the multimodal model with video and text hidden) and two multimodal variants. It reports event precision, recall, and F1 at an overlap of 0.3, block F1, the share of true pause frames labeled as a disfluency, onset and offset error, clip-level F1, calibration error, and model forward time. Held-out speakers are used when the data has more than one speaker.

See `docs/limitations.md` before reading those numbers as anything but a property of the data that was actually trained on.
