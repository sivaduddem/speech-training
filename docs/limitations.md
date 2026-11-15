# Limitations

## The paper is not a performance claim for this system

MMSD-Net reports clip-level precision, recall, and F1 on a mixture of SEP-28k, FluencyBank, and Adults Who Stutter. The best line in that table is about 90 F1, roughly 2 points above their strongest audio baseline and about 16 points above the weakest. Those figures are for a binary decision on a whole clip.

This project does not reproduce that table. The published paper does not include per-type scores, a silent-block split, a false-positive rate on natural pauses, localization error, or latency. The authors note that a qualitative look at different kinds of stuttering was future work. A binary gain from “adding video” does not by itself say that video helps silent blocks.

## Data that is not in this repository

- SEP-28k is podcast audio. Labels are clip-level and can mark more than one behavior in the same three-second clip. There is no face track, and the labels are not frame-accurate event boundaries.
- FluencyBank and the Adults Who Stutter recordings are the audiovisual source MMSD-Net relies on. They are distributed under an access agreement. They are not included here and are not downloaded by the loader.
- SpeakingFaces, which the paper uses as fluent audiovisual speech, is a command dataset, not a stuttering corpus.
- A local manifest can point at feature arrays you are allowed to use. The quality of any score is then limited by that annotation: who labeled it, whether boundaries exist, and whether the camera and microphone were synchronized.

The synthetic ablation builds blocks and pauses with the same weak silence cue and different visual channels. If the audio-plus-video model separates them and the audio-only model does not, that shows the training code is using the visual channel. It does not show that the same thing happens for a real speaker.

## The live detector

The on-device detector is interpretable and conservative on purpose. Silence alone does not become a block. A block mark requires a held mouth shape while the sound has dropped after recent speech. Without a camera, those moments stay with pauses. Repetition uses rhythm in the loudness envelope and, when a transcript arrives, repeated words or word parts. A steady voiced sound can be marked as a prolongation, which will sometimes include an ordinary long vowel.

Confidence shown in the interface is capped. It is not a clinical probability. Visible delay is the time until a mark appears on this machine, and it is not the paper’s accuracy.

Streaming speech recognition often has no word times. Linguistic marks are spread across the utterance that produced them and labeled approximate. The repetition rules work best when the transcript separates words with spaces. They do not treat “um” or “uh” as events.

Facial landmarks and blendshapes describe shape and movement. They do not measure muscle tension, effort, or emotion, and the interface does not present them that way.

Microphone processing that removes echo or noise can also remove the very changes this tool is trying to see, so those browser filters are left off. Room noise, a far microphone, and a face turned away from the camera all change the counts. Session totals are not comparable across rooms as if they were a stable personal score.

The recording clock and the feature clock can drift by a fraction of a second. Review seeks to the feature time, which is the time the detector used.

## What the companion will not do

It will not diagnose stuttering. It will not tell someone to stop, to be fluent, or to eliminate a moment it marked. A reminder, if one is used, is a sentence the person wrote and a timer they chose.
