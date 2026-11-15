import { drawCharts } from "./lib/charts";
import { applyDelayedCue, emptyDetector, stepDetector, type DetState } from "./lib/detector";
import {
  envelopeAutocorr,
  lowBandRatio,
  magnitudeSpectrum,
  rms,
  spectralCentroid,
  spectralFlux,
  voicedScore,
  zeroCrossingRate,
} from "./lib/audioFeatures";
import {
  describeCoupling,
  jawOpening,
  lipAperture,
  lipCompressionShape,
  measureLipMotion,
  mouthWidth,
  poseFromMatrix,
  type MotionState,
  type Pt,
} from "./lib/faceFeatures";
import { liveMessage } from "./lib/feedback";
import { findLinguisticEvents, type LinguisticCue } from "./lib/linguistic";
import { drawMesh } from "./lib/mesh";
import { type ResearchFrame } from "./lib/researchSchema";
import { saveSession, type StoredSession } from "./lib/storage";
import { detectFace, loadFaceLandmarker, type VisionFace } from "./lib/vision";
import type { FeatureSample, FeedbackMode, SpeechEvent } from "./types";

export interface Snapshot {
  status: "idle" | "starting" | "live" | "error";
  error: string | null;
  elapsed: number;
  mode: FeedbackMode;
  meshVisible: boolean;
  micLevel: number;
  speechActive: boolean;
  tracking: boolean;
  visionReady: boolean;
  visionError: string | null;
  cameraOn: boolean;
  transcript: string;
  interim: string;
  readingSpoken: string;
  events: SpeechEvent[];
  reminder: string | null;
  showMarkers: boolean;
  showList: boolean;
  relation: string;
  face: { lipAperture: number; jaw: number; compression: number; silentMove: number } | null;
  textAvailable: boolean;
  recognitionSupported: boolean;
  videoCoverage: number;
  feedbackDelayMs: number | null;
  visionMs: number | null;
  language: string;
  goals: string[];
  reminderText: string;
  reminderIntervalSec: number;
}

interface FaceNow {
  stamp: number;
  aperture: number;
  jaw: number;
  compression: number;
  width: number;
  lipVelocity: number;
  landmarkMotion: number;
  poseSpeed: number;
}

const MODES: FeedbackMode[] = ["quiet", "gentle", "reminders"];

function pref(key: string, fallback: string): string {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function writePref(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* private mode */
  }
}

export class SessionEngine {
  private listeners = new Set<(snapshot: Snapshot) => void>();
  private video: HTMLVideoElement | null = null;
  private mesh: HTMLCanvasElement | null = null;
  private chart: HTMLCanvasElement | null = null;
  private stream: MediaStream | null = null;
  private previewStream: MediaStream | null = null;
  private audioStream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private recognition: SpeechRecognitionLike | null = null;
  private detector: DetState = emptyDetector();
  private samplesSeen = 0;
  private sampleRate = 48000;
  private envelope: number[] = [];
  private previousMag: Float32Array | null = null;
  private previousRms = 0;
  private motion: MotionState = { refLeft: null, refRight: null, previous: null };
  private lastAperture: number | null = null;
  private lastPose: { yaw: number; pitch: number; roll: number } | null = null;
  private lastFaceStamp = 0;
  private faceNow: FaceNow | null = null;
  private landmarks: Pt[] | null = null;
  private history: FeatureSample[] = [];
  private research: ResearchFrame[] = [];
  private audioFrames = 0;
  private videoOkFrames = 0;
  private finalTranscript = "";
  private interim = "";
  private readingIgnore = "";
  private readingFollow = false;
  private liveCues: LinguisticCue[] = [];
  private interimCues: LinguisticCue[] = [];
  private applied = new Set<string>();
  private utteranceStart = -1;
  private lastVoicedAt = -1;
  private lastGrowth = 0;
  private voicedRun = 0;
  private seenEvents = new Set<number>();
  private delays: number[] = [];
  private visionMs: number | null = null;
  private visionReady = false;
  private visionError: string | null = null;
  private visionStamp = 0;
  private visionTimer = 0;
  private previewStarted = false;
  private uiTimer = 0;
  private startedWall = 0;
  private running = false;
  private status: Snapshot["status"] = "idle";
  private error: string | null = null;
  private micLevel = 0;
  private speechActive = false;
  private tracking = false;
  private relation = "Traces will appear once you start.";
  private reminderVisible = false;
  private reminderShown: string | null = null;
  private lastReminderAt = -10_000;
  private mode: FeedbackMode;
  private meshVisible: boolean;
  private language: string;
  private goals: string[];
  private reminderText: string;
  private reminderIntervalSec: number;
  private recognitionSupported = Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);

  constructor() {
    const storedMode = pref("alongside.mode", "quiet");
    this.mode = MODES.includes(storedMode as FeedbackMode) ? (storedMode as FeedbackMode) : "quiet";
    this.meshVisible = pref("alongside.mesh", "1") !== "0";
    this.language = pref("alongside.language", "en-US");
    this.reminderText = pref("alongside.reminder", "You can take your time.");
    this.reminderIntervalSec = Math.max(30, Number(pref("alongside.interval", "180")) || 180);
    try {
      const parsed = JSON.parse(pref("alongside.goals", "[]")) as unknown;
      this.goals = Array.isArray(parsed) ? parsed.filter((item) => typeof item === "string") : [];
    } catch {
      this.goals = [];
    }
  }

  subscribe(listener: (snapshot: Snapshot) => void): () => void {
    this.listeners.add(listener);
    listener(this.getSnapshot());
    return () => this.listeners.delete(listener);
  }

  getSnapshot(): Snapshot {
    const elapsed = this.audioNow();
    const message = liveMessage({
      mode: this.mode,
      now: elapsed,
      lastReminderAt: this.lastReminderAt,
      intervalSec: this.reminderIntervalSec,
      reminderText: this.reminderText,
      detected: [],
    });
    return {
      status: this.status,
      error: this.error,
      elapsed,
      mode: this.mode,
      meshVisible: this.meshVisible,
      micLevel: this.micLevel,
      speechActive: this.speechActive,
      tracking: this.tracking,
      visionReady: this.visionReady,
      visionError: this.visionError,
      cameraOn: Boolean(this.previewStream?.getVideoTracks().some((track) => track.readyState === "live")),
      transcript: this.finalTranscript,
      interim: this.interim,
      readingSpoken: this.readingSpoken(),
      events: this.detector.events.map((event) => ({ ...event, evidence: [...event.evidence] })),
      reminder: this.reminderVisible ? this.reminderShown : null,
      showMarkers: message.showMarkers,
      showList: this.mode === "gentle",
      relation: this.relation,
      face: this.faceNow
        ? {
            lipAperture: this.faceNow.aperture,
            jaw: this.faceNow.jaw,
            compression: this.faceNow.compression,
            silentMove: this.micLevel < 0.02 ? this.faceNow.lipVelocity + this.faceNow.landmarkMotion : 0,
          }
        : null,
      textAvailable: this.finalTranscript.length > 0 || this.interim.length > 0,
      recognitionSupported: this.recognitionSupported,
      videoCoverage: this.audioFrames ? this.videoOkFrames / this.audioFrames : 0,
      feedbackDelayMs: median(this.delays),
      visionMs: this.visionMs,
      language: this.language,
      goals: [...this.goals],
      reminderText: this.reminderText,
      reminderIntervalSec: this.reminderIntervalSec,
    };
  }

  attach(elements: { video: HTMLVideoElement; mesh: HTMLCanvasElement; chart: HTMLCanvasElement }): void {
    this.video = elements.video;
    this.mesh = elements.mesh;
    this.chart = elements.chart;
    if (this.previewStream && this.video.srcObject !== this.previewStream) {
      this.video.srcObject = this.previewStream;
      this.video.muted = true;
      void this.video.play().catch(() => undefined);
    }
    this.ensurePreview();
  }

  reloadFaceModel(): void {
    this.visionReady = false;
    this.visionError = null;
    this.publish();
    void this.loadModel();
  }

  setMode(mode: FeedbackMode): void {
    this.mode = mode;
    writePref("alongside.mode", mode);
    if (mode !== "reminders") this.reminderVisible = false;
    this.publish();
  }

  setMeshVisible(visible: boolean): void {
    this.meshVisible = visible;
    writePref("alongside.mesh", visible ? "1" : "0");
    this.paint();
    this.publish();
  }

  setLanguage(language: string): void {
    this.language = language;
    writePref("alongside.language", language);
    if (this.recognition) this.recognition.lang = language;
    this.publish();
  }

  setGoals(goals: string[]): void {
    this.goals = goals;
    writePref("alongside.goals", JSON.stringify(goals));
    this.publish();
  }

  setReminder(text: string, intervalSec: number): void {
    this.reminderText = text;
    this.reminderIntervalSec = Math.max(30, intervalSec);
    writePref("alongside.reminder", text);
    writePref("alongside.interval", String(this.reminderIntervalSec));
    this.publish();
  }

  dismissReminder(): void {
    this.reminderVisible = false;
    this.lastReminderAt = this.audioNow();
    this.publish();
  }

  /** Ignore speech from before this call, so a passage can start at the first word. */
  markReading(): void {
    this.readingIgnore = `${this.finalTranscript} ${this.interim}`.trim();
    this.publish();
  }

  /** Listen for words when a passage is open, including before a session starts. */
  followReading(active: boolean): void {
    this.readingFollow = active;
    if (active) this.startRecognition();
    else if (!this.running) this.stopRecognition();
  }

  async start(): Promise<void> {
    if (this.running || this.status === "starting") return;
    this.status = "starting";
    this.error = null;
    this.publish();
    try {
      if (!this.previewStream) await this.openCamera();
      const audio = { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 };
      this.audioStream = await navigator.mediaDevices.getUserMedia({ audio });
      this.stream = new MediaStream([
        ...(this.previewStream?.getVideoTracks() ?? []),
        ...this.audioStream.getAudioTracks(),
      ]);
      this.audioContext = new AudioContext();
      await this.audioContext.resume();
      await this.audioContext.audioWorklet.addModule("/worklets/capture.js");
      const source = this.audioContext.createMediaStreamSource(this.stream);
      const worklet = new AudioWorkletNode(this.audioContext, "capture-processor");
      const mute = this.audioContext.createGain();
      mute.gain.value = 0;
      source.connect(worklet);
      worklet.connect(mute);
      mute.connect(this.audioContext.destination);
      this.running = true;
      this.startedWall = Date.now();
      this.lastReminderAt = 0;
      worklet.port.onmessage = (event: MessageEvent<{ samples: Float32Array; sampleRate: number }>) => {
        this.onAudio(event.data.samples, event.data.sampleRate);
      };
      this.startRecorder();
      this.startRecognition();
      this.status = "live";
      this.uiTimer = window.setInterval(() => this.publish(), 160);
      this.publish();
    } catch (error) {
      this.running = false;
      this.error = error instanceof Error ? friendlyMediaError(error) : "The microphone could not start.";
      window.clearInterval(this.uiTimer);
      this.stopRecognition();
      if (this.readingFollow) this.startRecognition();
      await this.audioContext?.close().catch(() => undefined);
      this.audioContext = null;
      this.releaseAudio();
      this.status = "idle";
      this.publish();
    }
  }

  async stop(): Promise<StoredSession | null> {
    if (!this.running && this.status !== "live") return null;
    this.running = false;
    window.clearInterval(this.uiTimer);
    const elapsed = this.audioNow();
    for (const event of this.detector.events) event.state = "final";
    const recording = await this.finishRecording();
    this.stopRecognition();
    await this.audioContext?.close().catch(() => undefined);
    this.audioContext = null;
    this.releaseAudio();
    const session: StoredSession = {
      id: `${this.startedWall || Date.now()}`,
      startedAt: this.startedWall || Date.now(),
      duration: elapsed,
      mode: this.mode,
      events: this.detector.events.map((event) => ({ ...event, evidence: [...event.evidence] })),
      transcript: this.finalTranscript,
      reflection: "",
      features: this.history.filter((_, index) => index % 4 === 0),
      research: this.research,
      videoCoverage: this.audioFrames ? this.videoOkFrames / this.audioFrames : 0,
      textAvailable: this.finalTranscript.length > 0,
      language: this.language,
      goals: [...this.goals],
      recording,
    };
    const saved = await saveSession(session);
    this.resetBuffers();
    this.status = "idle";
    this.publish();
    return saved;
  }

  private ensurePreview(): void {
    if (this.previewStarted) return;
    this.previewStarted = true;
    void this.openCamera();
    void this.loadModel();
  }

  private async openCamera(): Promise<void> {
    try {
      this.previewStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      if (this.video) {
        this.video.srcObject = this.previewStream;
        this.video.muted = true;
        await this.video.play();
      }
      this.kickVision();
      this.publish();
    } catch (error) {
      this.error = error instanceof Error ? friendlyMediaError(error) : "The camera could not start.";
      this.publish();
    }
  }

  private async loadModel(): Promise<void> {
    try {
      await loadFaceLandmarker();
      this.visionReady = true;
      this.visionError = null;
      this.kickVision();
      this.publish();
    } catch (error) {
      this.visionReady = false;
      this.visionError =
        error instanceof Error && error.message
          ? error.message
          : "The face model did not load. This page needs cdn.jsdelivr.net and storage.googleapis.com.";
      this.publish();
    }
  }

  private kickVision(): void {
    if (this.visionTimer) return;
    const tick = () => {
      this.visionTimer = window.setTimeout(tick, 80);
      this.readFace();
    };
    this.visionTimer = window.setTimeout(tick, 40);
  }

  private readFace(): void {
    if (!this.visionReady || !this.video) return;
    const started = performance.now();
    let timestamp = Math.round(performance.now());
    if (timestamp <= this.visionStamp) timestamp = this.visionStamp + 1;
    this.visionStamp = timestamp;
    try {
      const face = detectFace(this.video, timestamp);
      this.visionMs = performance.now() - started;
      this.applyFace(face);
    } catch {
      this.tracking = false;
      this.landmarks = null;
      this.paint();
    }
  }

  private applyFace(face: VisionFace | null): void {
    const wasTracking = this.tracking;
    const stamp = this.running ? this.audioNow() : 0;
    if (!face) {
      this.tracking = false;
      this.landmarks = null;
      this.faceNow = null;
      this.paint();
      if (wasTracking) this.publish();
      return;
    }
    const landmarks = face.landmarks.map(([x, y, z]) => ({ x, y, z }));
    const measured = measureLipMotion(landmarks, this.motion);
    this.motion = measured.state;
    const aperture = lipAperture(landmarks);
    const pose = poseFromMatrix(face.matrix, landmarks[33] ?? { x: 0, y: 0 }, landmarks[263] ?? { x: 1, y: 0 });
    const dt = this.lastFaceStamp ? Math.max(0.04, stamp - this.lastFaceStamp) : 0.07;
    const lipVelocity = this.lastAperture === null ? 0 : Math.abs(aperture - this.lastAperture) / dt;
    const poseSpeed = this.lastPose
      ? Math.hypot(pose.yaw - this.lastPose.yaw, pose.pitch - this.lastPose.pitch, pose.roll - this.lastPose.roll) / dt
      : 0;
    this.lastAperture = aperture;
    this.lastPose = pose;
    this.lastFaceStamp = stamp;
    this.tracking = true;
    this.landmarks = landmarks;
    this.faceNow = {
      stamp,
      aperture,
      jaw: jawOpening(face.blendshapes),
      compression: lipCompressionShape(face.blendshapes),
      width: mouthWidth(landmarks),
      lipVelocity,
      landmarkMotion: measured.motion,
      poseSpeed,
    };
    this.paint();
    if (!wasTracking) this.publish();
  }

  private startRecorder(): void {
    if (!this.stream) return;
    this.chunks = [];
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9,opus")
      ? "video/webm;codecs=vp9,opus"
      : MediaRecorder.isTypeSupported("video/webm")
        ? "video/webm"
        : "";
    this.recorder = mime
      ? new MediaRecorder(this.stream, { mimeType: mime, videoBitsPerSecond: 700_000 })
      : new MediaRecorder(this.stream);
    this.recorder.ondataavailable = (event) => {
      if (event.data.size) this.chunks.push(event.data);
    };
    this.recorder.start(1000);
  }

  private finishRecording(): Promise<Blob | null> {
    const recorder = this.recorder;
    if (!recorder || recorder.state === "inactive") return Promise.resolve(null);
    return new Promise((resolve) => {
      recorder.onstop = () => resolve(new Blob(this.chunks, { type: recorder.mimeType || "video/webm" }));
      recorder.stop();
    });
  }

  private startRecognition(): void {
    if (this.recognition) return;
    const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.recognitionSupported = Boolean(Ctor);
    if (!Ctor) return;
    const recognition = new Ctor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = this.language;
    recognition.onresult = (event) => this.onTranscript(event);
    recognition.onerror = () => undefined;
    recognition.onend = () => {
      if (this.recognition !== recognition) return;
      if (!this.running && !this.readingFollow) return;
      try {
        recognition.start();
      } catch {
        /* Chrome throws if start overlaps an existing session. */
      }
    };
    try {
      recognition.start();
      this.recognition = recognition;
    } catch (error) {
      this.recognition = null;
      const named = error instanceof DOMException ? error.name : "";
      if (named !== "NotAllowedError" && named !== "InvalidStateError") this.recognitionSupported = false;
    }
  }

  private stopRecognition(): void {
    const recognition = this.recognition;
    this.recognition = null;
    try {
      recognition?.stop();
    } catch {
      /* already stopped */
    }
  }

  private readingSpoken(): string {
    const heard = `${this.finalTranscript} ${this.interim}`.trim();
    if (this.readingIgnore && heard.startsWith(this.readingIgnore)) return heard.slice(this.readingIgnore.length).trim();
    if (this.readingIgnore && this.finalTranscript.startsWith(this.readingIgnore)) {
      return `${this.finalTranscript.slice(this.readingIgnore.length)} ${this.interim}`.trim();
    }
    return heard;
  }

  private onAudio(samples: Float32Array, sampleRate: number): void {
    if (!this.running) return;
    this.sampleRate = sampleRate;
    this.samplesSeen += samples.length;
    const t = this.samplesSeen / sampleRate;
    const loudness = rms(samples);
    const crossings = zeroCrossingRate(samples);
    const spectrum = magnitudeSpectrum(samples);
    const centroid = spectralCentroid(spectrum, sampleRate);
    const flux = spectralFlux(spectrum, this.previousMag);
    const lowBand = lowBandRatio(spectrum, sampleRate);
    this.previousMag = spectrum;
    const voiced = voicedScore(loudness, crossings, lowBand);
    this.envelope.push(loudness);
    if (this.envelope.length > 40) this.envelope.shift();
    const face = this.faceNow && Math.abs(this.faceNow.stamp - t) < 0.35 ? this.faceNow : null;
    const speaking = voiced > 0.5 && loudness > 0.02;
    this.voicedRun = speaking ? this.voicedRun + 1 : 0;
    if (speaking) {
      this.lastVoicedAt = t;
      if (this.utteranceStart < 0) this.utteranceStart = t;
    } else if (this.lastVoicedAt >= 0 && t - this.lastVoicedAt > 0.8) {
      this.utteranceStart = -1;
    }
    const textReady = this.finalTranscript.length > 0 || this.interim.length > 0;
    const cues = [...this.liveCues, ...this.interimCues].filter((cue) => t >= cue.start - 0.05 && t <= cue.end + 0.2);
    const repetitionCue = cues.filter((cue) => cue.kind === "repetition").reduce((max, cue) => Math.max(max, cue.confidence), 0);
    const restartCue = cues.filter((cue) => cue.kind === "restart").reduce((max, cue) => Math.max(max, cue.confidence), 0);
    const prolongationCue = textReady && this.voicedRun > 12 && t - this.lastGrowth > 0.6 ? 0.65 : 0;
    this.liveCues = this.liveCues.filter((cue) => cue.end > t - 1);
    this.detector = stepDetector(this.detector, {
      t,
      rms: loudness,
      voiced,
      flux,
      autocorr: envelopeAutocorr(this.envelope),
      videoPresent: Boolean(face),
      lipAperture: face?.aperture ?? 0,
      jawOpen: face?.jaw ?? 0,
      lipCompression: face?.compression ?? 0,
      lipVelocity: face?.lipVelocity ?? 0,
      landmarkMotion: face?.landmarkMotion ?? 0,
      textPresent: textReady,
      repetitionCue,
      restartCue,
      prolongationCue,
    });
    for (const event of this.detector.events) {
      if (this.seenEvents.has(event.id)) continue;
      this.seenEvents.add(event.id);
      this.delays.push(Math.max(0, (t - event.start) * 1000));
      if (this.delays.length > 12) this.delays.shift();
    }
    const silentMove = face && loudness < 0.02 ? face.lipVelocity + face.landmarkMotion : 0;
    this.history.push({
      t,
      rms: loudness,
      lip: face?.aperture ?? 0,
      jaw: face?.jaw ?? 0,
      compression: face?.compression ?? 0,
      silentMove,
    });
    if (this.history.length > 500) this.history.splice(0, this.history.length - 500);
    if (this.audioFrames % 4 === 0) {
      this.research.push({
        t,
        audio: [Math.log10(loudness + 1e-4), crossings, centroid, flux, lowBand, envelopeAutocorr(this.envelope), voiced, loudness - this.previousRms],
        visual: face
          ? [face.aperture, face.jaw, face.compression, face.width, face.lipVelocity, face.landmarkMotion, face.poseSpeed, 1]
          : [0, 0, 0, 0, 0, 0, 0, 0],
        text: [repetitionCue, restartCue, prolongationCue, textReady ? 1 : 0],
      });
    }
    this.previousRms = loudness;
    this.audioFrames += 1;
    if (face) this.videoOkFrames += 1;
    this.micLevel = loudness;
    this.speechActive = speaking;
    const recent = this.history.filter((sample) => sample.t >= t - 8);
    if (recent.length >= 10) {
      this.relation = describeCoupling(recent.map((sample) => ({ rms: sample.rms, aperture: sample.lip, silentMove: sample.silentMove })));
    }
    if (this.mode === "reminders" && !this.reminderVisible && t - this.lastReminderAt >= this.reminderIntervalSec && this.reminderText.trim()) {
      this.reminderVisible = true;
      this.reminderShown = this.reminderText.trim();
    }
    this.paint();
  }

  private onTranscript(event: SpeechResultEvent): void {
    let interim = "";
    let finals = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const text = event.results[index][0].transcript;
      if (event.results[index].isFinal) finals += text;
      else interim += text;
    }
    const now = this.audioNow();
    if (finals.trim()) {
      this.finalTranscript = `${this.finalTranscript} ${finals}`.trim();
      this.interim = "";
      this.interimCues = [];
      const start = this.utteranceStart >= 0 ? this.utteranceStart : Math.max(0, now - 1.5);
      for (const cue of findLinguisticEvents(finals, start, Math.max(now, start + 0.3))) {
        const key = `${cue.kind}:${Math.round(cue.start * 2)}:${cue.detail}`;
        if (this.applied.has(key)) continue;
        this.applied.add(key);
        if (cue.end < now - 0.15) this.detector = applyDelayedCue(this.detector, cue, now);
        else this.liveCues.push(cue);
      }
    } else {
      this.interim = interim.trim();
      const start = this.utteranceStart >= 0 ? this.utteranceStart : Math.max(0, now - 1.5);
      this.interimCues = interim.trim() ? findLinguisticEvents(interim, start, Math.max(now, start + 0.3)) : [];
    }
    if ((finals + interim).trim()) this.lastGrowth = now;
    this.publish();
  }

  private paint(): void {
    if (this.chart) {
      const show = this.mode === "gentle";
      drawCharts(this.chart, this.history, this.detector.events, this.audioNow(), show);
    }
    if (this.mesh && this.video) drawMesh(this.mesh, this.video, this.landmarks, this.meshVisible);
  }

  private audioNow(): number {
    if (!this.sampleRate) return 0;
    return this.samplesSeen / this.sampleRate;
  }

  private publish(): void {
    const snapshot = this.getSnapshot();
    for (const listener of this.listeners) listener(snapshot);
  }

  private releaseAudio(): void {
    this.audioStream?.getTracks().forEach((track) => track.stop());
    this.audioStream = null;
    this.stream = null;
  }

  private resetBuffers(): void {
    this.detector = emptyDetector();
    this.samplesSeen = 0;
    this.envelope = [];
    this.previousMag = null;
    this.history = [];
    this.research = [];
    this.audioFrames = 0;
    this.videoOkFrames = 0;
    this.finalTranscript = "";
    this.interim = "";
    this.readingIgnore = "";
    this.liveCues = [];
    this.interimCues = [];
    this.applied.clear();
    this.seenEvents.clear();
    this.delays = [];
    this.faceNow = null;
    this.landmarks = null;
    this.motion = { refLeft: null, refRight: null, previous: null };
    this.utteranceStart = -1;
    this.reminderVisible = false;
    this.micLevel = 0;
    this.speechActive = false;
    this.relation = "Traces will appear once you start.";
  }
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function friendlyMediaError(error: Error): string {
  if (error.name === "NotAllowedError") return "The browser did not get permission to use the microphone or camera.";
  if (error.name === "NotFoundError") return "No camera or microphone was found.";
  return error.message || "The microphone could not start.";
}
