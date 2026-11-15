import { useEffect, useRef, useState } from "react";
import type { SessionEngine, Snapshot } from "./engine";
import { SCRIPTS } from "./lib/scripts";
import { formatClock, type FeedbackMode } from "./types";

const MODES: { id: FeedbackMode; title: string; body: string }[] = [
  { id: "quiet", title: "Quiet listening", body: "Record the session and keep the screen calm. Look afterward if you want to." },
  { id: "gentle", title: "Gentle markers", body: "Small marks on the timeline. No messages while you are talking." },
  { id: "reminders", title: "My reminders", body: "A sentence you wrote, on a timer you choose. It does not appear because something was detected." },
];

const LANGUAGES = [
  ["en-US", "English (US)"],
  ["en-GB", "English (UK)"],
  ["es-ES", "Spanish"],
  ["fr-FR", "French"],
  ["de-DE", "German"],
  ["hi-IN", "Hindi"],
  ["zh-CN", "Chinese"],
];

const EVENT_COPY = {
  repetition: ["Repetition", "A sound or word sounded like it happened more than once."],
  prolongation: ["Prolonged sound", "A sound stayed steady longer than the speech around it."],
  block: ["Possible block", "Sound dropped while the mouth held a shape. An observation, not a diagnosis."],
  restart: ["Restart", "The transcript looked like a phrase began again or was revised."],
  pause: ["Pause", "A quiet moment, kept separate from the other observations."],
} as const;

export function LiveView({
  engine,
  snap,
  onStopped,
}: {
  engine: SessionEngine;
  snap: Snapshot;
  onStopped: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const meshRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const [scriptId, setScriptId] = useState<string | null>(null);
  const script = SCRIPTS.find((item) => item.id === scriptId) ?? null;

  useEffect(() => {
    if (videoRef.current && meshRef.current && chartRef.current) {
      engine.attach({ video: videoRef.current, mesh: meshRef.current, chart: chartRef.current });
    }
  }, [engine]);

  const live = snap.status === "live" || snap.status === "starting";
  const pauses = snap.events.filter((event) => event.type === "pause");
  const marked = snap.events.filter((event) => event.type !== "pause");

  return (
    <div className="live">
      <section className="stage-column">
        <div className="stage">
          <div className="stage-media">
            <video ref={videoRef} playsInline muted />
            <canvas ref={meshRef} className="mesh" />
          </div>
          {snap.status === "idle" && <p className="stage-hint">The preview starts when you do.</p>}
          {live && !snap.cameraOn && <p className="stage-hint">Camera is off, so silent blocks stay grouped with pauses.</p>}
          {live && snap.cameraOn && snap.visionError && <p className="stage-hint">{snap.visionError}</p>}
          {live && snap.cameraOn && !snap.visionReady && !snap.visionError && (
            <p className="stage-hint">Starting face tracking.</p>
          )}
          {live && snap.cameraOn && snap.visionReady && !snap.tracking && (
            <p className="stage-hint">Looking for a face in the frame.</p>
          )}
          <div className="presence" data-on={snap.speechActive ? "yes" : "no"}>
            <span />
            {snap.speechActive ? "Speech in the microphone" : "Quiet"}
          </div>
        </div>
        <div className="transport">
          <button
            className="primary"
            type="button"
            disabled={snap.status === "starting"}
            onClick={() => {
              if (live) {
                void engine.stop().then(() => onStopped());
              } else {
                void engine.start();
              }
            }}
          >
            {snap.status === "starting" ? "Starting" : live ? "End session" : "Start session"}
          </button>
          <button type="button" className={snap.meshVisible ? "on" : ""} onClick={() => engine.setMeshVisible(!snap.meshVisible)}>
            {snap.meshVisible ? "Hide mesh" : "Show mesh"}
          </button>
          <div className="meter" aria-hidden="true">
            <span style={{ width: `${Math.min(100, snap.micLevel * 420)}%` }} />
          </div>
          <strong className="clock">{formatClock(snap.elapsed)}</strong>
        </div>
        {snap.error && <p className="banner">{snap.error}</p>}
        <section className="scripts" aria-label="Something to read">
          <h2>Something to read</h2>
          <p className="fine">Optional. Pick a short passage if you want words in front of you. You can also just talk.</p>
          <div className="script-picks">
            {SCRIPTS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={scriptId === item.id ? "on" : ""}
                aria-pressed={scriptId === item.id}
                onClick={() => setScriptId((current) => (current === item.id ? null : item.id))}
              >
                {item.title}
              </button>
            ))}
          </div>
          {script && <p className="script">{script.body}</p>}
        </section>
        <canvas ref={chartRef} className="charts" />
        <p className="relation">{snap.relation}</p>
        <p className="fine">
          Lip shape is a blendshape score. It does not measure muscle tension, effort, or emotion. Loudness and movement are measurements, not a score.
        </p>
      </section>
      <aside className="panel">
        <h2>How should I show up?</h2>
        <div className="modes">
          {MODES.map((mode) => (
            <button
              key={mode.id}
              type="button"
              className={snap.mode === mode.id ? "mode on" : "mode"}
              onClick={() => engine.setMode(mode.id)}
              aria-pressed={snap.mode === mode.id}
            >
              <strong>{mode.title}</strong>
              <span>{mode.body}</span>
            </button>
          ))}
        </div>
        {snap.mode === "reminders" && (
          <label className="reminder-edit">
            Your sentence
            <textarea
              value={snap.reminderText}
              onChange={(event) => engine.setReminder(event.target.value, snap.reminderIntervalSec)}
              maxLength={180}
            />
            <span className="fine">Shown every {snap.reminderIntervalSec}s while a session is running.</span>
            <input
              type="range"
              min={30}
              max={600}
              step={15}
              value={snap.reminderIntervalSec}
              onChange={(event) => engine.setReminder(snap.reminderText, Number(event.target.value))}
              aria-label="Reminder interval in seconds"
            />
          </label>
        )}
        <label className="language">
          Transcript language
          <select value={snap.language} onChange={(event) => engine.setLanguage(event.target.value)}>
            {LANGUAGES.map(([code, label]) => (
              <option key={code} value={code}>
                {label}
              </option>
            ))}
          </select>
        </label>
        {!snap.recognitionSupported && (
          <p className="fine">This browser has no live transcript. Audio and video still run, and text is left out of the decision.</p>
        )}
        {snap.reminder && (
          <div className="reminder" role="status">
            <p>{snap.reminder}</p>
            <button type="button" onClick={() => engine.dismissReminder()}>
              Dismiss
            </button>
          </div>
        )}
        <h2>What I noticed</h2>
        {snap.mode === "quiet" && <p className="fine">Listening quietly. Marks stay in the session for later, and nothing is highlighted now.</p>}
        {snap.mode === "reminders" && <p className="fine">Reminders are on your timer. Observations wait in review.</p>}
        {snap.showList && marked.length === 0 && pauses.length === 0 && <p className="fine">Nothing marked yet.</p>}
        {snap.showList && (
          <ul className="events">
            {marked.map((event) => (
              <li key={event.id}>
                <div>
                  <strong>{EVENT_COPY[event.type][0]}</strong>
                  <em>{event.state === "provisional" ? "Still settling" : "Kept"}</em>
                </div>
                <p>{EVENT_COPY[event.type][1]}</p>
                <p className="fine">
                  {formatClock(event.start)}–{formatClock(event.end)} · {Math.round(event.confidence * 100)}% on-device estimate
                  {event.timing === "approximate" ? " · transcript timing is approximate" : ""}
                </p>
              </li>
            ))}
          </ul>
        )}
        {snap.showList && pauses.length > 0 && (
          <p className="fine">
            {pauses.length} pause{pauses.length === 1 ? "" : "s"} kept separate.
          </p>
        )}
        {(snap.transcript || snap.interim) && (
          <p className="transcript">
            {snap.transcript} <span>{snap.interim}</span>
          </p>
        )}
      </aside>
    </div>
  );
}
