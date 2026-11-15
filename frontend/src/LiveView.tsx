import { useEffect, useMemo, useRef, useState } from "react";
import type { SessionEngine, Snapshot } from "./engine";
import { parseScript, sentencePage, sentencesOf, spokenCursor, spokenWords, type PassageSentence } from "./lib/prompter";
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
  shown,
  onStopped,
}: {
  engine: SessionEngine;
  snap: Snapshot;
  shown: boolean;
  onStopped: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const meshRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const [scriptId, setScriptId] = useState<string | null>(null);
  const script = SCRIPTS.find((item) => item.id === scriptId) ?? null;
  const lines = useMemo(() => (script ? parseScript(script.body) : []), [script]);
  const sentences = useMemo(() => sentencesOf(lines), [lines]);
  const matched = useMemo(
    () => spokenCursor(lines.flat().map((word) => word.norm), spokenWords(snap.readingSpoken)),
    [lines, snap.readingSpoken],
  );
  const [cursor, setCursor] = useState(0);
  const [page, setPage] = useState(0);
  const [followed, setFollowed] = useState(0);
  const [seenScript, setSeenScript] = useState(scriptId);
  if (scriptId !== seenScript) {
    setSeenScript(scriptId);
    setCursor(0);
    setPage(0);
    setFollowed(0);
  }
  const heardPage = sentencePage(sentences, cursor);
  useEffect(() => {
    setCursor((current) => (snap.readingSpoken.trim() === "" ? 0 : Math.max(current, matched)));
  }, [matched, scriptId, snap.readingSpoken]);
  useEffect(() => {
    if (snap.readingSpoken.trim() !== "") return;
    setFollowed(0);
    setPage(0);
  }, [snap.readingSpoken]);
  useEffect(() => {
    if (snap.readingSpoken.trim() === "" || heardPage <= followed) return;
    setFollowed(heardPage);
    setPage((current) => Math.max(current, heardPage));
  }, [heardPage, followed, snap.readingSpoken]);

  useEffect(() => {
    if (videoRef.current && meshRef.current && chartRef.current) {
      engine.attach({ video: videoRef.current, mesh: meshRef.current, chart: chartRef.current });
    }
  }, [engine]);

  useEffect(() => {
    engine.markReading();
  }, [engine, scriptId]);

  useEffect(() => {
    engine.followReading(shown && script !== null);
  }, [engine, shown, script]);

  const live = snap.status === "live" || snap.status === "starting";
  const readyToStart = snap.visionReady && snap.cameraOn;
  const modelLabel = snap.visionError
    ? snap.visionError
    : !snap.visionReady
      ? "Loading the face model."
      : !snap.cameraOn
        ? "Face model ready. Turning the camera on."
        : snap.tracking
          ? "Face model ready."
          : "Face model ready. Looking for a face.";
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
          {script && (
            <Teleprompter
              sentences={sentences}
              cursor={cursor}
              page={Math.min(page, Math.max(0, sentences.length - 1))}
              onPage={setPage}
            />
          )}
          {!script && !live && !snap.cameraOn && !snap.error && <p className="stage-hint">Turning the camera on.</p>}
          {live && !snap.cameraOn && <p className="stage-hint">Camera is off, so silent blocks stay grouped with pauses.</p>}
          {live && (
            <div className="presence" data-on={snap.speechActive ? "yes" : "no"}>
              <span />
              {snap.speechActive ? "Speech in the microphone" : "Quiet"}
            </div>
          )}
        </div>
        <div className="readiness">
          <p className="model-status" data-state={snap.visionError ? "error" : readyToStart ? "ready" : "loading"}>
            {modelLabel}
          </p>
          {snap.visionError && (
            <button type="button" onClick={() => engine.reloadFaceModel()}>
              Try the face model again
            </button>
          )}
        </div>
        <div className="transport">
          <button
            className="primary"
            type="button"
            disabled={snap.status === "starting" || (!live && !readyToStart)}
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
          <p className="fine">Optional. One sentence at a time, on the camera, with the next one already in view. Next line moves on even when a word was missed.</p>
          <div className="script-picks">
            {SCRIPTS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={scriptId === item.id ? "on" : ""}
                aria-pressed={scriptId === item.id}
                onClick={() => {
                  const next = scriptId === item.id ? null : item.id;
                  setScriptId(next);
                  if (next !== scriptId) engine.markReading();
                  engine.followReading(shown && next !== null);
                }}
              >
                {item.title}
              </button>
            ))}
            {script && (
              <button
                type="button"
                onClick={() => {
                  engine.markReading();
                  setPage(0);
                  setFollowed(0);
                }}
              >
                From the start
              </button>
            )}
          </div>
          {script && !snap.recognitionSupported && (
            <p className="fine">This browser has no live transcript, so the words stay still.</p>
          )}
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

function Teleprompter({
  sentences,
  cursor,
  page,
  onPage,
}: {
  sentences: PassageSentence[];
  cursor: number;
  page: number;
  onPage: (page: number) => void;
}) {
  const current = sentences[page];
  const upcoming = sentences[page + 1];
  const last = Math.max(0, sentences.length - 1);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target?.isContentEditable) return;
      if (event.key === "ArrowDown" || event.key === "ArrowRight") {
        event.preventDefault();
        onPage(Math.min(page + 1, last));
      } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
        event.preventDefault();
        onPage(Math.max(page - 1, 0));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last, onPage, page]);

  if (!current) return null;
  return (
    <div className="prompter" aria-label="Passage">
      <div className="prompter-nav">
        <button type="button" onClick={() => onPage(Math.max(page - 1, 0))} disabled={page <= 0}>
          Back
        </button>
        <span>
          {page + 1} of {sentences.length}
        </span>
        <button type="button" onClick={() => onPage(Math.min(page + 1, last))} disabled={page >= last}>
          Next line
        </button>
      </div>
      <p className="prompter-current">{renderWords(current, cursor, true)}</p>
      {upcoming && (
        <button type="button" className="prompter-upcoming" onClick={() => onPage(Math.min(page + 1, last))}>
          {renderWords(upcoming, cursor, false)}
        </button>
      )}
    </div>
  );
}

function renderWords(sentence: PassageSentence, cursor: number, mark: boolean) {
  return sentence.words.map((word, offset) => {
    const wordIndex = sentence.start + offset;
    const state = !mark ? "ahead" : wordIndex < cursor ? "said" : wordIndex === cursor ? "now" : "next";
    return (
      <span key={wordIndex} className={state} aria-current={state === "now" ? "true" : undefined}>
        {word.text}{" "}
      </span>
    );
  });
}
