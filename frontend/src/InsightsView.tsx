import { useEffect, useState } from "react";
import type { SessionEngine, Snapshot } from "./engine";
import { insightLines } from "./lib/feedback";
import { listSessions, type StoredSession } from "./lib/storage";
import { GOALS } from "./types";

export function InsightsView({ engine, snap }: { engine: SessionEngine; snap: Snapshot }) {
  const [sessions, setSessions] = useState<StoredSession[]>([]);
  useEffect(() => {
    void listSessions().then(setSessions);
  }, []);
  const insight = insightLines({
    sessions: sessions.map((session) => ({ startedAt: session.startedAt, duration: session.duration, events: session.events })),
    goals: snap.goals,
  });

  return (
    <div className="insights">
      <section>
        <h2>What you asked the companion to support</h2>
        <ul className="goals">
          {GOALS.map((goal) => (
            <li key={goal.id}>
              <label>
                <input
                  type="checkbox"
                  checked={snap.goals.includes(goal.id)}
                  onChange={() => {
                    const next = snap.goals.includes(goal.id)
                      ? snap.goals.filter((item) => item !== goal.id)
                      : [...snap.goals, goal.id];
                    engine.setGoals(next);
                  }}
                />
                {goal.label}
              </label>
            </li>
          ))}
        </ul>
        {insight.lead.map((line) => (
          <p key={line} className="lead">
            {line}
          </p>
        ))}
        {insight.details.map((line) => (
          <p key={line} className="fine">
            {line}
          </p>
        ))}
        {sessions.some((session) => session.reflection.trim()) && (
          <>
            <h2>Notes you wrote</h2>
            <ul className="notes">
              {sessions
                .filter((session) => session.reflection.trim())
                .map((session) => (
                  <li key={session.id}>{session.reflection}</li>
                ))}
            </ul>
          </>
        )}
      </section>
      <section className="research">
        <h2>About the detection</h2>
        <p>
          The live session runs on this computer. Sound, face shape, and the transcript each vote only when that signal is actually available. A late transcript can revise a moment after you have already moved on. Natural pauses and filled pauses such as “um” are not marked as stuttering.
        </p>
        <p>
          Visible marker delay
          {snap.feedbackDelayMs === null ? " has not been measured in this session yet." : ` in this session is about ${Math.round(snap.feedbackDelayMs)} ms from the start of a marked moment.`}
          {snap.visionMs !== null ? ` The last face frame took about ${Math.round(snap.visionMs)} ms to send.` : ""}
        </p>
        <p className="fine">
          That delay is the on-device detector, not a published accuracy number. Live sound features are loudness, spectral change, and rhythm so the screen can stay responsive. Wav2Vec 2.0 is an optional embedding frontend for training on 16 kHz recordings. MMSD-Net is the architectural reference for separate encoders and attention fusion. Its paper scores are clip-level results on other datasets, and they are not this companion’s accuracy.
        </p>
      </section>
    </div>
  );
}
