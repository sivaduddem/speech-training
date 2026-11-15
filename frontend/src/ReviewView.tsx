import { useEffect, useRef, useState } from "react";
import { drawCharts } from "./lib/charts";
import { sessionSummary } from "./lib/feedback";
import { researchDocument } from "./lib/researchSchema";
import { deleteSession, listSessions, updateReflection, type StoredSession } from "./lib/storage";
import { formatClock } from "./types";

export function ReviewView() {
  const [sessions, setSessions] = useState<StoredSession[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);

  const selected = sessions.find((session) => session.id === selectedId) ?? null;

  useEffect(() => {
    void listSessions().then((rows) => {
      setSessions(rows);
      setSelectedId((current) => current ?? rows[0]?.id ?? null);
    });
  }, []);

  useEffect(() => {
    const url = selected?.recording ? URL.createObjectURL(selected.recording) : null;
    setVideoUrl(url);
    setNote(selected?.reflection ?? "");
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [selected]);

  useEffect(() => {
    if (!selected || !chartRef.current) return;
    const cursor = videoRef.current ? videoRef.current.currentTime : selected.duration;
    drawCharts(chartRef.current, selected.features, selected.events, cursor, true, Math.max(8, selected.duration));
  }, [selected]);

  function syncChart(): void {
    if (!selected || !chartRef.current || !videoRef.current) return;
    drawCharts(chartRef.current, selected.features, selected.events, videoRef.current.currentTime, true, Math.max(8, selected.duration));
  }

  return (
    <div className="review">
      <aside>
        <h2>Sessions</h2>
        {sessions.length === 0 && <p className="fine">When you end a session, it will wait here on this computer.</p>}
        <ul className="session-list">
          {sessions.map((session) => (
            <li key={session.id}>
              <button type="button" className={session.id === selectedId ? "on" : ""} onClick={() => setSelectedId(session.id)}>
                <strong>{new Date(session.startedAt).toLocaleString()}</strong>
                <span>{formatClock(session.duration)}</span>
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <section>
        {!selected && <p>Choose a session when you want to look back.</p>}
        {selected && (
          <>
            <p className="lead">These are the observations from this session. Opening them is your choice.</p>
            {sessionSummary(selected).map((line) => (
              <p key={line}>{line}</p>
            ))}
            {videoUrl ? (
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                playsInline
                onTimeUpdate={syncChart}
              />
            ) : (
              <p className="fine">The video file was not stored. The timeline and notes still are. Alignment of a recording, when present, can drift by a fraction of a second.</p>
            )}
            <canvas ref={chartRef} className="charts" />
            <ul className="events">
              {selected.events.map((event) => (
                <li key={event.id}>
                  <button
                    type="button"
                    onClick={() => {
                      if (videoRef.current) videoRef.current.currentTime = event.start;
                      syncChart();
                    }}
                  >
                    <strong>{event.type === "block" ? "Possible block" : event.type}</strong>
                    <span>
                      {formatClock(event.start)}–{formatClock(event.end)}
                    </span>
                  </button>
                  <p className="fine">{event.evidence.join(" · ") || "Marked from the session."}</p>
                </li>
              ))}
            </ul>
            <label>
              Anything you want to remember
              <textarea
                value={note}
                onChange={(event) => setNote(event.target.value)}
                onBlur={() => void updateReflection(selected.id, note)}
              />
            </label>
            <div className="row-actions">
              <button
                type="button"
                onClick={() => {
                  const blob = new Blob([JSON.stringify(researchDocument(selected.research, selected.events), null, 2)], {
                    type: "application/json",
                  });
                  const url = URL.createObjectURL(blob);
                  const link = document.createElement("a");
                  link.href = url;
                  link.download = `alongside-${selected.id}.json`;
                  link.click();
                  URL.revokeObjectURL(url);
                }}
              >
                Download feature log
              </button>
              <button
                type="button"
                className="danger"
                onClick={() => {
                  void deleteSession(selected.id).then(() => {
                    setSessions((rows) => rows.filter((session) => session.id !== selected.id));
                    setSelectedId(null);
                  });
                }}
              >
                Delete session
              </button>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
