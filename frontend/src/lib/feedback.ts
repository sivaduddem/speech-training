import type { EventType, FeedbackMode, SpeechEvent } from "../types";
import { formatClock } from "../types";

const BANNED = [/you should/i, /stop stutter/i, /be fluent/i, /speak fluently/i, /eliminate/i, /muscle tension/i, /anxious/i, /nervous/i];

export function liveMessage(input: {
  mode: FeedbackMode;
  now: number;
  lastReminderAt: number;
  intervalSec: number;
  reminderText: string;
  detected: SpeechEvent[];
}): { text: string | null; showMarkers: boolean } {
  // Detections never choose the words. Reminders are the user's own sentence,
  // on a timer they picked.
  void input.detected;
  if (input.mode === "quiet") return { text: null, showMarkers: false };
  if (input.mode === "gentle") return { text: null, showMarkers: true };
  const due = input.now - input.lastReminderAt >= input.intervalSec;
  const text = input.reminderText.trim();
  return { text: due && text ? text : null, showMarkers: false };
}

export function sessionSummary(session: {
  duration: number;
  events: SpeechEvent[];
  videoCoverage: number;
}): string[] {
  const marked = session.events.filter((event) => event.type !== "pause");
  const pauses = session.events.filter((event) => event.type === "pause");
  const lines = [
    `This session was ${formatClock(session.duration)}.`,
    marked.length === 0
      ? "Nothing was marked as a repetition, prolonged sound, possible block, or restart."
      : `${marked.length} observation${marked.length === 1 ? "" : "s"} can be replayed. They are not a fluency score.`,
    `${pauses.length} pause${pauses.length === 1 ? "" : "s"} kept separate from those observations.`,
  ];
  if (session.videoCoverage < 0.4) {
    lines.push("The camera was missing for much of this session, so silent blocks were not separated from pauses.");
  }
  return lines;
}

export function insightLines(input: {
  sessions: { startedAt: number; duration: number; events: SpeechEvent[] }[];
  goals: string[];
  now?: number;
}): { lead: string[]; details: string[] } {
  const now = input.now ?? Date.now();
  const recent = input.sessions.filter((session) => session.startedAt >= now - 7 * 24 * 3600 * 1000);
  const minutes = Math.max(0, Math.round(recent.reduce((sum, session) => sum + session.duration, 0) / 60));
  const count = (type: EventType) =>
    recent.reduce((sum, session) => sum + session.events.filter((event) => event.type === type).length, 0);
  const lead: string[] = [];
  const goals = input.goals;
  if (goals.length === 0 || goals.includes("company")) {
    lead.push(
      minutes === 0
        ? "No sessions in the last seven days."
        : `You spent about ${minutes} minute${minutes === 1 ? "" : "s"} with the companion this week.`,
    );
  }
  if (goals.includes("review")) {
    lead.push(
      recent.length === 0
        ? "Nothing is waiting in review yet."
        : `${recent.length} session${recent.length === 1 ? "" : "s"} can be opened again.`,
    );
  }
  if (goals.includes("pace")) {
    lead.push("Practice reminders stay on the schedule you set. They are not tied to what was detected.");
  }
  if (goals.includes("notice-blocks")) {
    const blocks = count("block");
    lead.push(
      blocks === 0
        ? "No possible blocks were marked in the last seven days."
        : `${blocks} possible block${blocks === 1 ? "" : "s"} can be replayed from recent sessions.`,
    );
  }
  if (goals.includes("notice-repetitions")) {
    const repetitions = count("repetition");
    lead.push(
      repetitions === 0
        ? "No repetitions were marked in the last seven days."
        : `${repetitions} repetition${repetitions === 1 ? "" : "s"} can be replayed from recent sessions.`,
    );
  }
  const details = [
    "Pauses are counted on their own and are not treated as mistakes.",
    "Counts change with the room, the microphone, and whether the camera could see your mouth.",
    "These notes describe the session. They are not a diagnosis or a target to drive down.",
  ];
  return { lead, details };
}

export function containsBannedCoaching(lines: string[]): boolean {
  return lines.some((line) => BANNED.some((pattern) => pattern.test(line)));
}
