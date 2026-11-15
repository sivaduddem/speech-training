import type { FeatureSample, FeedbackMode, SpeechEvent } from "../types";
import type { ResearchFrame } from "./researchSchema";

export interface StoredSession {
  id: string;
  startedAt: number;
  duration: number;
  mode: FeedbackMode;
  events: SpeechEvent[];
  transcript: string;
  reflection: string;
  features: FeatureSample[];
  research: ResearchFrame[];
  videoCoverage: number;
  textAvailable: boolean;
  language: string;
  goals: string[];
  recording: Blob | null;
}

const DB_NAME = "alongside";
const STORE = "sessions";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: "id" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function saveSession(session: StoredSession): Promise<StoredSession> {
  const db = await openDb();
  try {
    await requestToPromise(db.transaction(STORE, "readwrite").objectStore(STORE).put(session));
    return session;
  } catch {
    const withoutRecording = { ...session, recording: null };
    await requestToPromise(db.transaction(STORE, "readwrite").objectStore(STORE).put(withoutRecording));
    return withoutRecording;
  } finally {
    db.close();
  }
}

export async function listSessions(): Promise<StoredSession[]> {
  const db = await openDb();
  try {
    const rows = await requestToPromise(db.transaction(STORE, "readonly").objectStore(STORE).getAll());
    return (rows as StoredSession[]).sort((a, b) => b.startedAt - a.startedAt);
  } finally {
    db.close();
  }
}

export async function updateReflection(id: string, reflection: string): Promise<void> {
  const db = await openDb();
  try {
    const existing = (await requestToPromise(db.transaction(STORE, "readonly").objectStore(STORE).get(id))) as
      | StoredSession
      | undefined;
    if (!existing) return;
    existing.reflection = reflection;
    await requestToPromise(db.transaction(STORE, "readwrite").objectStore(STORE).put(existing));
  } finally {
    db.close();
  }
}

export async function deleteSession(id: string): Promise<void> {
  const db = await openDb();
  try {
    await requestToPromise(db.transaction(STORE, "readwrite").objectStore(STORE).delete(id));
  } finally {
    db.close();
  }
}
