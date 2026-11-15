import { useEffect, useState } from "react";
import { SessionEngine, type Snapshot } from "./engine";

export function useSession(): { engine: SessionEngine; snap: Snapshot } {
  const [engine] = useState(() => new SessionEngine());
  const [snap, setSnap] = useState<Snapshot>(() => engine.getSnapshot());
  useEffect(() => engine.subscribe(setSnap), [engine]);
  return { engine, snap };
}
