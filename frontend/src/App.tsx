import { useState } from "react";
import { InsightsView } from "./InsightsView";
import { LiveView } from "./LiveView";
import { ReviewView } from "./ReviewView";
import { useSession } from "./useSession";

type Page = "live" | "review" | "insights";

export function App() {
  const { engine, snap } = useSession();
  const [page, setPage] = useState<Page>("live");
  const [ready, setReady] = useState(() => {
    try {
      return localStorage.getItem("alongside.onboarded") === "1";
    } catch {
      return true;
    }
  });

  if (!ready) {
    return (
      <main className="welcome">
        <p className="eyebrow">Speech companion</p>
        <h1>Alongside</h1>
        <p>
          This stays on your computer. The camera and microphone are used to follow timing, sound, and mouth movement while you talk.
        </p>
        <p>Nothing here is a fluency score. A pause is not a mistake, and a detection is not an instruction.</p>
        <button
          className="primary"
          type="button"
          onClick={() => {
            try {
              localStorage.setItem("alongside.onboarded", "1");
            } catch {
              /* continue even if storage is blocked */
            }
            setReady(true);
          }}
        >
          Continue on this device
        </button>
      </main>
    );
  }

  return (
    <main>
      <header className="top">
        <div>
          <p className="eyebrow">Speech companion</p>
          <h1>Alongside</h1>
        </div>
        <nav>
          {(
            [
              ["live", "Live"],
              ["review", "Review"],
              ["insights", "Insights"],
            ] as const
          ).map(([id, label]) => (
            <button key={id} type="button" aria-current={page === id ? "page" : undefined} onClick={() => setPage(id)}>
              {label}
            </button>
          ))}
        </nav>
        {snap.status === "live" && page !== "live" && <p className="listening">A session is still running.</p>}
      </header>
      <div className={page === "live" ? "" : "parked"}>
        <LiveView engine={engine} snap={snap} onStopped={() => setPage("review")} />
      </div>
      {page === "review" && <ReviewView />}
      {page === "insights" && <InsightsView engine={engine} snap={snap} />}
      <footer>
        Observations are not a diagnosis. They are here if you want help understanding your own speech.
      </footer>
    </main>
  );
}
