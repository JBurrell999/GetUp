"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ENV_BY_ID } from "@/lib/environments";
import { EXAMPLES } from "@/lib/examples";
import { useCapture } from "./CaptureProvider";
import { EnvPick } from "./Environments";
import { Uploader } from "./Uploader";

/** Where the video will be filed, with a way to change it; or the choice itself if there's none yet. */
function EnvLine({ locked }: { locked: boolean }) {
  const { env, setEnv } = useCapture();
  const [changing, setChanging] = useState(false);
  const place = env ? ENV_BY_ID.get(env) : undefined;
  if (!place || changing) {
    return (
      <>
        <p className="usheet-sub">Pick the closest match. The robot copies your video in that place.</p>
        <EnvPick
          value={env}
          onPick={(id) => {
            setEnv(id);
            setChanging(false);
          }}
        />
      </>
    );
  }
  return (
    <p className="up-env">
      <img src={place.shots[0].small} alt="" width={640} height={360} />
      <span>
        The <strong>{place.name.toLowerCase()}</strong>
      </span>
      {!locked && (
        <button type="button" className="link-btn" onClick={() => setChanging(true)}>
          Change
        </button>
      )}
    </p>
  );
}

// One panel for every upload: where you're filming, what you're doing, then the file.
export function UploadSheet() {
  const { uploadOpen, uploadWhat, closeUpload, env } = useCapture();
  const ref = useRef<HTMLDialogElement>(null);
  const whatRef = useRef<HTMLInputElement>(null);
  const [what, setWhat] = useState("");
  const [busy, setBusy] = useState(false);
  const [round, setRound] = useState(0); // remounts the uploader for "Upload another"

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (uploadOpen && !d.open) {
      // a fresh start each time, unless a video is still going up from last time
      if (!busy) {
        setWhat(uploadWhat ?? "");
        setRound((r) => r + 1);
      }
      d.showModal();
      d.querySelector<HTMLElement>(".sheet-title")?.focus({ preventScroll: true });
    }
    if (!uploadOpen && d.open) d.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only on open/close
  }, [uploadOpen]);

  const another = useCallback(() => {
    setWhat("");
    setRound((r) => r + 1);
    requestAnimationFrame(() => whatRef.current?.focus());
  }, []);

  return (
    <dialog
      ref={ref}
      className="sheet"
      aria-labelledby="usheet-title"
      onClose={() => uploadOpen && closeUpload()}
      onClick={(e) => {
        if (e.target === e.currentTarget) closeUpload();
      }}
    >
      <div className="sheet-inner">
        <header className="sheet-head">
          <p className="usheet-kicker">Anything you like, one thing per video</p>
          <button type="button" className="sheet-close" onClick={closeUpload} aria-label="Close">
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M5 5l10 10M15 5L5 15" />
            </svg>
          </button>
        </header>

        <div className="sheet-body">
          <h2 className="sheet-title" id="usheet-title" tabIndex={-1}>
            Upload a video
          </h2>

          <section className="ustep" aria-labelledby="ustep-where">
            <h3 className="ustep-k" id="ustep-where">
              <span className="num">1</span> Where are you filming?
            </h3>
            <EnvLine locked={busy} />
          </section>

          <section className="ustep" aria-labelledby="ustep-what">
            <h3 className="ustep-k" id="ustep-what">
              <span className="num">2</span>
              <label htmlFor="usheet-what">What are you doing in it?</label>
            </h3>
            <input
              ref={whatRef}
              id="usheet-what"
              className="what-input"
              value={what}
              maxLength={80}
              disabled={busy}
              autoComplete="off"
              placeholder="In a few words, like “pour a drink”"
              onChange={(e) => setWhat(e.target.value)}
            />
            <div className="what-ideas" role="group" aria-label="Examples">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.id}
                  type="button"
                  disabled={busy}
                  aria-pressed={what.trim().toLowerCase() === ex.name.toLowerCase()}
                  onClick={() => setWhat(ex.name)}
                >
                  {ex.name}
                </button>
              ))}
            </div>
          </section>

          <section className="ustep" aria-labelledby="ustep-file">
            <h3 className="ustep-k" id="ustep-file">
              <span className="num">3</span> The video
            </h3>
            {env ? (
              <Uploader key={round} what={what} onBusy={setBusy} onAnother={another} />
            ) : (
              <p className="usheet-sub">Choose where you&apos;re filming first.</p>
            )}
          </section>
        </div>
      </div>
    </dialog>
  );
}
