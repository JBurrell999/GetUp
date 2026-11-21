"use client";

import { useEffect, useRef, useState } from "react";
import { useCapture } from "./CaptureProvider";

// Results of the motion_mirror pipeline: Gemini interprets a phone video (scene, object, which hand, when), the
// person's body is tracked from the video, and a Unitree H2 replays it in MuJoCo from the same camera viewpoint.
const GALLERY = [
  { src: "/mirror/chair.mp4", title: "Moving a chair", note: "Walk over, bend, lift it by the backrest, carry it, set it down." },
  { src: "/mirror/cup.mp4", title: "Fetching a cup", note: "Walk to the table, pick up a cup with the left hand, walk back holding it." },
  { src: "/mirror/bottle.mp4", title: "Drinking from a bottle", note: "Seated: reach across with the right hand, drink, put it back." },
];

interface Status {
  state: "running" | "done" | "error" | "unknown";
  stage: string;
  progress: number;
  elapsed_s: number;
  understood?: string;
  scores?: { scene?: number; motion?: number };
  video?: string;
  error?: string;
}

function TryIt() {
  const { judge } = useCapture();
  const [file, setFile] = useState<File | null>(null);
  const [id, setId] = useState<string | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!id) return;
    let stop = false;
    const tick = async () => {
      const r = await fetch(`/api/mirror/${id}`, { cache: "no-store" }).catch(() => null);
      if (r?.ok && !stop) {
        const s = (await r.json()) as Status;
        setStatus(s);
        if (s.state === "done" || s.state === "error") return;
      }
      if (!stop) setTimeout(tick, 2000);
    };
    tick();
    return () => {
      stop = true;
    };
  }, [id]);

  async function run() {
    if (!file) return;
    setSending(true);
    setError(null);
    const fd = new FormData();
    fd.append("video", file);
    fd.append("code", judge?.code ?? "");
    try {
      const r = await fetch("/api/mirror", { method: "POST", body: fd });
      const j = (await r.json()) as { id?: string; error?: string };
      if (!r.ok || !j.id) throw new Error(j.error || "Couldn't start the run.");
      setStatus({ state: "running", stage: "Reading the video", progress: 0.02, elapsed_s: 0 });
      setId(j.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
    }
  }

  const reset = () => {
    setFile(null);
    setId(null);
    setStatus(null);
    setError(null);
    if (input.current) input.current.value = "";
  };

  if (status?.state === "done" && status.video) {
    return (
      <div className="mirror-try">
        <video className="mirror-result" src={status.video} controls autoPlay loop playsInline muted />
        {status.understood && (
          <p className="mirror-understood">
            <span>Gemini understood</span> {status.understood}
          </p>
        )}
        <p className="mirror-meta num">
          {status.elapsed_s} s
          {status.scores?.motion !== undefined && <> · Gemini judge {status.scores.motion}/10</>}
        </p>
        <div className="up-actions">
          <a className="btn btn--quiet" href={status.video} download="side_by_side.mp4">
            Download
          </a>
          <button type="button" className="btn btn--accent" onClick={reset}>
            Try another video
          </button>
        </div>
      </div>
    );
  }

  if (id && status) {
    return (
      <div className="mirror-try" aria-live="polite">
        <p className="mirror-stage">{status.state === "error" ? "The run stopped." : status.stage}</p>
        <div className="send-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(status.progress * 100)}>
          <span style={{ transform: `scaleX(${status.progress})` }} />
        </div>
        <p className="mirror-meta num">
          {status.elapsed_s} s · usually about 2 to 3 minutes
        </p>
        {status.understood && (
          <p className="mirror-understood">
            <span>Gemini understood</span> {status.understood}
          </p>
        )}
        {status.state === "error" && (
          <>
            <p className="up-error" role="alert">
              {status.error}
            </p>
            <button type="button" className="btn btn--accent" onClick={reset}>
              Try again
            </button>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="mirror-try">
      <input
        ref={input}
        type="file"
        accept="video/*,.mov,.mp4,.m4v,.webm"
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />
      <p className="mirror-pick">
        {file ? (
          <>
            <strong>{file.name}</strong> · {(file.size / 1e6).toFixed(1)} MB
          </>
        ) : (
          "Pick a short phone video of someone doing one thing: picking something up, moving it, sitting, walking."
        )}
      </p>
      {error && (
        <p className="up-error" role="alert">
          {error}
        </p>
      )}
      <div className="up-actions">
        <button type="button" className="btn btn--quiet" onClick={() => input.current?.click()}>
          {file ? "Choose another" : "Choose a video"}
        </button>
        <button type="button" className="btn btn--accent" disabled={!file || sending} onClick={run}>
          {sending ? "Uploading…" : "Run it on the robot"}
        </button>
      </div>
    </div>
  );
}

export function Mirror() {
  const [avail, setAvail] = useState<boolean | null>(null);
  useEffect(() => {
    fetch("/api/mirror", { cache: "no-store" })
      .then((r) => r.json())
      .then((j: { available?: boolean }) => setAvail(Boolean(j.available)))
      .catch(() => setAvail(false));
  }, []);

  return (
    <section className="mirror" id="mirror" aria-labelledby="mirror-title">
      <div className="wrap">
        <div className="envs-head">
          <h2 className="envs-title" id="mirror-title">
            From a phone video to the robot
          </h2>
          <p className="envs-lede">
            Gemini watches the video and works out the scene, the object, which hand does what and when. The body is
            tracked from the video frame by frame, and a Unitree H2 does the same thing in MuJoCo, seen from the same
            camera. Left: the person. Right: the robot.
          </p>
        </div>

        <ul className="mirror-grid">
          {GALLERY.map((g) => (
            <li key={g.src} className="mirror-card">
              <video src={g.src} controls loop playsInline muted preload="metadata" />
              <p className="mirror-card-title">{g.title}</p>
              <p className="mirror-card-note">{g.note}</p>
            </li>
          ))}
        </ul>

        <div className="mirror-run">
          <h3 className="vids-title">Try it with your own video</h3>
          {avail === null && <p className="mirror-pick">Checking whether the robot pipeline is running…</p>}
          {avail === false && (
            <p className="mirror-pick">
              The robot pipeline runs on the team&apos;s laptop. Open this site from there (or through its ngrok link) to
              run your own video.
            </p>
          )}
          {avail && <TryIt />}
        </div>
      </div>
    </section>
  );
}
