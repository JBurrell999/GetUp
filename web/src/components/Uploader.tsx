"use client";

import { upload } from "@vercel/blob/client";
import { useEffect, useRef, useState, type DragEvent } from "react";
import { ENV_BY_ID } from "@/lib/environments";
import { exampleFor } from "@/lib/examples";
import { CONTENT_TYPE, formatBytes as size, MAX_BYTES, takePathname, videoExt } from "@/lib/takes";
import { useCapture } from "./CaptureProvider";

interface Probe {
  duration: number;
  width: number;
  height: number;
}

interface Check {
  level: "ok" | "warn" | "block";
  text: string;
}

type Phase =
  | { kind: "idle" }
  | { kind: "picked"; file: File; url: string; probe: Probe | null; checks: Check[] }
  | { kind: "sending"; file: File; loaded: number; total: number; started: number }
  | { kind: "done"; what: string }
  | { kind: "error"; message: string; file?: File; url?: string; probe?: Probe | null; checks?: Check[] };

const sec = (s: number) => (s < 60 ? `${s < 10 ? s.toFixed(1) : Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

function probe(url: string): Promise<Probe | null> {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    const done = (p: Probe | null) => {
      clearTimeout(timer);
      v.removeAttribute("src");
      v.load();
      resolve(p);
    };
    const timer = setTimeout(() => done(null), 8000);
    v.preload = "metadata";
    v.muted = true;
    v.onloadedmetadata = () =>
      done(Number.isFinite(v.duration) ? { duration: v.duration, width: v.videoWidth, height: v.videoHeight } : null);
    v.onerror = () => done(null);
    v.src = url;
  });
}

function check(file: File, p: Probe | null): Check[] {
  const out: Check[] = [];
  if (!videoExt(file)) return [{ level: "block", text: "That isn't a video we can take. Use MP4, MOV or WebM." }];
  if (file.size > MAX_BYTES) return [{ level: "block", text: `${size(file.size)} is over the 2 GB limit. Trim it first.` }];
  if (file.size === 0) return [{ level: "block", text: "The file is empty." }];

  if (!p) {
    out.push({ level: "warn", text: "This browser can't read the file, so it can't be checked here. It can still go up." });
    return out;
  }
  if (p.duration < 1) {
    out.push({ level: "warn", text: `Only ${sec(p.duration)} long. Make sure the whole movement is in it.` });
  } else if (p.duration > 180) {
    out.push({ level: "warn", text: `${sec(p.duration)} long. One thing per video is easiest to copy, so trim it if you can.` });
  } else {
    out.push({ level: "ok", text: `${sec(p.duration)} long` });
  }
  const short = Math.min(p.width, p.height);
  if (short > 0 && short < 720) {
    out.push({ level: "warn", text: `${p.width} × ${p.height} is low. 1080p tracks noticeably better.` });
  } else if (short > 0) {
    out.push({ level: "ok", text: `${p.width} × ${p.height}` });
  }
  return out;
}

function Mark({ level }: { level: Check["level"] }) {
  if (level === "ok")
    return (
      <svg viewBox="0 0 12 12" className="mark mark--ok" role="img" aria-label="OK">
        <path d="M2.5 6.5 5 9l4.5-6" />
      </svg>
    );
  return (
    <svg viewBox="0 0 12 12" className={`mark mark--${level}`} role="img" aria-label={level === "warn" ? "Check" : "Blocked"}>
      <path d="M6 2.5v4.5M6 9.2v.3" />
    </svg>
  );
}

interface Props {
  /** What the person says they're doing in the video. Required before sending. */
  what: string;
  /** True while a file is going up, so the panel can lock the place and the description. */
  onBusy?: (busy: boolean) => void;
  /** "Upload another": the panel clears the description and remounts this. */
  onAnother?: () => void;
}

export function Uploader({ what, onBusy, onAnother }: Props) {
  const { config, judge, requestJudge, addTake, env, closeUpload } = useCapture();
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const abort = useRef<AbortController | null>(null);
  const urlRef = useRef<string | null>(null);
  const named = what.trim();

  const dropUrl = () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
  };
  useEffect(() => () => {
    abort.current?.abort();
    dropUrl();
  }, []);
  useEffect(() => onBusy?.(phase.kind === "sending"), [phase.kind, onBusy]);

  async function pick(file: File | undefined) {
    if (!file) return;
    dropUrl();
    const url = URL.createObjectURL(file);
    urlRef.current = url;
    setPhase({ kind: "picked", file, url, probe: null, checks: [] });
    const p = videoExt(file) ? await probe(url) : null;
    if (urlRef.current !== url) return; // replaced meanwhile
    setPhase({ kind: "picked", file, url, probe: p, checks: check(file, p) });
  }

  function reset() {
    dropUrl();
    setPhase({ kind: "idle" });
    if (input.current) input.current.value = "";
  }

  async function send(file: File, p: Probe | null) {
    if (!env || !named) return; // the button stays disabled until both are set
    const who = judge ?? (await requestJudge());
    if (!who) return;
    const ext = videoExt(file)!;
    const pathname = takePathname(who.id, env, named, ext);
    const ctrl = new AbortController();
    abort.current = ctrl;
    const keep = { url: urlRef.current ?? undefined, probe: p, checks: check(file, p) };
    setPhase({ kind: "sending", file, loaded: 0, total: file.size, started: performance.now() });

    try {
      const blob = await upload(pathname, file, {
        access: config.access,
        handleUploadUrl: "/api/upload",
        clientPayload: JSON.stringify({ code: who.code ?? "" }),
        contentType: file.type || CONTENT_TYPE[ext],
        multipart: file.size > 8e6,
        abortSignal: ctrl.signal,
        onUploadProgress: ({ loaded, total }) =>
          setPhase((cur) => (cur.kind === "sending" ? { ...cur, loaded, total: total || cur.total } : cur)),
      });

      const meta = JSON.stringify({
        pathname: blob.pathname,
        name: who.name,
        code: who.code ?? "",
        consent: true,
        what: named,
        example: exampleFor(named)?.id ?? null,
        video: { duration: p?.duration, width: p?.width, height: p?.height, size: file.size, type: file.type, filename: file.name },
      });
      let saved = false;
      for (let i = 0; i < 3 && !saved; i++) {
        const r = await fetch("/api/takes", { method: "POST", headers: { "Content-Type": "application/json" }, body: meta }).catch(
          () => null,
        );
        saved = Boolean(r?.ok);
        if (!saved) await new Promise((res) => setTimeout(res, 800 * (i + 1)));
      }

      const parts = blob.pathname.split("/");
      addTake({ pathname: blob.pathname, env, what: parts[parts.length - 2], uploadedAt: new Date().toISOString(), size: file.size });
      dropUrl();
      setPhase({ kind: "done", what: named });
      if (!saved) console.warn("[getup] video uploaded, but its details didn't save", blob.pathname);
    } catch (err) {
      if (ctrl.signal.aborted) {
        setPhase({ kind: "picked", file, url: keep.url!, probe: p, checks: keep.checks });
        return;
      }
      const raw = err instanceof Error ? err.message : String(err);
      const message = /client token/i.test(raw)
        ? "The server turned the upload down. If the event code changed, use “Not you?” at the top and enter it again."
        : /network|fetch|failed to fetch/i.test(raw)
          ? "The connection dropped. Check your signal and try again; the video is still selected."
          : raw.replace(/^Vercel Blob:\s*/, "");
      setPhase({ kind: "error", message, file, ...keep });
    } finally {
      abort.current = null;
    }
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    pick(e.dataTransfer.files?.[0]);
  };

  const fileInput = (
    <input
      ref={input}
      type="file"
      accept="video/*,.mov,.mp4,.m4v,.webm"
      className="sr-only"
      tabIndex={-1}
      onChange={(e) => pick(e.target.files?.[0])}
    />
  );

  if (phase.kind === "idle") {
    return (
      <div
        className={drag ? "drop drop--over" : "drop"}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        {fileInput}
        <p className="drop-main">
          <span className="drop-desk">Drop the video here, or</span>
          <span className="drop-touch">Record it now, or pick one you&apos;ve filmed.</span>
        </p>
        <button type="button" className="btn btn--accent" onClick={() => input.current?.click()}>
          <span className="drop-desk">Choose a file</span>
          <span className="drop-touch">Record or choose video</span>
        </button>
        <p className="drop-fine">MP4, MOV or WebM, up to 2 GB. One thing per video.</p>
      </div>
    );
  }

  if (phase.kind === "sending") {
    const pct = phase.total ? Math.min(100, (phase.loaded / phase.total) * 100) : 0;
    const secs = (performance.now() - phase.started) / 1000;
    const rate = secs > 0.5 ? phase.loaded / secs : 0;
    const left = rate > 0 ? (phase.total - phase.loaded) / rate : null;
    return (
      <div className="send" aria-live="polite">
        <div className="send-pct">
          {Math.floor(pct)}
          <span>%</span>
        </div>
        <div
          className="send-bar"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.floor(pct)}
          aria-label="Upload progress"
        >
          <span style={{ transform: `scaleX(${pct / 100})` }} />
        </div>
        <p className="send-meta num">
          {size(phase.loaded)} of {size(phase.total)}
          {rate > 0 && <> · {(rate / 1e6).toFixed(1)} MB/s</>}
          {left !== null && left > 1 && pct < 100 && <> · about {left < 60 ? `${Math.ceil(left)} s` : `${Math.ceil(left / 60)} min`} left</>}
        </p>
        <p className="send-keep">Keep this page open until it finishes.</p>
        <button type="button" className="link-btn" onClick={() => abort.current?.abort()}>
          Cancel
        </button>
      </div>
    );
  }

  if (phase.kind === "done") {
    const place = env ? ENV_BY_ID.get(env) : undefined;
    return (
      <div className="up--done" aria-live="polite">
        <div className="done">
          <svg viewBox="0 0 40 40" className="done-mark" aria-hidden="true">
            <circle cx="20" cy="20" r="18" />
            <path d="M12 20.5l5.5 5.5L28.5 14" />
          </svg>
          <div>
            <p className="done-title">“{phase.what}” is in.</p>
            <p className="done-sub">
              Saved under {judge?.name}
              {place && <>, in the {place.name.toLowerCase()}</>}. Thanks.
            </p>
          </div>
        </div>
        <div className="done-actions">
          <button type="button" className="btn btn--accent" onClick={onAnother}>
            Upload another
          </button>
          <button type="button" className="btn btn--quiet" onClick={closeUpload}>
            Done
          </button>
        </div>
      </div>
    );
  }

  // picked, or error with the file still selected
  const file = phase.file;
  const url = phase.url;
  const checks = phase.kind === "picked" ? phase.checks : (phase.checks ?? []);
  const p = phase.kind === "picked" ? phase.probe : (phase.probe ?? null);
  const blocked = checks.some((c) => c.level === "block");
  const probing = phase.kind === "picked" && checks.length === 0;

  return (
    <div>
      {fileInput}
      <div className="picked">
        {url && !blocked && <video className="picked-video" src={url} controls playsInline muted preload="metadata" />}
        <div className="picked-info">
          <p className="picked-name">{file?.name}</p>
          <p className="picked-size num">{file ? size(file.size) : ""}</p>
          {probing ? (
            <p className="picked-probing">Reading the video…</p>
          ) : (
            <ul className="checks">
              {checks.map((c, i) => (
                <li key={i} className={`check check--${c.level}`}>
                  <Mark level={c.level} />
                  {c.text}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {phase.kind === "error" && (
        <p className="up-error" role="alert">
          {phase.message}
        </p>
      )}
      {!config.uploadsReady && (
        <p className="up-error" role="alert">
          Uploads aren&apos;t switched on for this site yet.
        </p>
      )}
      {!named && !blocked && (
        <p className="up-hint" role="status">
          Say what you&apos;re doing in it, above, to upload.
        </p>
      )}

      <div className="up-actions">
        <button
          type="button"
          className="btn btn--accent"
          disabled={!file || blocked || probing || !config.uploadsReady || !env || !named}
          onClick={() => file && send(file, p)}
        >
          {phase.kind === "error" ? "Try again" : "Upload"}
        </button>
        <button type="button" className="btn btn--quiet" onClick={() => input.current?.click()}>
          Choose another
        </button>
        <button type="button" className="link-btn" onClick={reset}>
          Remove
        </button>
      </div>
    </div>
  );
}
