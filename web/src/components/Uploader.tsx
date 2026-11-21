"use client";

import { upload } from "@vercel/blob/client";
import { useEffect, useRef, useState, type DragEvent } from "react";
import type { Task } from "@/lib/tasks";
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
  | { kind: "done"; takeNo: number }
  | { kind: "error"; message: string; file?: File; url?: string; probe?: Probe | null; checks?: Check[] };

const two = (n: number) => String(n).padStart(2, "0");
const sec = (s: number) => `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;

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

function check(task: Task, file: File, p: Probe | null): Check[] {
  const out: Check[] = [];
  if (!videoExt(file)) return [{ level: "block", text: "That isn't a video we can take. Use MP4, MOV or WebM." }];
  if (file.size > MAX_BYTES) return [{ level: "block", text: `${size(file.size)} is over the 2 GB limit. Trim it first.` }];
  if (file.size === 0) return [{ level: "block", text: "The file is empty." }];

  if (!p) {
    out.push({ level: "warn", text: "This browser can't read the file, so it can't be checked here. It can still go up." });
    return out;
  }
  if (task.stop === "time" && p.duration < task.max_s - 0.3) {
    out.push({ level: "warn", text: `${sec(p.duration)} long. This task needs ${sec(task.max_s)} of you holding it.` });
  } else if (task.stop === "still" && p.duration > task.max_s * 3 + 15) {
    out.push({ level: "warn", text: `${sec(p.duration)} long, for a task that takes up to ${sec(task.max_s)}. Fine, but trim the dead time if you can.` });
  } else {
    out.push({ level: "ok", text: `${sec(p.duration)} long` });
  }
  const short = Math.min(p.width, p.height);
  if (short > 0 && short < 720) {
    out.push({ level: "warn", text: `${p.width} × ${p.height} is low. 1080p tracks noticeably better.` });
  } else if (short > 0) {
    out.push({ level: "ok", text: `${p.width} × ${p.height}` });
  }
  if (p.height > p.width && task.category.startsWith("walking")) {
    out.push({ level: "warn", text: "Portrait video. For walking tasks, landscape keeps you in frame across the room." });
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

export function Uploader({ task, takeNo, next }: { task: Task; takeNo: number; next?: Task }) {
  const { config, judge, requestJudge, addTake, openTask } = useCapture();
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const abort = useRef<AbortController | null>(null);
  const urlRef = useRef<string | null>(null);

  const dropUrl = () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
  };
  useEffect(() => () => {
    abort.current?.abort();
    dropUrl();
  }, []);

  async function pick(file: File | undefined) {
    if (!file) return;
    dropUrl();
    const url = URL.createObjectURL(file);
    urlRef.current = url;
    setPhase({ kind: "picked", file, url, probe: null, checks: [] });
    const p = videoExt(file) ? await probe(url) : null;
    if (urlRef.current !== url) return; // replaced meanwhile
    setPhase({ kind: "picked", file, url, probe: p, checks: check(task, file, p) });
  }

  function reset() {
    dropUrl();
    setPhase({ kind: "idle" });
    if (input.current) input.current.value = "";
  }

  async function send(file: File, p: Probe | null) {
    const who = judge ?? (await requestJudge());
    if (!who) return;
    const ext = videoExt(file)!;
    const pathname = takePathname(who.id, task.id, ext);
    const ctrl = new AbortController();
    abort.current = ctrl;
    const keep = { url: urlRef.current ?? undefined, probe: p, checks: check(task, file, p) };
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

      addTake({ pathname: blob.pathname, task: task.id, uploadedAt: new Date().toISOString(), size: file.size });
      dropUrl();
      setPhase({ kind: "done", takeNo });
      if (!saved) console.warn("[getup] take uploaded, but its details didn't save", blob.pathname);
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
      <div className="up">
        <h3 className="up-title mono">Upload take {two(takeNo)}</h3>
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
          <button type="button" className="btn btn--ink" onClick={() => input.current?.click()}>
            <span className="drop-desk">Choose a file</span>
            <span className="drop-touch">Record or choose video</span>
          </button>
          <p className="drop-fine mono">MP4 · MOV · WebM · up to 2 GB · one task per video</p>
        </div>
      </div>
    );
  }

  if (phase.kind === "sending") {
    const pct = phase.total ? Math.min(100, (phase.loaded / phase.total) * 100) : 0;
    const secs = (performance.now() - phase.started) / 1000;
    const rate = secs > 0.5 ? phase.loaded / secs : 0;
    const left = rate > 0 ? (phase.total - phase.loaded) / rate : null;
    return (
      <div className="up" aria-live="polite">
        <h3 className="up-title mono">Sending take {two(takeNo)}</h3>
        <div className="send">
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
          <p className="send-meta mono">
            {size(phase.loaded)} of {size(phase.total)}
            {rate > 0 && <> · {(rate / 1e6).toFixed(1)} MB/s</>}
            {left !== null && left > 1 && pct < 100 && <> · about {left < 60 ? `${Math.ceil(left)} s` : `${Math.ceil(left / 60)} min`} left</>}
          </p>
          <p className="send-keep">Keep this page open until it finishes.</p>
          <button type="button" className="link" onClick={() => abort.current?.abort()}>
            Cancel
          </button>
        </div>
      </div>
    );
  }

  if (phase.kind === "done") {
    return (
      <div className="up up--done" aria-live="polite">
        <div className="done">
          <svg viewBox="0 0 40 40" className="done-mark" aria-hidden="true">
            <circle cx="20" cy="20" r="18" />
            <path d="M12 20.5l5.5 5.5L28.5 14" />
          </svg>
          <div>
            <p className="done-title">Take {two(phase.takeNo)} is in.</p>
            <p className="done-sub">Saved under {judge?.name}. You can add another take of this task any time.</p>
          </div>
        </div>
        <div className="done-actions">
          {next && (
            <button type="button" className="btn btn--rec" onClick={() => openTask(next.id)}>
              Next: {next.name}
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M3 8h9M8.5 4.5 12 8l-3.5 3.5" />
              </svg>
            </button>
          )}
          <button type="button" className="btn btn--ghost" onClick={reset}>
            Add another take
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
    <div className="up">
      <h3 className="up-title mono">Upload take {two(takeNo)}</h3>
      {fileInput}
      <div className="picked">
        {url && !blocked && (
          <video className="picked-video" src={url} controls playsInline muted preload="metadata" />
        )}
        <div className="picked-info">
          <p className="picked-name">{file?.name}</p>
          <p className="mono picked-size">{file ? size(file.size) : ""}</p>
          {probing ? (
            <p className="mono picked-probing">Reading the video…</p>
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

      <div className="up-actions">
        <button
          type="button"
          className="btn btn--rec"
          disabled={!file || blocked || probing || !config.uploadsReady}
          onClick={() => file && send(file, p)}
        >
          {phase.kind === "error" ? "Try again" : `Upload take ${two(takeNo)}`}
        </button>
        <button type="button" className="btn btn--ghost" onClick={() => input.current?.click()}>
          Choose another
        </button>
        <button type="button" className="link" onClick={reset}>
          Remove
        </button>
      </div>
    </div>
  );
}
