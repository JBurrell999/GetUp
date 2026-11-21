"use client";

import { useState } from "react";
import { ENV_BY_ID } from "@/lib/environments";
import { EXAMPLES } from "@/lib/examples";
import { formatBytes, humanize, type RobotFile, type TakeSummary } from "@/lib/takes";
import { useCapture } from "./CaptureProvider";

const FILE_LABEL: Record<RobotFile, string> = { side_by_side: "You and the robot", robot: "Robot", wide: "Wide view" };

/** How far the robot's copy has got, and a button to watch it once it's ready. */
function RobotStatus({ take, open, onToggle }: { take: TakeSummary; open: boolean; onToggle: () => void }) {
  const r = take.robot;
  if (!take.env) return <span className="robot-state">—</span>;
  if (!r) return <span className="robot-state">Robot: queued</span>;
  if (r.status === "processing") return <span className="robot-state robot-state--busy">Robot: copying…</span>;
  if (r.status === "failed" || !r.files.length)
    return <span className="robot-state robot-state--bad">Robot: couldn&apos;t copy</span>;
  return (
    <button type="button" className="robot-watch" aria-expanded={open} onClick={onToggle}>
      {open ? "Hide robot" : "Watch robot"}
      {r.score != null && <span className="num"> · {r.score}/10</span>}
    </button>
  );
}

function RobotPlayer({ take }: { take: TakeSummary }) {
  const files = take.robot?.files ?? [];
  const [file, setFile] = useState<RobotFile>(files[0] ?? "side_by_side");
  const src = `/api/robot?take=${encodeURIComponent(take.pathname)}&file=${file}`;
  return (
    <div className="robot-player">
      {files.length > 1 && (
        <div className="robot-tabs" role="group" aria-label="Which view">
          {files.map((f) => (
            <button key={f} type="button" aria-pressed={f === file} onClick={() => setFile(f)}>
              {FILE_LABEL[f]}
            </button>
          ))}
        </div>
      )}
      <video key={src} src={src} controls playsInline loop autoPlay muted preload="metadata" />
    </div>
  );
}

const two = (n: number) => String(n).padStart(2, "0");
const when = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

function Arrow() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3 8h9M8.5 4.5 12 8l-3.5 3.5" />
    </svg>
  );
}

// There's no list to work through: a few ideas, and room for anything else.
export function Ideas() {
  const { openUpload, hydrated, judge, takes, takesError } = useCapture();
  const mine = [...takes].sort((a, b) => b.uploadedAt.localeCompare(a.uploadedAt));
  const [watching, setWatching] = useState<string | null>(null);

  return (
    <section className="ideas" id="ideas" aria-labelledby="ideas-title">
      <div className="wrap">
        <div className="envs-head">
          <h2 className="envs-title" id="ideas-title">
            Film anything
          </h2>
          <p className="envs-lede">
            There&apos;s no list to get through. Move something, sit, jump, roll around on a mat. Here are a few ideas to
            start with; you name each video yourself when you upload it.
          </p>
        </div>

        <ul className="ideas-grid">
          {EXAMPLES.map((ex, i) => (
            <li key={ex.id}>
              <button type="button" className="idea" onClick={() => openUpload(ex.name)}>
                <span className="idea-no num">{two(i + 1)}</span>
                <span className="idea-name">{ex.name}</span>
                <span className="idea-hint">{ex.hint}</span>
                <span className="idea-go">
                  Upload this <Arrow />
                </span>
              </button>
            </li>
          ))}
          <li>
            <button type="button" className="idea idea--own" onClick={() => openUpload()}>
              <span className="idea-no num">+</span>
              <span className="idea-name">Something else</span>
              <span className="idea-hint">Anything a person does. Say what it is when you upload.</span>
              <span className="idea-go">
                Upload a video <Arrow />
              </span>
            </button>
          </li>
        </ul>

        {hydrated && judge && (mine.length > 0 || takesError) && (
          <div className="vids" id="mine">
            <h3 className="vids-title">
              Your videos <span className="num">{mine.length}</span>
            </h3>
            {takesError && <p className="notice">{takesError} New uploads still work.</p>}
            <ol>
              {mine.map((t) => (
                <li key={t.pathname}>
                  <span className="vids-what">{humanize(t.what)}</span>
                  <span>{t.env ? ENV_BY_ID.get(t.env)?.name : "—"}</span>
                  <span>{when(t.uploadedAt)}</span>
                  <span className="num">{formatBytes(t.size)}</span>
                  <RobotStatus
                    take={t}
                    open={watching === t.pathname}
                    onToggle={() => setWatching((w) => (w === t.pathname ? null : t.pathname))}
                  />
                  {watching === t.pathname && t.robot?.status === "complete" && <RobotPlayer take={t} />}
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    </section>
  );
}
