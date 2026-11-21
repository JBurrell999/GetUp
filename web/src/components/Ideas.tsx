"use client";

import { ENV_BY_ID } from "@/lib/environments";
import { EXAMPLES } from "@/lib/examples";
import { formatBytes, humanize } from "@/lib/takes";
import { useCapture } from "./CaptureProvider";

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
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    </section>
  );
}
