"use client";

import { TASKS } from "@/lib/tasks";
import { useCapture } from "./CaptureProvider";

const CORE = TASKS.filter((t) => t.tier === "core");

export function TopBar() {
  const { config, hydrated, judge, setJudge, requestJudge, takesByTask, takes, openTask } = useCapture();
  const done = CORE.filter((t) => takesByTask.has(t.id)).length;

  return (
    <>
      {!config.uploadsReady && (
        <div className="notice" role="status">
          <div className="wrap notice-inner">
            <span className="mono">Uploads off</span>
            No Blob store is connected to this deployment yet, so videos can&apos;t be sent. Everything else works.
          </div>
        </div>
      )}
      <header className="bar">
        <div className="wrap bar-inner">
        <a className="wordmark" href="#top" aria-label="GetUp capture, back to top">
          GetUp<span className="wordmark-sub">capture</span>
        </a>

        <div className="bar-right">
          {hydrated && judge && (
            <div className="meter" aria-label={`${done} of ${CORE.length} core tasks recorded`}>
              <span className="meter-count mono">
                <b>{done}</b>/{CORE.length} core
              </span>
              <span className="meter-ticks" aria-hidden="true">
                {CORE.map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    tabIndex={-1}
                    className={takesByTask.has(t.id) ? "tick tick--done" : "tick"}
                    title={t.name}
                    onClick={() => openTask(t.id)}
                  />
                ))}
              </span>
            </div>
          )}

          {!hydrated ? (
            <span className="who who--placeholder" />
          ) : judge ? (
            <div className="who">
              <span className="who-name">{judge.name}</span>
              <span className="who-meta mono">
                {judge.id} · {takes.length} {takes.length === 1 ? "take" : "takes"}
              </span>
              <button
                type="button"
                className="who-switch"
                onClick={() => {
                  setJudge(null);
                  requestJudge();
                }}
              >
                Not you?
              </button>
            </div>
          ) : (
            <button type="button" className="btn btn--ghost btn--sm" onClick={() => requestJudge()}>
              Add your name
            </button>
          )}
        </div>
        </div>
      </header>
    </>
  );
}
