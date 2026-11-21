"use client";

import { useEffect, useRef } from "react";
import { endHint, sceneNo, splitProp, TASK_BY_ID, TASKS } from "@/lib/tasks";
import { formatBytes } from "@/lib/takes";
import { useCapture } from "./CaptureProvider";
import { Uploader } from "./Uploader";

const two = (n: number) => String(n).padStart(2, "0");
const when = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

export function TaskSheet() {
  const { openId, openTask, order, takesByTask, judge } = useCapture();
  const ref = useRef<HTMLDialogElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);
  const task = openId ? TASK_BY_ID.get(openId) : undefined;

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (task && !d.open) {
      d.showModal();
      // start screen readers (and the focus) on the task, not on the close button
      titleRef.current?.focus({ preventScroll: true });
    }
    if (!task && d.open) d.close();
  }, [task]);

  useEffect(() => {
    bodyRef.current?.scrollTo({ top: 0 });
  }, [openId]);

  const ids = task && order.includes(task.id) ? order : TASKS.map((t) => t.id);
  const at = task ? ids.indexOf(task.id) : -1;
  const prev = at > 0 ? TASK_BY_ID.get(ids[at - 1]) : undefined;
  const next = at >= 0 && at < ids.length - 1 ? TASK_BY_ID.get(ids[at + 1]) : undefined;
  // after an upload, suggest the next task in the list that still has no take
  const nextOpen = task
    ? (ids.slice(at + 1).find((id) => !takesByTask.has(id)) ?? ids.find((id) => id !== task.id && !takesByTask.has(id)))
    : undefined;

  const mine = task ? (takesByTask.get(task.id) ?? []) : [];

  return (
    <dialog
      ref={ref}
      className="sheet"
      aria-labelledby="sheet-title"
      onClose={() => openId && openTask(null)}
      onClick={(e) => {
        if (e.target === e.currentTarget) openTask(null);
      }}
    >
      {task && (
        <div className="sheet-inner">
          <header className="slate">
            <div className="slate-grid">
              <div className="slate-cell">
                <span className="slate-k mono">Scene</span>
                <span className="slate-v">{sceneNo(task)}</span>
              </div>
              <div className="slate-cell">
                <span className="slate-k mono">Take</span>
                <span className="slate-v">
                  <i className="rec" aria-hidden="true" />
                  {two(mine.length + 1)}
                </span>
              </div>
              <div className="slate-cell slate-cell--judge">
                <span className="slate-k mono">Judge</span>
                <span className="slate-v slate-v--sm mono">{judge?.id ?? "—"}</span>
              </div>
            </div>
            <div className="slate-foot mono">
              <span>
                {task.category} · {task.tier} · <span className="slate-id">{task.id}</span>
              </span>
              <button type="button" className="slate-close" onClick={() => openTask(null)} aria-label="Close">
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" />
                </svg>
              </button>
            </div>
          </header>

          <div className="sheet-body" ref={bodyRef}>
            <h2 className="sheet-title" id="sheet-title" ref={titleRef} tabIndex={-1}>
              {task.name}
            </h2>
            <p className="sheet-instr">{task.instruction}</p>

            <dl className="spec">
              <div>
                <dt className="mono">Props</dt>
                <dd>
                  {task.props.length === 0 ? (
                    <span className="none">Nothing, just you.</span>
                  ) : (
                    <ul className="spec-props">
                      {task.props.map((p) => {
                        const { name, extra } = splitProp(p);
                        return (
                          <li key={p}>
                            {name}
                            {extra && <span className="spec-extra">{extra}</span>}
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </dd>
              </div>
              <div>
                <dt className="mono">Ending</dt>
                <dd>{endHint(task)}</dd>
              </div>
              {task.note && (
                <div>
                  <dt className="mono">Note</dt>
                  <dd>{task.note}</dd>
                </div>
              )}
            </dl>

            <Uploader
              key={task.id}
              task={task}
              takeNo={mine.length + 1}
              next={nextOpen ? TASK_BY_ID.get(nextOpen) : undefined}
            />

            {mine.length > 0 && (
              <div className="mine">
                <h3 className="mono">Your takes</h3>
                <ol>
                  {mine.map((t, i) => (
                    <li key={t.pathname}>
                      <span className="mono">Take {two(i + 1)}</span>
                      <span>{when(t.uploadedAt)}</span>
                      <span className="mono">{formatBytes(t.size)}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>

          <nav className="sheet-nav" aria-label="Other tasks">
            <button type="button" disabled={!prev} onClick={() => prev && openTask(prev.id)}>
              <span className="mono">Previous</span>
              <span className="sheet-nav-name">{prev?.name ?? "—"}</span>
            </button>
            <button type="button" disabled={!next} onClick={() => next && openTask(next.id)}>
              <span className="mono">Next</span>
              <span className="sheet-nav-name">{next?.name ?? "—"}</span>
            </button>
          </nav>
        </div>
      )}
    </dialog>
  );
}
