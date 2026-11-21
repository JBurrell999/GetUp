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
  const { openId, openTask, order, takesByTask } = useCapture();
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
          <header className="sheet-head">
            <p className="sheet-meta">
              <span className="num">{sceneNo(task)}</span>
              <span>{task.category[0].toUpperCase() + task.category.slice(1)}</span>
              {task.tier !== "extended" && <span className={`tag tag--${task.tier}`}>{task.tier}</span>}
            </p>
            <button type="button" className="sheet-close" onClick={() => openTask(null)} aria-label="Close">
              <svg viewBox="0 0 20 20" aria-hidden="true">
                <path d="M5 5l10 10M15 5L5 15" />
              </svg>
            </button>
          </header>

          <div className="sheet-body" ref={bodyRef}>
            <h2 className="sheet-title" id="sheet-title" ref={titleRef} tabIndex={-1}>
              {task.name}
            </h2>
            <p className="sheet-instr">{task.instruction}</p>

            <dl className="facts">
              <div>
                <dt>You&apos;ll need</dt>
                <dd>
                  {task.props.length === 0 ? (
                    "Nothing but you."
                  ) : (
                    <ul>
                      {task.props.map((p) => {
                        const { name, extra } = splitProp(p);
                        return (
                          <li key={p}>
                            {name[0].toUpperCase() + name.slice(1)}
                            {extra && <span className="facts-extra"> ({extra})</span>}
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </dd>
              </div>
              <div>
                <dt>Ending the take</dt>
                <dd>{endHint(task)}</dd>
              </div>
              {task.note && (
                <div>
                  <dt>Worth knowing</dt>
                  <dd>{task.note}</dd>
                </div>
              )}
            </dl>

            <Uploader key={task.id} task={task} takeNo={mine.length + 1} next={nextOpen ? TASK_BY_ID.get(nextOpen) : undefined} />

            {mine.length > 0 && (
              <div className="mine">
                <h3>Your takes</h3>
                <ol>
                  {mine.map((t, i) => (
                    <li key={t.pathname}>
                      <span>Take {two(i + 1)}</span>
                      <span>{when(t.uploadedAt)}</span>
                      <span className="num">{formatBytes(t.size)}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>

          <nav className="sheet-nav" aria-label="Other tasks">
            <button type="button" disabled={!prev} onClick={() => prev && openTask(prev.id)}>
              <span className="sheet-nav-k">Previous</span>
              <span className="sheet-nav-name">{prev?.name ?? "—"}</span>
            </button>
            <button type="button" disabled={!next} onClick={() => next && openTask(next.id)}>
              <span className="sheet-nav-k">Next</span>
              <span className="sheet-nav-name">{next?.name ?? "—"}</span>
            </button>
          </nav>
        </div>
      )}
    </dialog>
  );
}
