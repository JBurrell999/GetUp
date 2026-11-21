"use client";

import { useEffect, useMemo, useState } from "react";
import { CATEGORIES, countByTier, lengthLabel, sceneNo, splitProp, TASKS, TIERS, type Task, type Tier } from "@/lib/tasks";
import { useCapture } from "./CaptureProvider";

type TierFilter = Tier | "all";
const TIER_KEY = "getup.tier";

function matches(t: Task, q: string) {
  if (!q) return true;
  const hay = `${t.name} ${t.instruction} ${t.id} ${t.category} ${t.props.join(" ")}`.toLowerCase();
  return q
    .toLowerCase()
    .split(/\s+/)
    .every((w) => hay.includes(w));
}

export function ShotList() {
  const { takesByTask, takesError, openTask, setOrder, judge } = useCapture();
  const [tier, setTier] = useState<TierFilter>("core");
  const [category, setCategory] = useState<string>("all");
  const [query, setQuery] = useState("");

  useEffect(() => {
    try {
      const saved = localStorage.getItem(TIER_KEY) as TierFilter | null;
      if (saved && (saved === "all" || TIERS.some((t) => t.id === saved))) setTier(saved);
    } catch {}
  }, []);

  const pickTier = (t: TierFilter) => {
    setTier(t);
    setCategory("all");
    try {
      localStorage.setItem(TIER_KEY, t);
    } catch {}
  };

  const inTier = useMemo(() => TASKS.filter((t) => tier === "all" || t.tier === tier), [tier]);
  const categories = useMemo(
    () => CATEGORIES.map((c) => ({ name: c, n: inTier.filter((t) => t.category === c).length })).filter((c) => c.n > 0),
    [inTier],
  );
  const visible = useMemo(
    () => inTier.filter((t) => (category === "all" || t.category === category) && matches(t, query.trim())),
    [inTier, category, query],
  );
  const groups = useMemo(
    () =>
      CATEGORIES.map((c) => ({ name: c, tasks: visible.filter((t) => t.category === c) })).filter(
        (g) => g.tasks.length > 0,
      ),
    [visible],
  );

  useEffect(() => setOrder(groups.flatMap((g) => g.tasks.map((t) => t.id))), [groups, setOrder]);

  const doneHere = inTier.filter((t) => takesByTask.has(t.id)).length;
  const tierInfo = TIERS.find((t) => t.id === tier);

  return (
    <section className="shots" id="shots" aria-labelledby="shots-title">
      <div className="wrap">
        <div className="shots-head">
          <div>
            <p className="eyebrow mono">02 · Shot list</p>
            <h2 className="h2" id="shots-title">
              Pick a task, film it, upload it.
            </h2>
            <p className="shots-sub">
              {tierInfo ? tierInfo.blurb : "Every task, in the order a full session runs them."}{" "}
              {judge && !takesError && (
                <span className="mono shots-done">
                  {doneHere} of {inTier.length} done
                </span>
              )}
              {takesError && <span className="shots-error">{takesError} New uploads still work.</span>}
            </p>
          </div>

          <div className="tabs" role="group" aria-label="Task set">
            {TIERS.map((t) => (
              <button
                key={t.id}
                aria-pressed={tier === t.id}
                className="tab"
                onClick={() => pickTier(t.id)}
              >
                <span className="tab-long">{t.label}</span>
                <span className="tab-short">{t.short}</span>
                <span className="tab-n mono">{countByTier(t.id)}</span>
              </button>
            ))}
            <button aria-pressed={tier === "all"} className="tab" onClick={() => pickTier("all")}>
              All<span className="tab-n mono">{TASKS.length}</span>
            </button>
          </div>
        </div>

        <div className="filters">
          <label className="search">
            <span className="sr-only">Search tasks</span>
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <circle cx="7" cy="7" r="4.5" />
              <path d="M10.5 10.5 14 14" />
            </svg>
            <input
              type="search"
              placeholder="Search: chair, box, mat…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          <div className="cats" role="group" aria-label="Category">
            <button className="cat" aria-pressed={category === "all"} onClick={() => setCategory("all")}>
              All categories
            </button>
            {categories.map((c) => (
              <button
                key={c.name}
                className="cat"
                aria-pressed={category === c.name}
                onClick={() => setCategory(category === c.name ? "all" : c.name)}
              >
                {c.name[0].toUpperCase() + c.name.slice(1)}
                <span className="mono">{c.n}</span>
              </button>
            ))}
          </div>
        </div>

        {groups.length === 0 ? (
          <p className="empty">
            Nothing matches “{query}”.{" "}
            <button className="link" onClick={() => setQuery("")}>
              Clear the search
            </button>
          </p>
        ) : (
          <div className="sheet-list">
            <div className="row row--head mono" aria-hidden="true">
              <span>No.</span>
              <span>Task</span>
              <span>Props</span>
              <span>Length</span>
              <span>Takes</span>
            </div>
            {groups.map((g) => (
              <section key={g.name} className="group" aria-label={g.name}>
                <h3 className="group-name mono">
                  {g.name}
                  <span>{g.tasks.length}</span>
                </h3>
                <ul>
                  {g.tasks.map((t) => {
                    const n = takesByTask.get(t.id)?.length ?? 0;
                    return (
                      <li key={t.id}>
                        <button type="button" className="row" onClick={() => openTask(t.id)}>
                          <span className="row-no mono">{sceneNo(t)}</span>
                          <span className="row-main">
                            <span className="row-name">
                              {t.name}
                              {tier === "all" && t.tier !== "extended" && (
                                <span className={`tier tier--${t.tier}`}>{t.tier}</span>
                              )}
                            </span>
                            <span className="row-instr">{t.instruction}</span>
                          </span>
                          <span className="row-props">
                            {t.props.length === 0 ? (
                              <span className="none">—</span>
                            ) : (
                              t.props.map((p) => (
                                <span key={p} className="prop">
                                  {splitProp(p).name}
                                </span>
                              ))
                            )}
                          </span>
                          <span className="row-len mono">{lengthLabel(t)}</span>
                          <span className={n ? "row-takes row-takes--done mono" : "row-takes mono"}>
                            {n ? (
                              <>
                                <i aria-hidden="true" />
                                {n} {n === 1 ? "take" : "takes"}
                              </>
                            ) : (
                              <span aria-label="no takes yet">—</span>
                            )}
                          </span>
                          <svg className="row-go" viewBox="0 0 16 16" aria-hidden="true">
                            <path d="M3 8h9M8.5 4.5 12 8l-3.5 3.5" />
                          </svg>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </section>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
