"use client";

import { useEffect, useMemo, useState } from "react";
import { CATEGORIES, lengthLabel, sceneNo, splitProp, TASKS, TIERS, type Task, type Tier } from "@/lib/tasks";
import { useCapture } from "./CaptureProvider";

type TierFilter = Tier | "all";
const TIER_KEY = "getup.tier";
const cap = (s: string) => s[0].toUpperCase() + s.slice(1);

function matches(t: Task, q: string) {
  if (!q) return true;
  const hay = `${t.name} ${t.instruction} ${t.id} ${t.category} ${t.props.join(" ")}`.toLowerCase();
  return q
    .toLowerCase()
    .split(/\s+/)
    .every((w) => hay.includes(w));
}

export function ShotList() {
  const { config, takesByTask, takesError, openTask, setOrder, judge } = useCapture();
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
    () => CATEGORIES.map((c) => ({ name: c, tasks: visible.filter((t) => t.category === c) })).filter((g) => g.tasks.length),
    [visible],
  );
  useEffect(() => setOrder(groups.flatMap((g) => g.tasks.map((t) => t.id))), [groups, setOrder]);

  const done = inTier.filter((t) => takesByTask.has(t.id)).length;
  const sets: { id: TierFilter; label: string; n: number }[] = [
    ...TIERS.map((t) => ({ id: t.id as TierFilter, label: t.label, n: TASKS.filter((x) => x.tier === t.id).length })),
    { id: "all", label: "Everything", n: TASKS.length },
  ];
  const blurb = TIERS.find((t) => t.id === tier)?.blurb ?? "Every task, in the order a full session runs them.";

  return (
    <section className="tasks" id="tasks" aria-labelledby="tasks-title">
      <div className="wrap tasks-grid">
        <aside className="side">
          <h2 className="side-title" id="tasks-title">
            Tasks
          </h2>
          <p className="side-blurb">{blurb}</p>

          <nav className="sets" aria-label="Task set">
            {sets.map((s) => (
              <button key={s.id} type="button" aria-pressed={tier === s.id} onClick={() => pickTier(s.id)}>
                <span>{s.label}</span>
                <span className="num">{s.n}</span>
              </button>
            ))}
          </nav>

          {judge && (
            <div className="side-progress">
              <p>
                <span className="num">{done}</span> of <span className="num">{inTier.length}</span> filmed
              </p>
              <span className="meter" aria-hidden="true">
                <span style={{ width: `${inTier.length ? (done / inTier.length) * 100 : 0}%` }} />
              </span>
            </div>
          )}

          <div className="cats" role="group" aria-label="Category">
            <button type="button" aria-pressed={category === "all"} onClick={() => setCategory("all")}>
              <span>All categories</span>
            </button>
            {categories.map((c) => (
              <button
                key={c.name}
                type="button"
                aria-pressed={category === c.name}
                onClick={() => setCategory(category === c.name ? "all" : c.name)}
              >
                <span>{cap(c.name)}</span>
                <span className="num">{c.n}</span>
              </button>
            ))}
          </div>
        </aside>

        <div className="list">
          {!config.uploadsReady && (
            <p className="notice" role="status">
              Uploads aren&apos;t switched on for this site yet. You can browse the tasks, but videos can&apos;t be sent.
            </p>
          )}
          {takesError && (
            <p className="notice" role="status">
              {takesError} New uploads still work.
            </p>
          )}

          <label className="search">
            <span className="sr-only">Search tasks</span>
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <circle cx="8.5" cy="8.5" r="5.5" />
              <path d="M13 13l4.5 4.5" />
            </svg>
            <input type="search" placeholder="Search tasks or props" value={query} onChange={(e) => setQuery(e.target.value)} />
            {query && <span className="search-n num">{visible.length}</span>}
          </label>

          {groups.length === 0 ? (
            <p className="empty">
              Nothing matches “{query}”.{" "}
              <button type="button" className="link-btn" onClick={() => setQuery("")}>
                Clear the search
              </button>
            </p>
          ) : (
            groups.map((g) => (
              <section key={g.name} className="group" aria-label={g.name}>
                <h3 className="group-name">
                  {cap(g.name)}
                  <span className="num">{g.tasks.length}</span>
                </h3>
                <ul>
                  {g.tasks.map((t) => {
                    const n = takesByTask.get(t.id)?.length ?? 0;
                    return (
                      <li key={t.id}>
                        <button type="button" className="row" onClick={() => openTask(t.id)}>
                          <span className="row-no num">{sceneNo(t)}</span>
                          <span className="row-main">
                            <span className="row-name">
                              {t.name}
                              {tier === "all" && t.tier !== "extended" && <span className={`tag tag--${t.tier}`}>{t.tier}</span>}
                            </span>
                            <span className="row-instr">{t.instruction}</span>
                          </span>
                          <span className="row-props">{t.props.map((p) => splitProp(p).name).join(", ")}</span>
                          <span className="row-len">{lengthLabel(t)}</span>
                          <span className={n ? "row-done" : "row-done row-done--none"}>
                            {n ? (
                              <>
                                <svg viewBox="0 0 16 16" aria-hidden="true">
                                  <path d="M3.5 8.5l3 3 6-7" />
                                </svg>
                                <span className="num">{n}</span>
                                <span className="sr-only">{n === 1 ? "take" : "takes"}</span>
                              </>
                            ) : (
                              <span className="sr-only">not filmed yet</span>
                            )}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </section>
            ))
          )}
        </div>
      </div>
    </section>
  );
}
