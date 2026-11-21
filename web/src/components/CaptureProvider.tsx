"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { PublicConfig } from "@/lib/config";
import { TASK_BY_ID } from "@/lib/tasks";
import type { TakeSummary } from "@/lib/takes";

export interface Judge {
  id: string;
  name: string;
  code?: string;
}

interface CaptureState {
  config: PublicConfig;
  /** False until localStorage has been read, so the header doesn't flash "set your name". */
  hydrated: boolean;
  judge: Judge | null;
  setJudge: (j: Judge | null) => void;
  /** Opens the name dialog; resolves with the judge, or null if they backed out. */
  requestJudge: () => Promise<Judge | null>;
  judgeDialog: { open: boolean; resolve: (j: Judge | null) => void };

  takes: TakeSummary[];
  takesByTask: Map<string, TakeSummary[]>;
  takesError: string | null;
  addTake: (t: TakeSummary) => void;

  openId: string | null;
  openTask: (id: string | null) => void;
  /** Ids in the shot list's current order and filter, for previous/next in the sheet. */
  order: string[];
  setOrder: (ids: string[]) => void;
}

const Ctx = createContext<CaptureState | null>(null);

export function useCapture() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useCapture outside CaptureProvider");
  return c;
}

const JUDGE_KEY = "getup.judge";

function taskFromUrl() {
  const id = new URLSearchParams(window.location.search).get("task");
  return id && TASK_BY_ID.has(id) ? id : null;
}

export function CaptureProvider({ config, children }: { config: PublicConfig; children: ReactNode }) {
  const [hydrated, setHydrated] = useState(false);
  const [judge, setJudgeState] = useState<Judge | null>(null);
  const [takes, setTakes] = useState<TakeSummary[]>([]);
  const [takesError, setTakesError] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [order, setOrder] = useState<string[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const resolver = useRef<((j: Judge | null) => void) | null>(null);

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(JUDGE_KEY) ?? "null");
      if (saved?.id && saved?.name) setJudgeState(saved);
    } catch {}
    setOpenId(taskFromUrl());
    setHydrated(true);
    const onPop = () => {
      pushed.current = false;
      setOpenId(taskFromUrl());
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const setJudge = useCallback((j: Judge | null) => {
    setJudgeState(j);
    setTakes([]);
    try {
      if (j) localStorage.setItem(JUDGE_KEY, JSON.stringify(j));
      else localStorage.removeItem(JUDGE_KEY);
    } catch {}
  }, []);

  useEffect(() => {
    if (!judge) return;
    let live = true;
    setTakesError(null);
    fetch(`/api/takes?judge=${encodeURIComponent(judge.id)}`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: { takes: TakeSummary[] }) => {
        if (!live) return;
        // keep anything uploaded while the request was in flight
        setTakes((cur) => {
          const seen = new Set(d.takes.map((t) => t.pathname));
          return [...d.takes, ...cur.filter((t) => !seen.has(t.pathname))];
        });
      })
      .catch(() => live && setTakesError("Couldn't load your earlier takes."));
    return () => {
      live = false;
    };
  }, [judge]);

  const addTake = useCallback((t: TakeSummary) => setTakes((cur) => [...cur, t]), []);

  const takesByTask = useMemo(() => {
    const m = new Map<string, TakeSummary[]>();
    for (const t of takes) m.set(t.task, [...(m.get(t.task) ?? []), t]);
    return m;
  }, [takes]);

  // Opening a task from the list adds a history entry, so the phone's Back button closes
  // the sheet. Moving between tasks inside the sheet replaces that entry.
  const pushed = useRef(false);
  const openTask = useCallback((id: string | null) => {
    const url = new URL(window.location.href);
    const current = url.searchParams.get("task");
    setOpenId(id);
    if (!id) {
      if (pushed.current) {
        pushed.current = false;
        window.history.back();
      } else {
        url.searchParams.delete("task");
        window.history.replaceState(null, "", url);
      }
      return;
    }
    url.searchParams.set("task", id);
    if (current) {
      window.history.replaceState(null, "", url);
    } else {
      window.history.pushState(null, "", url);
      pushed.current = true;
    }
  }, []);

  const requestJudge = useCallback(() => {
    resolver.current?.(null);
    setDialogOpen(true);
    return new Promise<Judge | null>((res) => {
      resolver.current = res;
    });
  }, []);

  const resolveDialog = useCallback((j: Judge | null) => {
    setDialogOpen(false);
    resolver.current?.(j);
    resolver.current = null;
  }, []);

  const value: CaptureState = {
    config,
    hydrated,
    judge,
    setJudge,
    requestJudge,
    judgeDialog: { open: dialogOpen, resolve: resolveDialog },
    takes,
    takesByTask,
    takesError,
    addTake,
    openId,
    openTask,
    order,
    setOrder,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
