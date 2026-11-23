"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import type { PublicConfig } from "@/lib/config";
import { isEnvId, type EnvId } from "@/lib/environments";
import type { TakeSummary } from "@/lib/takes";
import type { Mode } from "./sim/engine";
import { STATIC } from "@/lib/site";

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

  /** What this person has uploaded so far, oldest first. */
  takes: TakeSummary[];
  takesError: string | null;
  addTake: (t: TakeSummary) => void;

  /** The upload panel. `what` prefills the description (from an example), or leaves it empty. */
  uploadOpen: boolean;
  uploadWhat: string | null;
  openUpload: (what?: string) => void;
  closeUpload: () => void;

  /** What the 3D view at the top is showing: the robot, or where to film. */
  simMode: Mode;
  setSimMode: (m: Mode) => void;

  /** Where this person is filming. Chosen before the first upload and remembered on the device. */
  env: EnvId | null;
  setEnv: (e: EnvId) => void;
}

const Ctx = createContext<CaptureState | null>(null);

export function useCapture() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useCapture outside CaptureProvider");
  return c;
}

const JUDGE_KEY = "getup.judge";
const ENV_KEY = "getup.env";

export function CaptureProvider({ config, children }: { config: PublicConfig; children: ReactNode }) {
  const [hydrated, setHydrated] = useState(false);
  const [judge, setJudgeState] = useState<Judge | null>(null);
  const [takes, setTakes] = useState<TakeSummary[]>([]);
  const [takesError, setTakesError] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadWhat, setUploadWhat] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [simMode, setSimMode] = useState<Mode>("follow");
  const [env, setEnvState] = useState<EnvId | null>(null);
  const resolver = useRef<((j: Judge | null) => void) | null>(null);

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(JUDGE_KEY) ?? "null");
      if (saved?.id && saved?.name) setJudgeState(saved);
    } catch {}
    // ?env=kitchen (say, from a QR code in that room) preselects the place
    const fromUrl = new URLSearchParams(window.location.search).get("env");
    let savedEnv: string | null = null;
    try {
      savedEnv = localStorage.getItem(ENV_KEY);
    } catch {}
    if (isEnvId(fromUrl)) {
      setEnvState(fromUrl);
      try {
        localStorage.setItem(ENV_KEY, fromUrl);
      } catch {}
    } else if (isEnvId(savedEnv)) {
      setEnvState(savedEnv);
    }
    setHydrated(true);
    const onPop = () => {
      // the phone's Back button closes the upload panel instead of leaving the page
      if (!pushed.current) return;
      pushed.current = false;
      setUploadOpen(false);
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

  // Bumped to re-read the list while the robot is still copying something.
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    if (STATIC || !judge) return; // no uploads on the static site
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
      .catch(() => live && setTakesError("Couldn't load your earlier videos."));
    return () => {
      live = false;
    };
  }, [judge, refresh]);

  // The robot worker runs elsewhere, so check back every 15 s until every take has its result.
  const waiting = takes.some((t) => t.env && (!t.robot || t.robot.status === "processing"));
  useEffect(() => {
    if (!judge || !waiting) return;
    const id = setTimeout(() => setRefresh((n) => n + 1), 15_000);
    return () => clearTimeout(id);
  }, [judge, waiting, refresh]);

  const setEnv = useCallback((e: EnvId) => {
    setEnvState(e);
    try {
      localStorage.setItem(ENV_KEY, e);
    } catch {}
  }, []);

  const addTake = useCallback((t: TakeSummary) => setTakes((cur) => [...cur, t]), []);

  // Opening the upload panel adds a history entry, so the phone's Back button closes it.
  const pushed = useRef(false);
  const openUpload = useCallback((what?: string) => {
    setUploadWhat(what ?? null);
    setUploadOpen(true);
    if (!pushed.current) {
      window.history.pushState({ getupUpload: true }, "");
      pushed.current = true;
    }
  }, []);
  const closeUpload = useCallback(() => {
    setUploadOpen(false);
    if (pushed.current) {
      pushed.current = false;
      window.history.back();
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
    takesError,
    addTake,
    uploadOpen,
    uploadWhat,
    openUpload,
    closeUpload,
    simMode,
    setSimMode,
    env,
    setEnv,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
