"use client";

import { useEffect, useRef, useState } from "react";
import type { Anchor, Mode, SimEngine } from "./engine";

export interface SimInfo {
  source: string;
  lap: number;
  steps: number;
  speed: number;
}

interface Props {
  mode: Mode;
  onReady?: (info: SimInfo) => void;
}

// The 3D room with the walking G1. three.js loads only on the client, after the page is up.
export function SimView({ mode, onReady }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const engine = useRef<SimEngine | null>(null);
  const labels = useRef<Record<string, HTMLDivElement | null>>({});
  const [state, setState] = useState<"loading" | "ready" | "failed">("loading");
  const [progress, setProgress] = useState(0);
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  useEffect(() => {
    let disposed = false;
    let e: SimEngine | null = null;
    let ro: ResizeObserver | null = null;
    let io: IntersectionObserver | null = null;
    const visible = { view: true, page: !document.hidden };
    const sync = () => (visible.view && visible.page ? e?.start() : e?.stop());
    const onVis = () => {
      visible.page = !document.hidden;
      sync();
    };

    (async () => {
      try {
        const probe = document.createElement("canvas");
        if (!probe.getContext("webgl2")) throw new Error("no webgl2");
        const { SimEngine } = await import("./engine");
        if (disposed || !canvas.current || !wrap.current) return;
        e = new SimEngine(canvas.current, {
          reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
          coarse: window.matchMedia("(pointer: coarse)").matches,
          onAnchors: (as: Anchor[]) => {
            for (const a of as) {
              const el = labels.current[a.id];
              if (!el) continue;
              el.style.transform = `translate3d(${a.x}px, ${a.y}px, 0)`;
              el.dataset.on = a.visible ? "1" : "0";
            }
          },
        });
        await e.load((p) => !disposed && setProgress(p));
        if (disposed) return e.dispose();
        engine.current = e;
        const box = wrap.current.getBoundingClientRect();
        e.resize(box.width, box.height);
        ro = new ResizeObserver(([entry]) => e?.resize(entry.contentRect.width, entry.contentRect.height));
        ro.observe(wrap.current);
        io = new IntersectionObserver(([entry]) => {
          visible.view = entry.isIntersecting;
          sync();
        });
        io.observe(wrap.current);
        document.addEventListener("visibilitychange", onVis);
        setState("ready");
        onReadyRef.current?.(e.info());
        sync();
      } catch (err) {
        console.warn("[sim] 3D view unavailable:", err);
        if (!disposed) setState("failed");
      }
    })();

    return () => {
      disposed = true;
      ro?.disconnect();
      io?.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      e?.dispose();
      engine.current = null;
    };
  }, []);

  useEffect(() => {
    engine.current?.setMode(mode);
  }, [mode, state]);

  return (
    <div className="sim" ref={wrap} data-state={state}>
      <canvas ref={canvas} className="sim-canvas" aria-label="Unitree G1 walking a loop in the simulated event room" role="img" />
      <div className="sim-label" ref={(el) => void (labels.current.zone = el)} data-on="0">
        <div className="sim-label-box">
          <span>Stand here</span>
          <small>About 4 × 3 m clear, in front of the stage</small>
        </div>
      </div>
      <div className="sim-label" ref={(el) => void (labels.current.tripod = el)} data-on="0">
        <div className="sim-label-box">
          <span>Phone here</span>
          <small>Hip height, 3–4 m away, 45° off your front</small>
        </div>
      </div>
      {state === "loading" && (
        <div className="sim-loading" aria-live="polite">
          Loading the room <span className="num">{Math.round(progress * 100)}%</span>
        </div>
      )}
    </div>
  );
}
