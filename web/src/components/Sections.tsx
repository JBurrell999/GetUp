"use client";

import { useState } from "react";
import { countByTier, TASKS } from "@/lib/tasks";
import { useCapture } from "./CaptureProvider";
import { SimView, type SimInfo } from "./sim/SimView";

const fmt = (n: number, d = 1) => n.toFixed(d).replace(/\.0$/, "");

export function Hero() {
  const { simMode, setSimMode } = useCapture();
  const [info, setInfo] = useState<SimInfo | null>(null);
  const film = simMode === "film";

  return (
    <section className="hero" id="top" aria-labelledby="hero-title">
      <SimView mode={simMode} onReady={setInfo} />
      <div className="hero-shade" aria-hidden="true" />

      <div className="hero-copy">
        <p className="hero-kicker">Unitree G1, walking a copy of this room in MuJoCo</p>
        <h1 className="hero-title" id="hero-title">
          It can walk.
          <br />
          Teach it the rest.
        </h1>
        <p className="hero-lede">
          We&apos;re filming people here doing {TASKS.length} everyday things, from sitting down to getting up off the
          floor, so the robot can learn to copy them. Pick a task, film it on your phone, upload it below.
        </p>
        <div className="hero-actions">
          <a className="btn btn--accent" href="#tasks">
            Film a task
          </a>
          <button type="button" className="btn btn--quiet" aria-pressed={film} onClick={() => setSimMode(film ? "follow" : "film")}>
            {film ? "Back to the robot" : "Where to stand"}
          </button>
        </div>
      </div>

      <p className="hero-meta" aria-hidden={!info}>
        {info && (
          <>
            <span>{info.source}</span>
            <span>
              {fmt(info.lap)} s lap · {info.steps} steps · {fmt(info.speed, 2)} m/s
            </span>
            <span className="hero-drag">Drag to look around</span>
          </>
        )}
      </p>
    </section>
  );
}

export function Setup() {
  const { setSimMode } = useCapture();
  const show = () => {
    setSimMode("film");
    document.getElementById("top")?.scrollIntoView({ behavior: "smooth" });
  };
  return (
    <section className="setup" id="setup" aria-labelledby="setup-title">
      <div className="wrap">
        <h2 className="setup-title" id="setup-title">
          Before you film
        </h2>
        <p className="setup-line">
          Put your phone on something steady at <em>hip height</em>, <em>3 to 4 metres</em> away, turned about{" "}
          <em>45°</em> from straight on. Clear <em>about 4 × 3 m</em> in front of the stage, keep your whole body and any
          props in frame, and <em>don&apos;t move the phone</em> between takes.
        </p>
        <div className="setup-foot">
          <button type="button" className="link-btn" onClick={show}>
            Show me in the room
          </button>
          <p className="setup-safety">
            Floor and fall tasks: slowly, on a mat. Skip anything that isn&apos;t comfortable.
          </p>
        </div>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="foot">
      <div className="wrap foot-row">
        <span>GetUp</span>
        <span>
          {TASKS.length} tasks, {countByTier("core")} of them core. The room and robot above are the project&apos;s own MuJoCo
          scene.
        </span>
      </div>
    </footer>
  );
}
