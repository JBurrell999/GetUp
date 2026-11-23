"use client";

import { useState } from "react";
import { useCapture } from "./CaptureProvider";
import { SimView, type SimInfo } from "./sim/SimView";
import { REPO_URL, STATIC } from "@/lib/site";

const fmt = (n: number, d = 1) => n.toFixed(d).replace(/\.0$/, "");

export function Hero() {
  const { simMode, setSimMode, openUpload } = useCapture();
  const [info, setInfo] = useState<SimInfo | null>(null);
  const film = simMode === "film";

  return (
    <section className="hero" id="top" aria-labelledby="hero-title">
      <SimView mode={simMode} onReady={setInfo} />
      <div className="hero-shade" aria-hidden="true" />

      <div className="hero-copy">
        <p className="hero-kicker">Unitree G1, walking a copy of the event room in MuJoCo</p>
        <h1 className="hero-title" id="hero-title">
          It can walk.
          <br />
          Teach it the rest.
        </h1>
        {STATIC ? (
          <p className="hero-lede">
            Film someone doing an everyday task on a phone. Gemini works out what&apos;s happening, and a Unitree humanoid in
            MuJoCo does the same thing, seen from the same camera. Built in one day at the Gemini 4 NYC hackathon.
          </p>
        ) : (
          <p className="hero-lede">
            We&apos;re collecting videos of people doing everyday things, from moving a chair to getting up off the floor,
            so the robot can learn to copy them. Film anything you like on your phone and upload it here.
          </p>
        )}
        <div className="hero-actions">
          {STATIC ? (
            <a className="btn btn--accent" href="#mirror">
              See the results
            </a>
          ) : (
            <button type="button" className="btn btn--accent" onClick={() => openUpload()}>
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M8 12.5V3.5M4.5 7L8 3.5 11.5 7M2.5 13.5h11" />
              </svg>
              Upload a video
            </button>
          )}
          <button type="button" className="btn btn--quiet" aria-pressed={film} onClick={() => setSimMode(film ? "follow" : "film")}>
            {film ? "Back to the robot" : "Where to stand"}
          </button>
        </div>
        {STATIC ? (
          <a className="hero-browse" href={REPO_URL}>
            Code and how it works on GitHub
          </a>
        ) : (
          <a className="hero-browse" href="#ideas">
            Need an idea? See a few examples
          </a>
        )}
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
  const { setSimMode, hydrated, env } = useCapture();
  // the stage and the "show me" view only make sense in the event room the 3D view shows
  const atEvent = !hydrated || !env || env === "hackathon";
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
          <em>45°</em> from straight on. Clear <em>about 4 × 3 m</em> {atEvent ? "in front of the stage" : "of open floor"}, keep
          your whole body and any props in frame, and <em>don&apos;t move the phone</em> between takes.
        </p>
        <div className="setup-foot">
          {atEvent && (
            <button type="button" className="link-btn" onClick={show}>
              Show me in the room
            </button>
          )}
          <p className="setup-safety">
            Floor work, falls and flips: slowly, on a mat. Skip anything that isn&apos;t comfortable.
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
        <span>Film anything. The room and robot above are the project&apos;s own MuJoCo scene.</span>
      </div>
    </footer>
  );
}
