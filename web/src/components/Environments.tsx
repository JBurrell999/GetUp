"use client";

import { useState } from "react";
import { ENV_BY_ID, ENVIRONMENTS, type EnvId, type Environment } from "@/lib/environments";
import { useCapture } from "./CaptureProvider";

export function Check() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3.5 8.5l3 3 6-7" />
    </svg>
  );
}

function Card({ env, on, onPick }: { env: Environment; on: boolean; onPick: () => void }) {
  const [view, setView] = useState(0);
  const shot = env.shots[view];
  return (
    <li className="env-card" data-on={on ? "1" : "0"}>
      <img
        className="env-shot"
        src={shot.src}
        srcSet={`${shot.small} 640w, ${shot.src} 1280w`}
        sizes="(max-width: 960px) 86vw, 33vw"
        alt={shot.alt}
        width={1280}
        height={720}
        loading="lazy"
        decoding="async"
      />
      <div className="env-thumbs" role="group" aria-label={`More views of the ${env.name.toLowerCase()}`}>
        {env.shots.map((s, k) => (
          <button key={s.src} type="button" aria-pressed={k === view} aria-label={`View ${k + 1}`} onClick={() => setView(k)}>
            <img src={s.small} alt="" width={640} height={360} loading="lazy" decoding="async" />
          </button>
        ))}
      </div>
      <div className="env-body">
        <p className="env-where">{env.where}</p>
        <h3 className="env-name">{env.name}</h3>
        <p className="env-blurb">{env.blurb}</p>
        <ul className="env-facts">
          {env.facts.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>
        {on ? (
          <p className="env-on">
            <Check />
            You&apos;re filming here
          </p>
        ) : (
          <button type="button" className="btn btn--quiet env-choose" onClick={onPick}>
            I&apos;m filming here
          </button>
        )}
      </div>
    </li>
  );
}

/** The page section: the three places, with screenshots. */
export function Environments() {
  const { hydrated, env, setEnv, openUpload } = useCapture();
  const chosen = hydrated && env ? ENV_BY_ID.get(env) : undefined;
  return (
    <section className="envs" id="where" aria-labelledby="envs-title">
      <div className="wrap">
        <div className="envs-head">
          <h2 className="envs-title" id="envs-title">
            Where are you filming?
          </h2>
          <p className="envs-lede">
            Pick the place most like where you are before you upload. The robot copies each video inside a simulated copy
            of that place, so the chairs, tables and boxes you use have a match.
          </p>
        </div>
        <ul className="envs-grid">
          {ENVIRONMENTS.map((e) => (
            <Card key={e.id} env={e} on={chosen?.id === e.id} onPick={() => setEnv(e.id)} />
          ))}
        </ul>
        <div className="envs-next" aria-live="polite">
          {chosen ? (
            <>
              <p>
                Your videos will be matched to the <strong>{chosen.name.toLowerCase()}</strong>. You can change it any time.
              </p>
              <button type="button" className="btn btn--accent" onClick={() => openUpload()}>
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M8 12.5V3.5M4.5 7L8 3.5 11.5 7M2.5 13.5h11" />
                </svg>
                Upload a video
              </button>
            </>
          ) : (
            <p>Choose one to continue. You can change it later, even for a single video.</p>
          )}
        </div>
      </div>
    </section>
  );
}

/** The compact version, for the upload picker and the task panel. */
export function EnvPick({ value, onPick }: { value: EnvId | null; onPick: (id: EnvId) => void }) {
  return (
    <ul className="envpick">
      {ENVIRONMENTS.map((e) => (
        <li key={e.id}>
          <button type="button" aria-pressed={value === e.id} onClick={() => onPick(e.id)}>
            <img src={e.shots[0].small} alt="" width={640} height={360} loading="lazy" decoding="async" />
            <span className="envpick-text">
              <span className="envpick-where">{e.where}</span>
              <span className="envpick-name">{e.name}</span>
              <span className="envpick-blurb">{e.blurb}</span>
            </span>
            <span className="envpick-mark">{value === e.id && <Check />}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
