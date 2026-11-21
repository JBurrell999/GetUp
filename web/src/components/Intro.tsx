import { SETUP_STEPS } from "@/lib/capture";
import { CATEGORIES, countByTier, TASKS } from "@/lib/tasks";
import { FloorPlan } from "./FloorPlan";

export function Hero() {
  const core = countByTier("core");
  return (
    <section className="hero" id="top">
      <div className="wrap hero-grid">
        <h1 className="display">
          Show a robot how people <em>move</em>
        </h1>
        <div className="hero-copy">
          <p className="lede">
            We&apos;re teaching a simulated Unitree G1 to sit down, get up, carry a box and {TASKS.length - 3} other
            everyday things. Film yourself doing them here in the room, one task per video. The sim tries to copy each
            take, and we score how close it gets.
          </p>
          <div className="hero-actions">
            <a className="btn btn--rec" href="#shots">
              Start with the {core} core tasks
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M8 3v9M4.5 8.5 8 12l3.5-3.5" />
              </svg>
            </a>
            <a className="link" href="#setup">
              Set up the camera first
            </a>
          </div>
        </div>
        <dl className="callsheet" aria-label="Session details">
          <div>
            <dt className="mono">Location</dt>
            <dd>Event room, the open aisle in front of the stage</dd>
          </div>
          <div>
            <dt className="mono">Robot</dt>
            <dd>Unitree G1, simulated in MuJoCo</dd>
          </div>
          <div>
            <dt className="mono">Camera</dt>
            <dd>One phone, fixed, landscape</dd>
          </div>
          <div>
            <dt className="mono">Tasks</dt>
            <dd>
              {TASKS.length} in {CATEGORIES.length} categories, {core} core
            </dd>
          </div>
          <div>
            <dt className="mono">Per video</dt>
            <dd>One task, start to finish</dd>
          </div>
        </dl>
      </div>
    </section>
  );
}

export function Setup() {
  return (
    <section className="setup" id="setup" aria-labelledby="setup-title">
      <div className="wrap setup-grid">
        <div className="setup-text">
          <p className="eyebrow mono">01 · Setup</p>
          <h2 className="h2" id="setup-title">
            Same spot, same camera, every take.
          </h2>
          <p className="setup-sub">
            Everything is measured from one fixed phone, so where it stands matters more than how good it is.
          </p>
          <ol className="steps">
            {SETUP_STEPS.map((s, i) => (
              <li key={s.title}>
                <span className="step-no mono">{String(i + 1).padStart(2, "0")}</span>
                <div>
                  <h3>{s.title}</h3>
                  <p>{s.body}</p>
                </div>
              </li>
            ))}
          </ol>
          <p className="safety">
            <strong>Floor and fall tasks:</strong> go slowly, use a mat, and skip anything that isn&apos;t comfortable.
            A skipped task is fine; a hurt judge isn&apos;t.
          </p>
        </div>
        <figure className="plan-fig">
          <FloorPlan />
          <figcaption className="mono">
            <span>Fig. 1</span> The room as the sim models it. Stand in the marked area, camera where shown.
          </figcaption>
        </figure>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="foot">
      <div className="wrap foot-grid mono">
        <span>GetUp</span>
        <span>Tasks from tasks.json · room from the MuJoCo scene</span>
        <a href="#top">Back to top ↑</a>
      </div>
    </footer>
  );
}
