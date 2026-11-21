"use client";

import { useEffect, useState } from "react";
import { TASKS } from "@/lib/tasks";
import { useCapture } from "./CaptureProvider";

const CORE = TASKS.filter((t) => t.tier === "core");

export function TopBar() {
  const { hydrated, judge, setJudge, requestJudge, takesByTask } = useCapture();
  const [solid, setSolid] = useState(false);
  const done = CORE.filter((t) => takesByTask.has(t.id)).length;

  useEffect(() => {
    let raf = 0;
    const check = () => {
      raf = 0;
      setSolid(window.scrollY > window.innerHeight * 0.75);
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(check);
    };
    check();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <header className={solid ? "bar bar--solid" : "bar"}>
      <a className="wordmark" href="#top">
        GetUp
      </a>
      <div className="bar-right">
        {hydrated && judge && (
          <a className="bar-progress" href="#tasks" title="Core tasks you've filmed">
            <span className="num">
              {done}/{CORE.length}
            </span>
            <span className="bar-meter" aria-hidden="true">
              <span style={{ width: `${(done / CORE.length) * 100}%` }} />
            </span>
            <span className="sr-only">core tasks filmed</span>
          </a>
        )}
        {!hydrated ? null : judge ? (
          <span className="bar-who">
            {judge.name}
            <button
              type="button"
              onClick={() => {
                setJudge(null);
                requestJudge();
              }}
            >
              Not you?
            </button>
          </span>
        ) : (
          <button type="button" className="bar-who-btn" onClick={() => requestJudge()}>
            Add your name
          </button>
        )}
      </div>
    </header>
  );
}
