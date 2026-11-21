"use client";

import { useEffect, useState } from "react";
import { useCapture } from "./CaptureProvider";

export function TopBar() {
  const { hydrated, judge, setJudge, requestJudge, takes, openUpload } = useCapture();
  const [solid, setSolid] = useState(false);
  const n = takes.length;

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
        {hydrated && judge && n > 0 && (
          <a className="bar-count" href="#mine">
            <span className="num">{n}</span> {n === 1 ? "video" : "videos"}
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
        <button type="button" className="btn btn--accent btn--sm" onClick={() => openUpload()}>
          Upload
        </button>
      </div>
    </header>
  );
}
