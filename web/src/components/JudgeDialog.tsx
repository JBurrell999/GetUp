"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { makeJudgeId } from "@/lib/takes";
import { useCapture } from "./CaptureProvider";

export function JudgeDialog() {
  const { config, judgeDialog, setJudge } = useCapture();
  const ref = useRef<HTMLDialogElement>(null);
  const [name, setName] = useState("");
  const [code, setCode] = useState("");
  const [consent, setConsent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (judgeDialog.open && !d.open) {
      setError(null);
      d.showModal();
    }
    if (!judgeDialog.open && d.open) d.close();
  }, [judgeDialog.open]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    const n = name.trim().replace(/\s+/g, " ");
    if (!n) return setError("Add your name, or initials if you'd rather.");
    if (!consent) return setError("Tick the box to say your videos can be used.");
    if (config.codeRequired) {
      setBusy(true);
      const r = await fetch("/api/code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      })
        .then((res) => res.json() as Promise<{ ok: boolean }>)
        .catch(() => null);
      setBusy(false);
      if (!r) return setError("Couldn't reach the server. Check your connection.");
      if (!r.ok) return setError("That code isn't right. Ask the team for today's.");
    }
    const judge = { id: makeJudgeId(n), name: n, code: config.codeRequired ? code.trim() : undefined };
    setJudge(judge);
    judgeDialog.resolve(judge);
  }

  return (
    <dialog
      ref={ref}
      className="modal"
      aria-labelledby="judge-title"
      onClose={() => judgeDialog.open && judgeDialog.resolve(null)}
    >
      <form onSubmit={submit} noValidate>
        <p className="modal-kicker">Before your first take</p>
        <h2 className="modal-title" id="judge-title">
          Who&apos;s filming?
        </h2>
        <p className="modal-sub">
          Your takes are grouped under your name so the team can line them up with your body measurements.
        </p>

        <label className="field">
          <span className="field-k">Name</span>
          <input
            autoFocus
            autoComplete="name"
            value={name}
            maxLength={60}
            onChange={(e) => setName(e.target.value)}
            placeholder="First name or initials"
          />
        </label>

        {config.codeRequired && (
          <label className="field">
            <span className="field-k">Event code</span>
            <input
              value={code}
              autoCapitalize="characters"
              autoCorrect="off"
              spellCheck={false}
              onChange={(e) => setCode(e.target.value)}
              placeholder="Ask the team if you don't have it"
            />
          </label>
        )}

        <label className="consent">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
          <span>
            The GetUp team can use my videos to build and test a simulated humanoid.
            {config.access === "private" && " They're stored privately and not published."}
          </span>
        </label>

        {error && (
          <p className="modal-error" role="alert">
            {error}
          </p>
        )}

        <div className="modal-actions">
          <button type="submit" className="btn btn--accent" disabled={busy}>
            {busy ? "Checking…" : "Continue"}
          </button>
          <button type="button" className="link-btn" onClick={() => judgeDialog.resolve(null)}>
            Cancel
          </button>
        </div>
      </form>
    </dialog>
  );
}
