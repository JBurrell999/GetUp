"""A/B: which Gemini model combination maps videos best?   uv run --env-file .env scripts/benchmark_models.py

Configs (no facts, in the room, 2 repeats each, all in parallel):
  H      Hydrogen interprets + Hydrogen judges
  C>H    Carbon interprets (scene + first motion) + Hydrogen judges
  H+C    Hydrogen interprets + dual judge (Hydrogen and Carbon in parallel)
Every final result gets the SAME audit (Hydrogen and Carbon both score its side-by-side once with one fixed prompt), so
configs are compared on equal terms, plus the simulator's physical metrics.
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from motion_mirror.mirror import Gemini, quality, vpart  # noqa: E402

H, CB = "gemini-4-hydrogen", "gemini-4-carbon"
CONFIGS = {"H": ["--model", H], "C>H": ["--model", CB, "--judge-model", H], "H+C": ["--model", H, "--judge-model2", CB]}
CLIPS = {"bottle": "videos/IMG_6520.MOV", "chair": "videos/IMG_6521.MOV"}
REPEATS = 2
AUDIT = """This is ONE side-by-side video: LEFT = a human, RIGHT = a robot (1.3 m, in a different, fixed simulated room)
replaying what the human does, from the same camera. Ignore the robot's size, the background and visual style.
Rate 0-10 how faithfully the robot reproduces the human's actions: what is done, with which hand, the posture over
time, the timing, and what happens to the object. Return JSON: {"score": f, "main_issue": str}"""

runs = []
for cfg, args in CONFIGS.items():
    for clip, vid in CLIPS.items():
        for r in range(REPEATS):
            name = f"ab_{cfg.replace('>', 'to').replace('+', 'and')}_{clip}_r{r}"
            runs.append((cfg, clip, name, [str(ROOT / "mm"), "mirror", vid, "--name", name, "--attempts", "1", "--no-bench"] + args))
t0 = time.time()
procs = [(c, k, n, subprocess.Popen(cmd, cwd=ROOT, stdout=open(ROOT / f"runs/{n}.log", "w"), stderr=subprocess.STDOUT))
         for c, k, n, cmd in runs]
for *_, p in procs:
    p.wait()
wall = time.time() - t0

auditors = [Gemini(H), Gemini(CB)]


def audit(name):
    out = ROOT / "runs/mirror2" / name
    S = json.loads((out / "summary.json").read_text())
    sbs = out / "best_side_by_side.mp4"
    with ThreadPoolExecutor(2) as ex:
        js = list(ex.map(lambda g: g.ask([vpart(sbs, fps=3.0), AUDIT], low_res=True), auditors))
    return S, [float(j.get("score") or 0) for j in js], js[0].get("main_issue", "")


rows = {}
with ThreadPoolExecutor(6) as ex:
    res = dict(zip([n for _, _, n, _ in runs], ex.map(lambda x: audit(x[2]) if (ROOT / "runs/mirror2" / x[2] / "summary.json").exists() else None, runs)))
lines = []
for cfg, clip, name, _ in runs:
    r = res.get(name)
    if r is None:
        lines.append(f"| {cfg} | {clip} | FAILED | | | | | |")
        continue
    S, aud, issue = r
    tel = S["best_telemetry"]
    coll = tel.get("collisions", [])
    sust = sum(1 for c in coll if c["t_end"] - c["t_start"] > 0.1 and c["max_depth_cm"] > 2)
    grasp = max((x["grip_pos_err_m"] for g in tel.get("grasps", []) for x in g["detail"]), default=0)
    rows.setdefault(cfg, []).append((sum(aud) / 2, sust, S["total_s"]))
    lines.append(f"| {cfg} | {clip} | **{sum(aud)/2:.2f}** (H {aud[0]}, C {aud[1]}) | {S['motion']['best_score']} | {sust} | "
                 f"{grasp*100:.1f} cm | {S['total_s']:.0f} s | {issue[:90]} |")
summary = "\n".join(f"| {cfg} | {sum(a for a, _, _ in v)/len(v):.2f} | {sum(s for _, s, _ in v)/len(v):.1f} | {sum(t for _, _, t in v)/len(v):.0f} s |"
                    for cfg, v in rows.items())
md = f"""# Gemini model combinations (stage 1, no facts, in the room)

Wall clock for all {len(runs)} runs in parallel: {wall:.0f} s. Audit = mean of Hydrogen and Carbon scoring the final
side-by-side with one fixed prompt (same for every config).

| config | mean audit | mean sustained collisions | mean time per video |
|---|---|---|---|
{summary}

| config | clip | audit | own judge score | sustained coll. | grasp err | time | audit: main issue |
|---|---|---|---|---|---|---|---|
""" + "\n".join(lines)
(ROOT / "benchmarks/MODEL_AB.md").write_text(md)
print(md)
