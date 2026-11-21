"""End-to-end stage-1 benchmark from raw phone videos (both phases, nothing cached), runs in parallel.
uv run scripts/benchmark_stage1.py        -> benchmarks/STAGE1_BENCHMARK.md"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOTTLE_FACTS = ["The person uses his RIGHT hand the whole time: his right arm (it comes from the LEFT side of the image) "
                "reaches ACROSS his body to the bottle, which sits on the table to the person's LEFT (the RIGHT side of "
                "the image). He drinks and puts it back with the same right hand.",
                "The person stays seated at the table the whole time."]
CHAIR_FACTS = ["The person lifts the chair with BOTH hands on the top of the backrest."]
RUNS = [("bench_bottle_facts_r1", "videos/IMG_6520.MOV", BOTTLE_FACTS), ("bench_bottle_facts_r2", "videos/IMG_6520.MOV", BOTTLE_FACTS),
        ("bench_bottle_nofacts", "videos/IMG_6520.MOV", []),
        ("bench_chair_facts_r1", "videos/IMG_6521.MOV", CHAIR_FACTS), ("bench_chair_facts_r2", "videos/IMG_6521.MOV", CHAIR_FACTS),
        ("bench_chair_nofacts", "videos/IMG_6521.MOV", [])]

procs = []
t0 = time.time()
for name, vid, facts in RUNS:
    cmd = [str(ROOT / "mm"), "mirror", vid, "--name", name] + sum([["--fact", f] for f in facts], [])
    procs.append((name, subprocess.Popen(cmd, cwd=ROOT, stdout=open(ROOT / f"runs/{name}.log", "w"), stderr=subprocess.STDOUT)))
for name, p in procs:
    p.wait()
wall = time.time() - t0

rows = []
for name, vid, facts in RUNS:
    f = ROOT / "runs/mirror2" / name / "summary.json"
    if not f.exists():
        rows.append(f"| {name} | FAILED (see runs/{name}.log) |||||||||")
        continue
    S = json.loads(f.read_text())
    tel = S["best_telemetry"]
    coll = tel.get("collisions", [])
    sustained = [c for c in coll if c["t_end"] - c["t_start"] > 0.1 and c["max_depth_cm"] > 2]
    grasp = max((x["grip_pos_err_m"] for g in tel.get("grasps", []) for x in g["detail"]), default=None)
    place = max((p["placement_err_m"] for p in tel.get("placements", [])), default=None)
    mr = S["motion_rounds"]
    rows.append(f"| {name} | {'yes' if facts else 'no'} | {S['env']['best_score']} ({S['env']['rounds']} rd) | "
                f"{' → '.join(str(r['score']) for r in mr)} | **{S['motion']['best_score']}** | {len(coll)} / {len(sustained)} | "
                f"{grasp if grasp is None else f'{grasp*100:.1f} cm'} | {place if place is None else f'{place*100:.1f} cm'} | "
                f"{sum(r['judge_s'] for r in mr)/len(mr):.0f} s | {S['total_s']:.0f} s |")

md = f"""# Stage 1 benchmark (Gemini mapping, kinematic replay)

Raw phone videos (`.MOV`, auto-converted), both phases from scratch, judge = {json.loads((ROOT/'runs/mirror2'/RUNS[0][0]/'summary.json').read_text())['judge_model'] if (ROOT/'runs/mirror2'/RUNS[0][0]/'summary.json').exists() else '?'}, 6 runs in parallel, wall clock {wall:.0f} s.

| run | facts | env score | motion scores by round | best | collisions all / sustained | grasp err | place err | judge s/round | total |
|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + """

*sustained* = longer than 0.1 s and deeper than 2 cm. Grasp / place error = worst hand / object distance from target.
"""
(ROOT / "benchmarks/STAGE1_BENCHMARK.md").write_text(md)
print(md)
