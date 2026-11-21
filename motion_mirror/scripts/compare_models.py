"""Send the same clip + prompt to each Gemini model, compare motion understanding and latency."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai
from google.genai import types

MODELS = ["gemini-4-argon", "gemini-4-carbon", "gemini-4-hydrogen", "gemini-4-neon", "gemini-3.8-flash"]

PROMPT = """You are the motion-analysis module of a system that teaches a Unitree G1 humanoid robot
(29 DoF: legs, 3-DoF waist, 7-DoF arms, no fingers) to copy human movements from casual video.
The camera angle is unknown and uncalibrated; the person may be partially out of frame.

Analyze the person's movement and return JSON with:
- "summary": one sentence describing the movement.
- "category": one of "upper_body_standing", "whole_body_standing", "locomotion", "floor", "other".
- "feasible_for_standing_upper_body_policy": boolean, and "feasibility_reason".
- "active_limbs": which of ["left_arm","right_arm","waist","legs"] move (use the PERSON's left/right, not the image's).
- "camera": short description of the camera viewpoint relative to the person and what is occluded.
- "phases": list of {"start_s","end_s","description"} covering the clip.
- "keyframes": list of up to 8 {"t_s", "left_arm": {"shoulder_flexion_deg","shoulder_abduction_deg","elbow_flexion_deg"},
   "right_arm": {...same...}, "torso_lean_forward_deg"} at the most important moments.
   Angles: shoulder flexion 0 = arm down at side, 90 = arm pointing forward, 180 = arm straight up;
   abduction 0 = at side, 90 = arm out sideways horizontal; elbow 0 = straight, 90 = right angle.
- "confidence": 0-1.
Return only JSON."""


def ask(client, model, video_bytes):
    t = time.time()
    try:
        r = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"), PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return model, time.time() - t, json.loads(r.text)
    except Exception as e:  # noqa: BLE001
        return model, time.time() - t, {"error": str(e)[:300]}


def main():
    out_dir = Path("runs/model_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client()
    for clip in sys.argv[1:]:
        data = Path(clip).read_bytes()
        with ThreadPoolExecutor(len(MODELS)) as ex:
            results = list(ex.map(lambda m: ask(client, m, data), MODELS))
        name = Path(clip).stem.replace(" ", "_")
        (out_dir / f"{name}.json").write_text(json.dumps({m: {"latency_s": round(l, 1), "result": r} for m, l, r in results}, indent=2))
        print(f"\n=== {clip}")
        for m, lat, r in results:
            if "error" in r:
                print(f"  {m:20s} {lat:5.1f}s ERROR {r['error']}")
                continue
            print(f"  {m:20s} {lat:5.1f}s conf={r.get('confidence')} cat={r.get('category')} feasible={r.get('feasible_for_standing_upper_body_policy')}"
                  f" limbs={r.get('active_limbs')} kf={len(r.get('keyframes', []))}\n      {r.get('summary')}")


if __name__ == "__main__":
    main()
