#!/bin/sh
# Usage: ./mm gemini_loop videos/clip.mp4 --name demo   |   ./mm plan runs/plans/x.json --video out.mp4
mod="$1"; shift; cd "$(dirname "$0")" && PYTHONPATH=src exec uv run --env-file .env python -m "motion_mirror.$mod" "$@"
