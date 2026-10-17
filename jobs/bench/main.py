"""Bench: timing + first look, small version of probe 1. Logic: src/physprobe/run_probe.py."""
import json, os
from pathlib import Path
from physprobe.run_probe import run
cfg = json.loads((Path(__file__).resolve().parent / "config.json").read_text())
run(cfg, Path(os.environ.get("JOB_OUT", "out_bench")), mock=os.environ.get("MOCK") == "1")
