"""Probe 1 (pre-registered settings in config.json). Logic: src/physprobe/run_probe.py."""
import json, os
from pathlib import Path
from physprobe.run_probe import run
cfg = json.loads((Path(__file__).resolve().parent / "config.json").read_text())
run(cfg, Path(os.environ.get("JOB_OUT", "out_probe1")), mock=os.environ.get("MOCK") == "1")
