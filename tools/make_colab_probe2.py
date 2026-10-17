"""Build notebooks/probe2_colab.ipynb with the code embedded (small; no repo access needed)."""
import base64, io, json, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for base in (ROOT / "src", ROOT / "jobs" / "probe2"):
        for p in base.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(ROOT))
payload = base64.b64encode(buf.getvalue()).decode()

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}

cells = [
md("""# ZUNA follow-up test (probe 2)

**Before you start:** Runtime → Change runtime type → **T4 GPU** → Save. Then **Runtime → Run all**.

- Allow Google Drive when asked: progress is saved there, so if Colab disconnects just **Run all** again and finished parts are skipped.
- Expected time: about **1 hour per part, 2 parts** (~2 h total). It prints "still running" every few minutes.
- At the end a file **`probe2_results.zip`** downloads. Attach it in the chat."""),
code("""# 1. Check the GPU and install ZUNA (2-3 min)
import subprocess, sys
gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "NONE - set Runtime > Change runtime type > T4 GPU, then Run all again")
assert gpu, "No GPU"
subprocess.run(f"{sys.executable} -m pip install -q zuna mne", shell=True, check=True)
print("ZUNA installed")"""),
code("""# 2. Save progress to Google Drive (allow access when asked)
from pathlib import Path
try:
    from google.colab import drive
    drive.mount("/content/drive")
    SAVE = Path("/content/drive/MyDrive/eegpp_probe2")
except Exception as e:
    print("Drive not available, saving locally only:", e)
    SAVE = Path("/content/eegpp_probe2")
SAVE.mkdir(parents=True, exist_ok=True)
print("Results folder:", SAVE)"""),
code(f"""# 3. Unpack the experiment code (embedded, ~{len(payload)//1024} KB)
import base64, io, zipfile, sys, os
PAYLOAD = "{payload}"
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall("/content/eegpp")
sys.path.insert(0, "/content/eegpp/src")
os.environ["PROBE_WORK"] = "/content/work"
print("code ready")"""),
code("""# 4. Run part A then part B (each ~1 h). Finished parts are skipped on a re-run.
import json, time
from pathlib import Path
from physprobe.run_probe import run
for part in ("A", "B"):
    out = SAVE / f"part_{part}"
    if (out / "log.json").exists() and json.loads((out / "log.json").read_text()).get("ok"):
        print(f"part {part}: already done, skipping"); continue
    cfg = json.loads(Path(f"/content/eegpp/jobs/probe2/config_{part}.json").read_text())
    print(f"part {part}: starting ({cfg['n_trials']} recordings x {cfg['copies']} samples)", flush=True)
    t0 = time.time()
    log = run(cfg, out, mock=False)
    print(f"part {part}: {'done' if log.get('ok') else 'FAILED'} in {(time.time() - t0) / 60:.0f} min")
    if not log.get("ok"):
        print(log.get("error", "")[-2000:]); break"""),
code("""# 5. Download the results (attach probe2_results.zip in the chat)
import shutil
zip_path = shutil.make_archive("/content/probe2_results", "zip", SAVE)
try:
    from google.colab import files
    files.download(zip_path)
except Exception:
    print("Download manually:", zip_path)"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "probe2_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out, out.stat().st_size // 1024, "KB")
