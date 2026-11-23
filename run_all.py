"""Run the whole approved FracAtlas pipeline in order. Resumable: stop at any time, run again to continue.

    python run_all.py            run / resume (skips steps already marked done in pipeline_state.json)
    python run_all.py --status   show which steps are done and the progress of the current training run
    python run_all.py --stop     ask the running pipeline to stop safely (creates the file STOP)

How resuming works
- Finished steps are recorded in pipeline_state.json and skipped on restart.
- Training steps continue from runs/<task>/<name>/weights/last.pt (YOLO(last.pt).train(resume=True)).
- STOP file: training stops right after the current epoch's checkpoint is saved; other steps stop at the
  next step boundary. Ctrl+C also works (at most the current epoch is lost).
- CUDA out-of-memory in a training step: the batch is halved, saved in the state file, and the run resumes.

Approved order (decisions already taken are fixed here):
  data prep -> damaged scan -> cross-split dedupe -> yolov8s-cls -> yolov8m-cls -> s+m ensemble ->
  val recall trade-off table -> user's choice (m at 80% recall) -> detect_clean (n) -> detect_clean_s (s) ->
  choose detector on VAL mAP50 -> ONE test evaluation -> analysis -> predict on 3 test images -> README.
  (The imgsz 640 / 150-epoch classifier step was skipped by the user: estimated 4-6 h.)
Test images are never used for training, tuning or selection; evaluate_test runs once (its step is then done).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from resume_util import EXIT_OOM, EXIT_STOPPED, STOP_FILE, ckpt_epoch, ckpt_status

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "pipeline_state.json"
LOG = ROOT / "logs_run_all.txt"
PY = sys.executable

S_CLS, M_CLS = "fracatlas_yolov8s_cls_512", "fracatlas_yolov8m_cls_512"
ENS = f"ensemble__{S_CLS}__{M_CLS}"
CHOSEN_CLS, CHOSEN_RECALL = M_CLS, "0.80"
DET_RUNS = ["fracatlas_yolov8n", "detect_clean", "detect_clean_s"]


def det_candidates():
    return [r for r in DET_RUNS if (ROOT / "runs" / "detect" / r / "weights" / "best.pt").exists()]


def evaluate_cmd():
    chosen = json.loads((ROOT / "runs" / "detect" / "detect_selection.json").read_text())["chosen"]
    return ["evaluate_test.py", "--det-run", chosen, "--cls-run", CHOSEN_CLS, "--val-candidates", S_CLS, ENS]


def predict_cmd():
    t = ROOT / "datasets" / "cls" / "test"
    fr = sorted((t / "fractured").glob("*.jpg"))
    nf = sorted((t / "non_fractured").glob("*.jpg"))
    return ["predict.py", str(fr[0]), str(fr[len(fr) // 2]), str(nf[0])]  # 2 fractured, 1 non-fractured


# name, command (list or callable), training info (task, run name, default batch) or None
STEPS = [
    ("prepare_data", ["prepare_data.py"], None),
    ("scan_damaged", ["scan_damaged.py"], None),
    ("dedupe_splits", ["dedupe_splits.py"], None),
    ("cls_s", ["train_classify.py", "--model", "yolov8s-cls.pt"], ("classify", S_CLS, 16)),
    ("cls_m", ["train_classify.py", "--model", "yolov8m-cls.pt", "--skip-build"], ("classify", M_CLS, 8)),
    ("cls_ensemble", ["ensemble.py", S_CLS, M_CLS], None),
    ("cls_recall_table", ["recall_table.py", S_CLS, M_CLS, ENS], None),
    ("cls_choose_threshold", ["set_threshold.py", CHOSEN_CLS, CHOSEN_RECALL], None),
    ("detect_clean", ["train_detect.py", "--name", "detect_clean"], ("detect", "detect_clean", 16)),
    ("detect_clean_s", ["train_detect.py", "--model", "yolov8s.pt", "--name", "detect_clean_s", "--patience", "20"],
     ("detect", "detect_clean_s", 8)),
    ("select_detect", lambda: ["select_detect.py", *det_candidates()], None),
    ("evaluate_test", evaluate_cmd, None),
    ("analyze", ["analyze.py"], None),
    ("predict", predict_cmd, None),
    ("fill_readme", ["fill_readme.py"], None),
    # Later attempts (yolov8m-cls 640 and a regularised retrain) did not beat the final model on test; they are
    # documented in FracAtlas_results.ipynb section 11 and their files were removed.
]


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"steps": {}}


def save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2))
    os.replace(tmp, STATE)  # atomic: a crash never leaves a half-written state file


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(args):
    """Run a step, streaming its output to the console and to logs_run_all.txt. Returns the exit code."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with open(LOG, "ab") as f:
        p = subprocess.Popen([PY, *args], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for chunk in iter(lambda: p.stdout.read1(4096), b""):
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                f.write(chunk)
            return p.wait()
        except KeyboardInterrupt:
            p.wait()
            raise


def training_progress(task, name):
    rc = ROOT / "runs" / task / name / "results.csv"
    if not rc.exists():
        return "not started"
    with open(rc, newline="") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return "started, no epoch finished yet"
    key = "metrics/mAP50(B)" if task == "detect" else "metrics/accuracy_top1"
    label = "val mAP50" if task == "detect" else "val top-1"
    best = max(rows, key=lambda r: float(r[key]))
    args = ROOT / "runs" / task / name / "args.yaml"
    ep = next((l.split(":")[1].strip() for l in args.read_text().splitlines() if l.startswith("epochs:")), "?")
    return (f"epoch {int(float(rows[-1]['epoch']))}/{ep}, best {label} {float(best[key]):.3f} "
            f"(epoch {int(float(best['epoch']))}), last.pt: {ckpt_status(ROOT / 'runs' / task / name / 'weights' / 'last.pt')}")


def status():
    st = load_state()["steps"]
    print(f"Pipeline state ({STATE.name}):")
    current = None
    for name, _, train in STEPS:
        s = st.get(name, {})
        mark = s.get("status", "pending")
        if mark != "done" and current is None:
            current = name
            mark = f"{mark}  <- next / current"
        extra = f"  (finished {s['finished']})" if s.get("finished") else ""
        print(f"  {name:<22} {mark}{extra}")
        if train and name == current:
            print(f"  {'':<22} {training_progress(train[0], train[1])}")
    print("All steps done." if current is None else f"Current step: {current}")
    if STOP_FILE.exists():
        print("STOP file present: the pipeline will stop at the next safe point.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    a = ap.parse_args()
    if a.status:
        return status()
    if a.stop:
        STOP_FILE.touch()
        return print("STOP file created: training stops after the current epoch, other steps at the next step.")

    if STOP_FILE.exists():
        STOP_FILE.unlink()  # a fresh start / resume clears an old stop request
    st = load_state()
    log("=== run_all.py started ===")
    for name, cmd, train in STEPS:
        s = st["steps"].setdefault(name, {})
        if s.get("status") == "done":
            log(f"skip {name} (done)")
            continue
        if STOP_FILE.exists():
            log(f"STOP file found before {name} -> stopped. Resume with: python run_all.py")
            return 0
        while True:
            args = cmd() if callable(cmd) else list(cmd)
            if train:
                task, run, default_batch = train
                batch = s.get("batch", default_batch)
                last = ROOT / "runs" / task / run / "weights" / "last.pt"
                cs = ckpt_status(last)
                if cs == "resumable":
                    log(f"{name}: resuming {run} from last.pt (epoch {ckpt_epoch(last)} done), batch {batch}")
                elif cs == "finished":
                    log(f"{name}: {run} training already finished")
                args += ["--batch", str(batch)]
            s.update(status="running", started=f"{datetime.now():%Y-%m-%d %H:%M:%S}")
            save_state(st)
            log(f"run {name}: python {' '.join(args)}")
            t0 = time.time()
            try:
                rc = run_cmd(args)
            except KeyboardInterrupt:
                s["status"] = "interrupted"
                save_state(st)
                log(f"{name} interrupted (Ctrl+C). Resume with: python run_all.py")
                return 130
            if rc == EXIT_OOM and train and batch > 1:
                s["batch"] = batch // 2
                save_state(st)
                log(f"{name}: CUDA out of memory at batch {batch} -> retrying with batch {batch // 2}")
                continue
            break
        if rc == EXIT_STOPPED:
            s["status"] = "stopped"
            save_state(st)
            log(f"{name} stopped safely via STOP file. Resume with: python run_all.py")
            return 0
        if rc != 0:
            s["status"] = f"failed (exit {rc})"
            save_state(st)
            log(f"{name} FAILED with exit code {rc}; see {LOG.name}. Fix the cause, then: python run_all.py")
            return rc
        s.update(status="done", finished=f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                 minutes=round((time.time() - t0) / 60, 1))
        save_state(st)
        log(f"done {name} ({s['minutes']} min)")
    log("=== all steps done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
