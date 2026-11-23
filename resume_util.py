"""Shared resume / safe-stop helpers for the training scripts and run_all.py.

- ckpt_status(last_pt): "none" (no checkpoint), "finished" (Ultralytics strips the optimizer and sets
  epoch = -1 when a run ends, incl. early stopping) or "resumable" (training was interrupted).
- add_stop_callback(model): after each epoch's checkpoint is saved, if the file STOP exists in the project
  folder, training exits with StopRequested. last.pt is complete at that point, so nothing is lost.
"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
STOP_FILE = ROOT / "STOP"
EXIT_OOM = 3
EXIT_STOPPED = 4


class StopRequested(Exception):
    pass


def ckpt_status(last_pt: Path) -> str:
    last_pt = Path(last_pt)
    if not last_pt.exists():
        return "none"
    ck = torch.load(last_pt, map_location="cpu", weights_only=False)
    if ck.get("epoch", -1) == -1 or ck.get("optimizer") is None:
        return "finished"
    return "resumable"


def ckpt_epoch(last_pt: Path) -> int:
    """Last completed epoch (1-based) stored in a resumable checkpoint."""
    return torch.load(Path(last_pt), map_location="cpu", weights_only=False).get("epoch", -1) + 1


def add_stop_callback(model):
    def _check(trainer):
        # trainer.stop is True on the final / early-stopping epoch: let that finish normally.
        if STOP_FILE.exists() and not trainer.stop:
            print(f"\nSTOP file found -> stopping after epoch {trainer.epoch + 1} (last.pt saved, resumable).")
            raise StopRequested()
    model.add_callback("on_model_save", _check)
    return model


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
