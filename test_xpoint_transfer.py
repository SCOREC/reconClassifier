#!/usr/bin/env python3
"""
Cross-domain inference: evaluate the best PKPM-trained model on 5M and 10M data.

This script:
  1. Extracts 5M.tgz and 10M.tgz (if not already extracted)
  2. Loads the best model checkpoint
  3. Runs evaluate_model_on_dataset on each dataset
  4. Saves per-dataset evaluation metrics

Usage (within SLURM job):
  python -u reconClassifier/test_xpoint_transfer.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# -- Monkey-patch: fix 5m/10m .gkyl component indexing bug in getData.py --
# The bug: getData.py passes comp=N to pg.data.GData which pre-selects the
# component, then tries data[..., N] which fails (array only has 1 element).
# Fix: drop the comp kwarg so GData returns all components.
import postgkyl as pg
_orig_GData_init = pg.data.GData.__init__
def _fixed_GData_init(self, *args, comp=None, **kwargs):
    _orig_GData_init(self, *args, **kwargs)
pg.data.GData.__init__ = _fixed_GData_init
# -- End monkey-patch --

# Add reconClassifier to path
RC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(RC_ROOT))

from XPointMLTest import XPointDataset, UNet
from eval_metrics import evaluate_model_on_dataset

# ── Configuration ──────────────────────────────────────────────────────
SOURCE_DIR = Path("/work/nvme/bfim/cwsmith/mlReconnection2025")
EXTRACT_DIR = Path("/work/nvme/bfim/ssridhar6/mlReconnection2025")
CACHE_BASE = EXTRACT_DIR / "cache"
BEST_MODEL = Path.home() / "mlReconnection/testdir_2026-04-02-13-23-05/checkpoints/best_model.pt"
OUTPUT_DIR = Path.home() / "mlReconnection/transfer_eval_results"

DATASETS = {
    "5M": {
        "tarball": SOURCE_DIR / "5M.tgz",
        "extract_subdir": EXTRACT_DIR / "5M",
        "param_file": "rt_5M_2d_turb_local-params.txt",
        "tar_prefix": "./",  # files are at root of tarball
    },
    "10M": {
        "tarball": SOURCE_DIR / "10M.tgz",
        "extract_subdir": EXTRACT_DIR / "10M",
        "param_file": "rt_10M_2d_turb_local-params.txt",
        "tar_prefix": "10M/",  # files are under 10M/ subdir
    },
}

# Model config must match training
BASE_CHANNELS = 64
DROPOUT_RATE = 0.055  # Trial #36 value (doesn't matter for eval, just architecture)


def extract_tarball(tarball_path, extract_dir, name):
    """Extract tarball if not already done."""
    param_candidates = list(extract_dir.glob("*-params.txt"))
    if param_candidates:
        print(f"  [{name}] Already extracted ({len(param_candidates)} param file(s) found)")
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{name}] Extracting {tarball_path} -> {extract_dir} ...")
    t0 = time.time()
    subprocess.run(
        ["tar", "xzf", str(tarball_path), "-C", str(extract_dir), "--strip-components=0"],
        check=True,
    )
    elapsed = time.time() - t0
    print(f"  [{name}] Extraction complete in {elapsed:.1f}s")


def discover_frames(extract_dir, param_file_name):
    """Discover available frame numbers from field files."""
    # Find the param file
    param_path = None
    for p in extract_dir.rglob(param_file_name):
        param_path = p
        break

    if param_path is None:
        # Try searching more broadly
        for p in extract_dir.rglob("*-params.txt"):
            param_path = p
            break

    if param_path is None:
        raise FileNotFoundError(f"No params file found in {extract_dir}")

    print(f"  Found param file: {param_path}")

    # Discover frame numbers from field files in same directory
    param_dir = param_path.parent
    field_files = sorted(param_dir.glob("*-field_*.gkyl"))
    frame_nums = set()
    for f in field_files:
        # Extract number from pattern like "...-field_42.gkyl"
        stem = f.stem  # e.g. "rt_5M_2d_turb_local-field_42"
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            frame_nums.add(int(parts[1]))

    frame_nums = sorted(frame_nums)
    print(f"  Found {len(frame_nums)} frames: {frame_nums[0]}-{frame_nums[-1]}")

    # Exclude frame 0 (often initial conditions, not interesting)
    frame_nums = [f for f in frame_nums if f > 0]

    return param_path, frame_nums


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model ─────────────────────────────────────────────────────
    print(f"\nLoading model from {BEST_MODEL}")
    model = UNet(input_channels=4, base_channels=BASE_CHANNELS, dropout_rate=DROPOUT_RATE).to(device)
    state_dict = torch.load(str(BEST_MODEL), map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")

    # ── Setup output ───────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    all_results = {}

    # ── Process each dataset ───────────────────────────────────────────
    for ds_name, ds_config in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"EVALUATING ON {ds_name} DATA")
        print(f"{'='*70}")

        # Extract
        extract_tarball(ds_config["tarball"], ds_config["extract_subdir"], ds_name)

        # Discover frames
        try:
            param_path, frame_nums = discover_frames(
                ds_config["extract_subdir"], ds_config["param_file"]
            )
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        # Load dataset using cache (pre-built by build_transfer_cache.py)
        cache_dir = CACHE_BASE / ds_name
        if not cache_dir.is_dir():
            print(f"  ERROR: Cache directory {cache_dir} not found.")
            print(f"  Run build_transfer_cache.py --dataset {ds_name} first.")
            continue

        print(f"  Loading {ds_name} dataset ({len(frame_nums)} frames, cache={cache_dir})...", flush=True)
        t0 = time.time()
        try:
            dataset = XPointDataset(
                str(param_path), frame_nums,
                xptCacheDir=cache_dir, rotateAndReflect=False,
            )
        except Exception as e:
            print(f"\n  ERROR loading dataset: {e}")
            import traceback
            traceback.print_exc()
            continue
        elapsed = time.time() - t0
        print(f"  All {len(frame_nums)} frames loaded in {elapsed:.1f}s")

        # Check grid size
        sample = dataset[0]
        grid_shape = sample["all"].shape
        print(f"  Grid shape: {grid_shape}  (channels, H, W)")

        # Evaluate
        print(f"  Running inference...")
        t0 = time.time()
        evaluator = evaluate_model_on_dataset(
            model, dataset, device,
            use_amp=use_amp, amp_dtype=amp_dtype, threshold=0.5,
        )
        elapsed = time.time() - t0

        evaluator.print_summary()

        # Save per-dataset results
        output_file = OUTPUT_DIR / f"eval_{ds_name.lower()}.json"
        evaluator.save_json(str(output_file))

        metrics = evaluator.get_global_metrics()
        metrics["inference_time_s"] = elapsed
        metrics["num_frames"] = len(frame_nums)
        metrics["grid_shape"] = list(grid_shape)
        all_results[ds_name] = metrics
        print(f"  Inference time: {elapsed:.1f}s ({elapsed/len(frame_nums):.2f}s/frame)")

        # Save partial results incrementally in case of timeout
        with open(OUTPUT_DIR / "transfer_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # ── Also re-evaluate on original PKPM validation set for comparison ──
    print(f"\n{'='*70}")
    print(f"RE-EVALUATING ON PKPM VALIDATION (baseline comparison)")
    print(f"{'='*70}")

    pkpm_param = "/work/nvme/bfim/cwsmith/mlReconnection2025/1024Res_v0/pkpm_2d_turb_p2-params.txt"
    pkpm_cache = "/work/nvme/bfim/cwsmith/mlReconnection2025/1024Res_v0/cache04082025"
    pkpm_val_frames = list(range(141, 150))

    print(f"  Loading PKPM validation ({len(pkpm_val_frames)} frames)...")
    t0 = time.time()
    pkpm_dataset = XPointDataset(
        pkpm_param, pkpm_val_frames,
        xptCacheDir=Path(pkpm_cache),
        rotateAndReflect=False,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    pkpm_evaluator = evaluate_model_on_dataset(
        model, pkpm_dataset, device,
        use_amp=use_amp, amp_dtype=amp_dtype, threshold=0.5,
    )
    elapsed = time.time() - t0
    pkpm_evaluator.print_summary()
    pkpm_evaluator.save_json(str(OUTPUT_DIR / "eval_pkpm_val.json"))

    pkpm_metrics = pkpm_evaluator.get_global_metrics()
    pkpm_metrics["inference_time_s"] = elapsed
    pkpm_metrics["num_frames"] = len(pkpm_val_frames)
    all_results["PKPM_val"] = pkpm_metrics

    # ── Summary comparison ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-DOMAIN TRANSFER SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':>12s}  {'F1':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'IoU':>7s}  {'Frames':>6s}  {'Grid':>12s}")
    print("-" * 70)
    for name, m in all_results.items():
        grid_str = "x".join(str(x) for x in m.get("grid_shape", [])) if "grid_shape" in m else "N/A"
        print(f"{name:>12s}  {m['f1_score']:7.4f}  {m['precision']:7.4f}  "
              f"{m['recall']:7.4f}  {m['iou']:7.4f}  {m.get('num_frames','?'):>6}  {grid_str:>12s}")
    print(f"{'='*70}")

    # Save combined summary
    summary_path = OUTPUT_DIR / "transfer_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
