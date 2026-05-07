#!/usr/bin/env python3
"""
Build X-point cache for 5M/10M datasets.

This pre-computes and caches the X-point finder results so that
test_xpoint_transfer.py can load all frames quickly.

Usage:
  python build_transfer_cache.py --dataset 5M --start 1 --end 75
  python build_transfer_cache.py --dataset 10M --start 76 --end 150
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

# -- Monkey-patch: fix 5m/10m .gkyl component indexing bug in getData.py --
import postgkyl as pg
_orig_GData_init = pg.data.GData.__init__
def _fixed_GData_init(self, *args, comp=None, **kwargs):
    _orig_GData_init(self, *args, **kwargs)
pg.data.GData.__init__ = _fixed_GData_init
# -- End monkey-patch --

RC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(RC_ROOT))

from XPointMLTest import XPointDataset

EXTRACT_DIR = Path(os.environ.get(
    "RC_EXTRACT_DIR",
    "/work/nvme/bfim/ssridhar6/mlReconnection2025",
))
CACHE_BASE = Path(os.environ.get(
    "RC_CACHE_BASE",
    "/work/nvme/bfim/ssridhar6/mlReconnection2025/cache",
))

DATASETS = {
    "5M": {
        "extract_subdir": EXTRACT_DIR / "5M",
        "param_file": "rt_5M_2d_turb_local-params.txt",
    },
    "10M": {
        "extract_subdir": EXTRACT_DIR / "10M",
        "param_file": "rt_10M_2d_turb_local-params.txt",
    },
}


def find_param_file(extract_dir, param_file_name):
    for p in extract_dir.rglob(param_file_name):
        return p
    for p in extract_dir.rglob("*-params.txt"):
        return p
    raise FileNotFoundError(f"No params file found in {extract_dir}")


def discover_all_frames(extract_dir):
    """Find all frame numbers from field files."""
    field_files = list(extract_dir.rglob("*-field_*.gkyl"))
    frame_nums = set()
    for f in field_files:
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            frame_nums.add(int(parts[1]))
    return sorted(f for f in frame_nums if f > 0)


def _process_frame(task):
    """Worker function: process a single frame. Returns (frame_num, elapsed)."""
    param_path, fnum, cache_dir = task
    t0 = time.time()
    dataset = XPointDataset(
        str(param_path),
        [fnum],
        xptCacheDir=cache_dir,
        rotateAndReflect=False,
    )
    elapsed = time.time() - t0
    return fnum, elapsed


def main():
    parser = argparse.ArgumentParser(description="Build X-point cache for transfer datasets")
    parser.add_argument("--dataset", required=True, choices=["5M", "10M"])
    parser.add_argument("--start", type=int, default=None, help="First frame index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Last frame index (inclusive)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    cache_dir = CACHE_BASE / args.dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    param_path = find_param_file(ds["extract_subdir"], ds["param_file"])
    print(f"Dataset: {args.dataset}")
    print(f"Param file: {param_path}")
    print(f"Cache dir: {cache_dir}")
    print(f"Workers: {args.workers}")

    all_frames = discover_all_frames(ds["extract_subdir"])
    print(f"Total frames available: {len(all_frames)} ({all_frames[0]}-{all_frames[-1]})")

    # Apply range filter
    start = args.start if args.start is not None else all_frames[0]
    end = args.end if args.end is not None else all_frames[-1]
    frames = [f for f in all_frames if start <= f <= end]
    print(f"Processing frames {start}-{end}: {len(frames)} frames")

    # Check which frames are already cached
    from XPointMLTest import cachedPgkylDataExists
    uncached = [f for f in frames if not cachedPgkylDataExists(cache_dir, f, "psi")]
    cached = len(frames) - len(uncached)
    print(f"Already cached: {cached}, need to compute: {len(uncached)}")

    if not uncached:
        print("All frames already cached!")
        return

    tasks = [(param_path, fnum, cache_dir) for fnum in uncached]

    if args.workers <= 1:
        # Sequential mode (original behavior)
        total_time = 0
        for i, task in enumerate(tasks):
            fnum = task[1]
            print(f"\n[{i+1}/{len(uncached)}] Frame {fnum}...", flush=True)
            _, elapsed = _process_frame(task)
            total_time += elapsed
            avg = total_time / (i + 1)
            remaining = avg * (len(uncached) - i - 1)
            print(f"    Done in {elapsed:.1f}s | Avg: {avg:.1f}s/frame | "
                  f"ETA: {remaining/3600:.1f}h remaining", flush=True)
    else:
        # Parallel mode
        print(f"\nStarting parallel processing with {args.workers} workers...", flush=True)
        total_time = 0
        completed = 0
        wall_start = time.time()
        with mp.Pool(processes=args.workers) as pool:
            for fnum, elapsed in pool.imap_unordered(_process_frame, tasks):
                completed += 1
                total_time += elapsed
                wall_elapsed = time.time() - wall_start
                avg_wall = wall_elapsed / completed
                remaining = avg_wall * (len(uncached) - completed)
                print(f"[{completed}/{len(uncached)}] Frame {fnum} done in {elapsed:.1f}s | "
                      f"Wall avg: {avg_wall:.1f}s/frame | "
                      f"ETA: {remaining/60:.1f}m remaining", flush=True)
        total_time = time.time() - wall_start

    print(f"\nCache building complete! Wall time: {total_time/60:.1f}m ({total_time/3600:.1f}h)")
    print(f"Cached {len(uncached)} frames to {cache_dir}")


if __name__ == "__main__":
    main()
