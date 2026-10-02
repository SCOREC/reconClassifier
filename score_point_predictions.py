#!/usr/bin/env python3
"""Compare predicted-points CSVs against ground-truth-points CSVs at the point level."""

import argparse
import json
import os
from pathlib import Path

from point_metrics import (
    evaluate_point_predictions,
    load_xpts_csvs_for_frames,
)


# Ground-truth CSVs live under the shared cache root; override via RC_CACHE_BASE
# (same convention as run_hessian_and_build_cache.py / predict_points.py).
CACHE_BASE = Path(os.environ.get("RC_CACHE_BASE", "/work/nvme/bfim/ssridhar6/mlReconnection2025/cache"))

GT_CSV_DIRS = {
    "PKPM": str(CACHE_BASE / "PKPM"),
    "5M":   str(CACHE_BASE / "5M"),
    "10M":  str(CACHE_BASE / "10M"),
    "PKPMv2": str(CACHE_BASE / "PKPMv2"),
}


def load_pred_csvs_with_confidence(pred_dir):
    """Load predicted xpts CSVs (row, col, confidence) into {frame_id: coords, frame_id: conf} dicts."""
    import re, csv
    import numpy as np
    pattern = re.compile(r"^(\d+)_xpts\.csv$")
    pred_by_frame = {}
    conf_by_frame = {}
    for path in sorted(Path(pred_dir).glob("*_xpts.csv")):
        m = pattern.match(path.name)
        if not m:
            continue
        fid = int(m.group(1))
        rows, confs = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                rows.append((int(row["row"]), int(row["col"])))
                confs.append(float(row.get("confidence", 1.0)))
        if rows:
            pred_by_frame[fid] = np.asarray(rows, dtype=int)
            conf_by_frame[fid] = np.asarray(confs, dtype=float)
        else:
            pred_by_frame[fid] = np.zeros((0, 2), dtype=int)
            conf_by_frame[fid] = np.zeros((0,), dtype=float)
    return pred_by_frame, conf_by_frame


def score_one(pred_dir, gt_dir, radius):
    pred_by_frame, conf_by_frame = load_pred_csvs_with_confidence(pred_dir)
    frame_ids = sorted(pred_by_frame.keys())
    gt_by_frame = load_xpts_csvs_for_frames(gt_dir, frame_ids, classes=("X",))
    return evaluate_point_predictions(pred_by_frame, gt_by_frame,
                                      radius=radius,
                                      pred_confidence_by_frame=conf_by_frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", required=True, type=Path,
                        help="Directory containing subdirs per dataset (PKPM/, 5M/, 10M/)")
    parser.add_argument("--datasets", nargs="+", default=["PKPM", "5M", "10M"],
                        choices=list(GT_CSV_DIRS))
    parser.add_argument("--radius", type=float, default=5.0,
                        help="Matching radius in pixels (default: 5.0)")
    parser.add_argument("--label", default="model",
                        help="Label to print in the summary row (e.g. 'PKPM-trained')")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Optional: write full results as JSON")
    args = parser.parse_args()

    print(f"\n{'='*78}")
    print(f"POINT-LEVEL F1  (label={args.label}, radius={args.radius:.1f} px)")
    print(f"{'='*78}")
    print(f"{'Dataset':<8} {'F1':>7} {'Prec':>7} {'Rec':>7} {'TP':>6} {'FP':>6} {'FN':>6} {'Frames':>7}")
    print("-" * 78)

    all_results = {}
    for ds in args.datasets:
        pred_dir = args.predictions_root / ds
        if not pred_dir.is_dir():
            print(f"{ds:<8} (no predictions directory at {pred_dir})")
            continue
        result = score_one(pred_dir, GT_CSV_DIRS[ds], radius=args.radius)
        g = result["global"]
        n_frames = len(result["per_frame"])
        print(f"{ds:<8} {g['f1']:>7.4f} {g['precision']:>7.4f} {g['recall']:>7.4f} "
              f"{g['tp']:>6d} {g['fp']:>6d} {g['fn']:>6d} {n_frames:>7d}")
        all_results[ds] = result

    print("=" * 78)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"label": args.label, "radius": args.radius,
                       "results": all_results}, f, indent=2)
        print(f"Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
