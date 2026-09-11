#!/usr/bin/env python3
"""Run a trained checkpoint on a dataset, extract X-point peaks via NMS, write per-frame predicted CSVs."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

RC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(RC_ROOT))

from XPointMLTest import XPointDataset, UNet, cachedPgkylDataExists
from point_metrics import extract_peaks


# Base dirs, overridable via env; same RC_EXTRACT_DIR / RC_CACHE_BASE convention
# as run_hessian_and_build_cache.py so a launcher sets paths once for the whole pipeline.
EXTRACT_DIR = Path(os.environ.get("RC_EXTRACT_DIR", "/work/nvme/bfim/ssridhar6/mlReconnection2025"))
CACHE_BASE = Path(os.environ.get("RC_CACHE_BASE", "/work/nvme/bfim/ssridhar6/mlReconnection2025/cache"))
PKPM_V0_ROOT = Path(os.environ.get("RC_PKPM_V0_ROOT", "/work/nvme/bfim/cwsmith/mlReconnection2025/1024Res_v0"))

DATASET_CONFIG = {
    "PKPM": {
        "param_path": str(PKPM_V0_ROOT / "pkpm_2d_turb_p2-params.txt"),
        "cache_dir": str(PKPM_V0_ROOT / "cache04082025"),
    },
    "5M": {
        "param_path": str(EXTRACT_DIR / "5M" / "rt_5M_2d_turb_local-params.txt"),
        "cache_dir": str(CACHE_BASE / "5M"),
    },
    "10M": {
        "param_path": str(EXTRACT_DIR / "10M" / "10M" / "rt_10M_2d_turb_local-params.txt"),
        "cache_dir": str(CACHE_BASE / "10M"),
    },
    "PKPMv2": {
        "param_path": str(EXTRACT_DIR / "1024Res_v2" / "rt_pkpm_2d_turb_p1-params.txt"),
        "cache_dir": str(CACHE_BASE / "PKPMv2"),
    },
}


def write_pred_csv(out_path, rows, cols, confidences):
    """Write predicted-points CSV with row, col, confidence columns."""
    with open(out_path, "w") as f:
        f.write("row,col,confidence\n")
        for r, c, conf in zip(rows, cols, confidences):
            f.write(f"{int(r)},{int(c)},{float(conf):.6f}\n")


def discover_cached_frames(cache_dir):
    """Return sorted list of frame numbers that have a cached _psi.npy in cache_dir."""
    import re
    pattern = re.compile(r"^(\d+)_psi\.npy$")
    out = []
    for p in Path(cache_dir).glob("*_psi.npy"):
        m = pattern.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=["PKPM", "5M", "10M"],
                        choices=list(DATASET_CONFIG))
    parser.add_argument("--output-root", required=True, type=Path,
                        help="Base directory; predictions go to <output-root>/<dataset>/{N}_xpts.csv")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Confidence threshold for peak retention (default: 0.3)")
    parser.add_argument("--max-components", type=int, default=20000,
                        help="Bail out and return no peaks above this component count (default: 20000)")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.055)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = UNet(input_channels=4, base_channels=args.base_channels,
                 dropout_rate=args.dropout_rate).to(device)
    state_dict = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"NMS: threshold={args.threshold}, max_components={args.max_components}")

    with torch.no_grad():
        for ds_name in args.datasets:
            cfg = DATASET_CONFIG[ds_name]
            cache_dir = Path(cfg["cache_dir"])
            out_dir = args.output_root / ds_name
            out_dir.mkdir(parents=True, exist_ok=True)
            frames = discover_cached_frames(cache_dir)
            print(f"\n=== {ds_name}: {len(frames)} cached frames -> {out_dir} ===")

            t0 = time.time()
            dataset = XPointDataset(cfg["param_path"], frames,
                                    xptCacheDir=cache_dir, rotateAndReflect=False)
            print(f"  Loaded dataset in {time.time()-t0:.1f}s")

            n_predicted = 0
            t1 = time.time()
            for item in dataset:
                fnum = item["fnum"]
                all_torch = item["all"].unsqueeze(0).to(device)
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(all_torch)
                    probs = torch.sigmoid(logits)
                heatmap = probs[0, 0].float().cpu().numpy()
                rows, cols, confs = extract_peaks(heatmap, threshold=args.threshold,
                                                  max_components=args.max_components)
                write_pred_csv(out_dir / f"{fnum}_xpts.csv", rows, cols, confs)
                n_predicted += len(rows)
            print(f"  Wrote {len(frames)} CSV(s) in {time.time()-t1:.1f}s "
                  f"({n_predicted} predicted points total)")


if __name__ == "__main__":
    main()
