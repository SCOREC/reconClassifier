#!/usr/bin/env python3
"""Backfill {N}_xpts.csv files for any cache directory containing {N}_xpts.npy."""

import argparse
import re
from pathlib import Path

import numpy as np


def write_xpts_csv(cache_dir, fnum, xpts, optsMax, optsMin):
    """Write {fnum}_xpts.csv with row, col, class columns (class in X/Omax/Omin)."""
    csv_path = cache_dir / f"{fnum}_xpts.csv"
    with open(csv_path, "w") as f:
        f.write("row,col,class\n")
        for (r, c) in xpts:
            f.write(f"{int(r)},{int(c)},X\n")
        if optsMax is not None and len(optsMax) > 0:
            for (r, c) in optsMax:
                f.write(f"{int(r)},{int(c)},Omax\n")
        if optsMin is not None and len(optsMin) > 0:
            for (r, c) in optsMin:
                f.write(f"{int(r)},{int(c)},Omin\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path,
                        help="Source dir to read {N}_xpts.npy files from")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write CSVs (default: same as cache_dir)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rewrite CSVs that already exist (default: skip)")
    args = parser.parse_args()

    if not args.cache_dir.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {args.cache_dir}")
    output_dir = args.output_dir if args.output_dir is not None else args.cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    xpt_files = sorted(args.cache_dir.glob("*_xpts.npy"))
    pattern = re.compile(r"^(\d+)_xpts\.npy$")
    frames = []
    for f in xpt_files:
        m = pattern.match(f.name)
        if m:
            frames.append(int(m.group(1)))
    frames.sort()
    print(f"Source dir: {args.cache_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Found {len(frames)} cached frame(s).")

    n_written = n_skipped = 0
    for fnum in frames:
        csv_path = output_dir / f"{fnum}_xpts.csv"
        if csv_path.exists() and not args.overwrite:
            n_skipped += 1
            continue
        xpts = np.load(args.cache_dir / f"{fnum}_xpts.npy")
        optsMax_path = args.cache_dir / f"{fnum}_optsMax.npy"
        optsMin_path = args.cache_dir / f"{fnum}_optsMin.npy"
        optsMax = np.load(optsMax_path) if optsMax_path.exists() else None
        optsMin = np.load(optsMin_path) if optsMin_path.exists() else None
        write_xpts_csv(output_dir, fnum, xpts, optsMax, optsMin)
        n_written += 1

    print(f"Wrote {n_written} CSV(s), skipped {n_skipped} existing.")


if __name__ == "__main__":
    main()
