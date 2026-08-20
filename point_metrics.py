"""Point-level precision/recall/F1 for X-point detection on coordinate lists."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.ndimage import maximum as cc_maximum


# PEAK EXTRACTION
def extract_peaks(heatmap, threshold=0.3, max_components=20000):
    """Extract one (row, col, confidence) peak per connected component above threshold (max-valued pixel in each).

    Ties within a component resolve to the lowest row-major index, matching
    np.argmax. Returns empty arrays if the component count exceeds
    max_components, which signals a speckled heatmap from an undertrained
    model rather than real peaks.
    """
    empty = (np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=float))
    above = heatmap > threshold
    if not above.any():
        return empty

    labels, n = cc_label(above)
    if max_components is not None and n > max_components:
        return empty

    # Per-component maxima, then the first pixel attaining each one.
    maxima = np.atleast_1d(cc_maximum(heatmap, labels, index=np.arange(1, n + 1)))
    lab_flat = labels.ravel()
    at_max = np.flatnonzero((lab_flat > 0) & (heatmap.ravel() == maxima[lab_flat - 1]))
    # flatnonzero is ascending, so a stable sort by label keeps the lowest
    # flat index first within each component.
    first = at_max[np.argsort(lab_flat[at_max], kind="stable")]
    _, starts = np.unique(lab_flat[first], return_index=True)
    peak_flat = first[starts]

    rows, cols = np.unravel_index(peak_flat, heatmap.shape)
    confs = heatmap[rows, cols].astype(float)
    return rows, cols, confs


# CSV I/O
def load_xpts_csv(csv_path: str | Path,
                  classes: Sequence[str] = ("X",)) -> np.ndarray:
    """Load (row, col) coordinates of the given classes from a per-frame xpts CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"xpts CSV not found: {csv_path}")

    kept = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["class"] in classes:
                kept.append((int(row["row"]), int(row["col"])))

    if not kept:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(kept, dtype=int)


# SPATIAL INDEX
class _GridIndex:
    """Bucket points into cells so nearest-neighbor lookups touch only the 3x3 cell neighborhood."""

    def __init__(self, points: np.ndarray, cell_size: float):
        self.cell_size = float(cell_size)
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, (r, c) in enumerate(points):
            self.buckets[(int(r // cell_size), int(c // cell_size))].append(i)
        self.points = points

    def candidates(self, r: float, c: float) -> list[int]:
        """Return indices of points in the query cell and its 8 neighbors."""
        cr, cc = int(r // self.cell_size), int(c // self.cell_size)
        out: list[int] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                bucket = self.buckets.get((cr + dr, cc + dc))
                if bucket:
                    out.extend(bucket)
        return out


# PER-FRAME MATCHING
def match_points(pred_pts: np.ndarray,
                 gt_pts: np.ndarray,
                 radius: float = 5.0,
                 pred_confidence: np.ndarray | None = None,
                 ) -> dict:
    """Greedy 1-to-1 nearest-neighbor match within `radius`; returns tp/fp/fn, P/R/F1, and per-pred matched gt index."""
    pred_pts = np.asarray(pred_pts).reshape(-1, 2)
    gt_pts = np.asarray(gt_pts).reshape(-1, 2)
    P, G = len(pred_pts), len(gt_pts)

    if P == 0 and G == 0:
        return {"tp": 0, "fp": 0, "fn": 0,
                "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "pred_match": np.zeros(0, dtype=int)}
    if P == 0:
        return {"tp": 0, "fp": 0, "fn": G,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "pred_match": np.zeros(0, dtype=int)}
    if G == 0:
        return {"tp": 0, "fp": P, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "pred_match": -np.ones(P, dtype=int)}

    order = (np.argsort(-np.asarray(pred_confidence))
             if pred_confidence is not None
             else np.arange(P))

    index = _GridIndex(gt_pts, cell_size=radius)
    gt_taken = np.zeros(G, dtype=bool)
    pred_match = -np.ones(P, dtype=int)
    r2 = radius * radius

    for i in order:
        pr, pc = pred_pts[i]
        best_j, best_d2 = -1, r2 + 1e-9
        for j in index.candidates(pr, pc):
            if gt_taken[j]:
                continue
            dr = float(pr - gt_pts[j, 0])
            dc = float(pc - gt_pts[j, 1])
            d2 = dr * dr + dc * dc
            if d2 <= r2 and d2 < best_d2:
                best_d2, best_j = d2, j
        if best_j >= 0:
            pred_match[i] = best_j
            gt_taken[best_j] = True

    tp = int((pred_match >= 0).sum())
    fp = P - tp
    fn = G - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "pred_match": pred_match}


# MULTI-FRAME AGGREGATION
def evaluate_point_predictions(pred_by_frame: Mapping[int, np.ndarray],
                               gt_by_frame: Mapping[int, np.ndarray],
                               radius: float = 5.0,
                               pred_confidence_by_frame: Mapping[int, np.ndarray] | None = None,
                               ) -> dict:
    """Aggregate `match_points` across frames into global (summed tp/fp/fn) and per-frame metrics."""
    frame_ids = sorted(set(pred_by_frame) | set(gt_by_frame))
    per_frame = []
    sum_tp = sum_fp = sum_fn = 0
    for fid in frame_ids:
        pred = pred_by_frame.get(fid, np.zeros((0, 2), dtype=int))
        gt = gt_by_frame.get(fid, np.zeros((0, 2), dtype=int))
        conf = (pred_confidence_by_frame.get(fid)
                if pred_confidence_by_frame is not None else None)
        result = match_points(pred, gt, radius=radius, pred_confidence=conf)
        sum_tp += result["tp"]
        sum_fp += result["fp"]
        sum_fn += result["fn"]
        per_frame.append({"frame_id": int(fid),
                          "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
                          "precision": result["precision"],
                          "recall": result["recall"],
                          "f1": result["f1"]})

    precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "global": {"tp": sum_tp, "fp": sum_fp, "fn": sum_fn,
                   "precision": float(precision), "recall": float(recall),
                   "f1": float(f1)},
        "per_frame": per_frame,
    }


def load_xpts_csvs_for_frames(cache_dir: str | Path,
                              frame_ids: Iterable[int],
                              classes: Sequence[str] = ("X",),
                              ) -> dict[int, np.ndarray]:
    """Load {fid}_xpts.csv for each requested frame into a {frame_id: coords} dict."""
    cache_dir = Path(cache_dir)
    out: dict[int, np.ndarray] = {}
    for fid in frame_ids:
        csv_path = cache_dir / f"{fid}_xpts.csv"
        if csv_path.exists():
            out[fid] = load_xpts_csv(csv_path, classes=classes)
        else:
            out[fid] = np.zeros((0, 2), dtype=int)
    return out
