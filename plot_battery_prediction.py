"""
Battery SOH plots per battery (single-image per chart, organized into subfolders)

Changes per user requirements:
- One battery per figure (Pred vs True curve)
- Residual analysis: keep only histogram and residual-vs-true scatter
- Remove zoomed windows, Q-Q plot, residual-over-time
- Save to subfolders: plots/curves/ and plots/residuals/
- Segmentation source aligned with main_v5_optimized.py + dataloader: use logging.txt -> data path list (TEST set)
  and re-read those CSVs to compute each battery length (after the same 3-sigma cleaning), then split pred/true accordingly
- If logging-based split fails, fall back to a simple single-segment split (entire sequence as one battery)

Usage:
  python yangzhi/PINN4SOH/plot_battery_prediction.py
  Edit RESULT_FOLDER below or pass via environment var RESULT_FOLDER.
"""

import os
import ast
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================= User config =============================
# Default to a sample run dir; you can edit this to any run_x folder
RESULT_FOLDER = os.environ.get(
    "RESULT_FOLDER",
    "/home/chihao/yangzhi/PINN4SOH/v5results/optimized/no_物理约束/run_2"
)

# Appearance (journal-like)
FIG_DPI = 300
LINE_WIDTH = 2.0
FONT_SIZE = 11
TITLE_SIZE = 13
LABEL_SIZE = 12
LEGEND_SIZE = 10
PRED_COLOR = "#E74C3C"
TRUE_COLOR = "#3498DB"
ERR_COLOR = "#2ECC71"

# =====================================================================

@dataclass
class Segment:
    name: str
    start: int
    end: int   # exclusive


# -------------------- helpers for reading series --------------------

def load_series(folder: str) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.load(os.path.join(folder, "pred_label.npy"), allow_pickle=True).flatten()
    true = np.load(os.path.join(folder, "true_label.npy"), allow_pickle=True).flatten()
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs true {true.shape}")
    return pred, true


# -------------------- find project root (to resolve relative data paths) --------------------

def find_project_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, 'data')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # fallback: one level up
    return os.path.abspath(os.path.join(start_dir, '..'))


# -------------------- parse logging for data paths --------------------

def _extract_lists_from_logging(log_text: str) -> List[List[str]]:
    """Return all path lists that appear immediately after lines containing 'data path:'"""
    lines = log_text.splitlines()
    lists: List[List[str]] = []
    for i, line in enumerate(lines):
        if 'data path:' in line:
            if i + 1 < len(lines):
                raw = lines[i + 1].strip()
                # try Python literal_eval first
                parsed = None
                try:
                    parsed = ast.literal_eval(raw)
                except Exception:
                    try:
                        # try JSON-ish: single quotes -> double quotes
                        parsed = json.loads(raw.replace("'", '"'))
                    except Exception:
                        parsed = None
                if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed):
                    lists.append(parsed)
    return lists


def _battery_id_from_path(p: str) -> Optional[int]:
    """Extract trailing integer id from filename like '...-12.csv'"""
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    # find the part after last '-'
    if '-' in stem:
        tail = stem.split('-')[-1]
    else:
        tail = stem
    try:
        return int(''.join(ch for ch in tail if ch.isdigit()))
    except Exception:
        return None


def choose_test_paths(path_lists: List[List[str]]) -> Optional[List[str]]:
    """From multiple 'data path' lists, choose the one that looks like TEST (ids % 5 == 0)."""
    if not path_lists:
        return None
    best = None
    best_score = -1
    for lst in path_lists:
        ids = [i for i in (_battery_id_from_path(p) for p in lst) if i is not None]
        if not ids:
            score = 0
        else:
            # score: proportion divisible by 5
            score = sum(1 for i in ids if i % 5 == 0) / float(len(ids))
        if score > best_score:
            best = lst
            best_score = score
    return best or path_lists[-1]


def read_logging_test_paths(result_folder: str) -> Optional[List[str]]:
    # prefer 'logging.txt'
    cand = os.path.join(result_folder, 'logging.txt')
    log_path = None
    if os.path.exists(cand):
        log_path = cand
    else:
        # find any .txt
        for fn in os.listdir(result_folder):
            if fn.lower().endswith('.txt'):
                log_path = os.path.join(result_folder, fn)
                break
    if log_path is None:
        return None

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception:
        return None

    lists = _extract_lists_from_logging(text)
    return choose_test_paths(lists)


def reconstruct_mit_test_paths(project_root: str) -> Optional[List[str]]:
    """
    Rebuild the MIT test set file list exactly as in main_v5_optimized.py:
      - root = 'data/MIT data'
      - batches = ['2017-05-12', '2017-06-30', '2018-04-12']
      - test_list: files whose trailing ID % 5 == 0
    Returns absolute paths.
    """
    mit_root = os.path.join(project_root, 'data', 'MIT data')
    if not os.path.isdir(mit_root):
        return None
    batches = ['2017-05-12', '2017-06-30', '2018-04-12']
    test_list: List[str] = []
    for batch in batches:
        batch_root = os.path.join(mit_root, batch)
        if not os.path.isdir(batch_root):
            continue
        try:
            files = os.listdir(batch_root)
        except Exception:
            continue
        for f in files:
            battery_id = _battery_id_from_path(f)
            if battery_id is not None and battery_id % 5 == 0:
                test_list.append(os.path.join(batch_root, f))
    return test_list if test_list else None


# -------------------- re-implement 3-sigma cleaning to match dataloader --------------------

def _three_sigma_indices(ser: pd.Series) -> np.ndarray:
    mu = ser.mean()
    sigma = ser.std()
    if pd.isna(mu) or pd.isna(sigma):
        return np.array([], dtype=int)
    lower = mu - 3 * sigma
    upper = mu + 3 * sigma
    idx = np.where((ser < lower) | (ser > upper))[0]
    return idx


def _delete_3sigma(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0).reset_index(drop=True)
    out_idx = []
    for col in df.columns:
        out = _three_sigma_indices(df[col])
        if out.size:
            out_idx.extend(out.tolist())
    if out_idx:
        df = df.drop(sorted(set(out_idx)), axis=0).reset_index(drop=True)
    return df


def battery_lengths_after_clean(paths: List[str], project_root: str) -> List[int]:
    lengths: List[int] = []
    for p in paths:
        ap = p if os.path.isabs(p) else os.path.join(project_root, p)
        if not os.path.exists(ap):
            # try join with data/ in case paths in log are relative to project root's parent
            alt = os.path.join(project_root, p.lstrip('./'))
            ap = alt if os.path.exists(alt) else p
        try:
            df = pd.read_csv(ap)
        except Exception:
            # if we cannot read, assume zero to avoid crash
            lengths.append(0)
            continue
        # insert cycle index at position df.shape[1]-1 (as in dataloader)
        df.insert(df.shape[1]-1, 'cycle index', np.arange(df.shape[0]))
        df = _delete_3sigma(df)
        n = max(0, df.shape[0] - 1)  # y1 length per battery
        lengths.append(n)
    return lengths


# -------------------- plotting --------------------

def set_style():
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.linewidth": 1.5,
        "grid.alpha": 0.3,
    })


def ensure_dirs(root: str) -> Dict[str, str]:
    plots_root = os.path.join(root, "plots")
    d_curves = os.path.join(plots_root, "curves")
    d_resid = os.path.join(plots_root, "residuals")
    os.makedirs(d_curves, exist_ok=True)
    os.makedirs(d_resid, exist_ok=True)
    return {"curves": d_curves, "residuals": d_resid}


def fig_curve(bname: str, x: np.ndarray, true: np.ndarray, pred: np.ndarray, out_file: str):
    set_style()
    # metrics
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIG_DPI)
    ax.plot(x, true, color=TRUE_COLOR, linewidth=LINE_WIDTH, label="True SOH")
    ax.plot(x, pred, color=PRED_COLOR, linewidth=max(1.0, LINE_WIDTH-0.3), alpha=0.9, label="Predicted SOH")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("SOH")
    ax.set_title(f"{bname} — Predicted vs True")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", framealpha=0.95)

    txt = f"MAE={mae:.4f}\nRMSE={rmse:.4f}\nR²={r2:.4f}"
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, ha="left", va="bottom",
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8, ec="black"))

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def fig_residual_hist(bname: str, residuals: np.ndarray, out_file: str):
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=FIG_DPI)
    ax.hist(residuals, bins=50, color=ERR_COLOR, edgecolor="black", linewidth=1.0)
    ax.axvline(np.mean(residuals), color="red", linestyle="--", linewidth=1.5, label=f"Mean={np.mean(residuals):.4f}")
    ax.axvline(0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Residual (Pred - True)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{bname} — Residual Distribution")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def fig_residual_scatter(bname: str, true: np.ndarray, residuals: np.ndarray, out_file: str):
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=FIG_DPI)
    ax.scatter(true, residuals, s=12, alpha=0.6, color=ERR_COLOR, edgecolors="black", linewidth=0.3)
    ax.axhline(0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xlabel("True SOH")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title(f"{bname} — Residual vs True")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def fig_residuals_summary(
    all_true: np.ndarray,
    all_pred: np.ndarray,
    per_battery_residuals: List[np.ndarray],
    per_battery_names: List[str],
    out_file: str,
    max_scatter_points: int = 8000,
):
    """
    Create a single multi-panel figure aggregating residual information across all batteries.
    Panels:
      (1) Residual histogram (all batteries)
      (2) CDF of absolute residuals
      (3) Residual vs True (scatter, optionally subsampled)
      (4) Boxplot of residuals per battery
    """
    set_style()

    residuals_all = (all_pred - all_true).astype(float)
    abs_res = np.abs(residuals_all)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=FIG_DPI)

    # Panel 1: Histogram of residuals
    ax = axes[0, 0]
    ax.hist(residuals_all, bins=60, color=ERR_COLOR, edgecolor="black", linewidth=0.7)
    ax.axvline(residuals_all.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={residuals_all.mean():.4f}")
    ax.axvline(0, color="black", linestyle=":", linewidth=1.2)
    ax.set_title("Residuals (All Batteries)")
    ax.set_xlabel("Residual (Pred - True)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="best")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Panel 2: CDF of |residual|
    ax = axes[0, 1]
    x_sorted = np.sort(abs_res)
    y_cdf = np.arange(1, x_sorted.size + 1) / x_sorted.size
    ax.plot(x_sorted, y_cdf, color=PRED_COLOR, linewidth=1.8)
    # mark percentiles
    for q, c in [(0.5, "#666666"), (0.9, "#AA5500"), (0.95, "#880000")]:
        v = np.quantile(abs_res, q)
        ax.axvline(v, color=c, linestyle="--", linewidth=1.2)
        ax.text(v, 0.02 + 0.06 * (q >= 0.9), f"{int(q*100)}%={v:.4f}", rotation=90, va="bottom", ha="right", color=c)
    ax.set_title("Empirical CDF of |Residual|")
    ax.set_xlabel("|Residual|")
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Panel 3: Residual vs True (scatter, subsampled)
    ax = axes[1, 0]
    n = all_true.size
    if n > max_scatter_points:
        idx = np.random.RandomState(42).choice(n, size=max_scatter_points, replace=False)
        t_vis = all_true[idx]
        r_vis = residuals_all[idx]
    else:
        t_vis = all_true
        r_vis = residuals_all
    ax.scatter(t_vis, r_vis, s=8, alpha=0.35, color=ERR_COLOR, edgecolors="none")
    ax.axhline(0, color="black", linestyle=":", linewidth=1.2)
    ax.set_title("Residual vs True (All Batteries)")
    ax.set_xlabel("True SOH")
    ax.set_ylabel("Residual (Pred - True)")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Panel 4: Boxplot by battery
    ax = axes[1, 1]
    # To avoid overcrowding, truncate very long names
    names = [n if len(n) <= 14 else (n[:11] + "...") for n in per_battery_names]
    data = [r for r in per_battery_residuals if r.size > 0]
    shown_names = [names[i] for i, r in enumerate(per_battery_residuals) if r.size > 0]
    bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('#DDEEE8')
        patch.set_edgecolor('#555555')
    ax.axhline(0, color="black", linestyle=":", linewidth=1.0)
    ax.set_title("Residuals by Battery (Boxplot)")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_xticks(np.arange(1, len(shown_names) + 1))
    ax.set_xticklabels(shown_names, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


# -------------------- main --------------------

def main(result_folder: str = RESULT_FOLDER):
    result_folder = os.path.abspath(result_folder)
    pred, true = load_series(result_folder)

    # Prefer canonical segmentation from logging + dataset
    test_paths = read_logging_test_paths(result_folder)
    segments: List[Segment] = []

    # Fallback 1: reconstruct from MIT data if logging missing
    if not test_paths:
        project_root = find_project_root(result_folder)
        test_paths = reconstruct_mit_test_paths(project_root)
        if test_paths:
            print(f"Segmentation source: MIT data rules (%5==0) | {len(test_paths)} files")

    if test_paths:
        if 'project_root' not in locals():
            project_root = find_project_root(result_folder)
        lens = battery_lengths_after_clean(test_paths, project_root)
        # Build segments, skipping zero-length files
        cur = 0
        for p, L in zip(test_paths, lens):
            name = os.path.splitext(os.path.basename(p))[0]
            if L <= 0:
                continue
            start, end = cur, cur + L
            segments.append(Segment(name=name, start=start, end=end))
            cur = end
        # Safety: ensure last end doesn't exceed series length
        if segments and segments[-1].end > len(true):
            overflow = segments[-1].end - len(true)
            segments[-1].end -= overflow
        # If sum lengths < len(true) due to data drift, append tail as Unknown
        total_len = sum(seg.end - seg.start for seg in segments)
        if total_len < len(true):
            print(f"Warning: concatenated test lengths ({total_len}) < array length ({len(true)}). Appending tail segment.")
            segments.append(Segment(name="Tail", start=total_len, end=len(true)))
    else:
        # Fallback: single segment
        print("Segmentation source: NONE (logging + MIT rules failed). Using single segment." )
        segments = [Segment(name="Battery_1", start=0, end=len(true))]

    out_dirs = ensure_dirs(result_folder)

    per_battery_residuals: List[np.ndarray] = []
    per_battery_names: List[str] = []

    for idx, seg in enumerate(segments, 1):
        s, e = seg.start, seg.end
        if e <= s:
            continue
        x = np.arange(e - s)
        p = pred[s:e]
        t = true[s:e]
        r = p - t

        # collect for summary
        per_battery_residuals.append(r)
        per_battery_names.append(seg.name)

        curve_path = os.path.join(out_dirs["curves"], f"{idx:02d}_{seg.name}.png")
        hist_path = os.path.join(out_dirs["residuals"], f"{idx:02d}_{seg.name}_residual_hist.png")
        sc_path = os.path.join(out_dirs["residuals"], f"{idx:02d}_{seg.name}_residual_scatter.png")

        fig_curve(seg.name, x, t, p, curve_path)
        fig_residual_hist(seg.name, r, hist_path)
        fig_residual_scatter(seg.name, t, r, sc_path)

    # Global residuals summary (all batteries in one figure)
    summary_path = os.path.join(out_dirs["residuals"], "00_residuals_summary.png")
    fig_residuals_summary(true, pred, per_battery_residuals, per_battery_names, summary_path)

    print(f"Done. Generated {len(segments)} batteries under {os.path.join(result_folder, 'plots')}\n"
          f"Curves -> {out_dirs['curves']}\nResiduals -> {out_dirs['residuals']}")


if __name__ == "__main__":
    main()
