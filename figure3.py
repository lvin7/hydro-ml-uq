from __future__ import annotations

import argparse
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from data_utils import data_prep

# Custom layers
from tcn import TCN
from tkan import TKAN


# Color palette
COLORS = {
    "nwp":   "#8B55A0A0",  # purple
    "feat":  "#3C81B39E",  # blue
    "arch":  "#E69D009D",  # orange-yellow
}



# -----------------------------
# Custom loss (safe for load_model even with compile=False)
# -----------------------------
class PinballLoss(tf.keras.losses.Loss):
    def __init__(self, quantile, name="pinball_loss"):
        super().__init__(name=name)
        self.quantile = float(quantile)

    def call(self, y_true, y_pred):
        err = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.quantile * err, (self.quantile - 1.0) * err))


CUSTOM_OBJECTS = {
    "TCN": TCN,
    "TKAN": TKAN,
    "PinballLoss": PinballLoss,
}


# -----------------------------
# Cache helpers (supports both naming schemes)
# -----------------------------
def run_id_from_row(row: pd.Series) -> str:
    # Compatible with earlier v2/v3 naming attempts
    return f"{row['model']}_{row['nwp']}_{row['feat']}_{row['tuner']}_{int(row['replicate'])}"


def pred_path_canonical(pred_cache_root: Path, run_dir: str) -> Path:
    # Preferred canonical layout: pred_cache/<run_dir>/y_pred.npy
    return pred_cache_root / Path(run_dir) / "y_pred.npy"


def pred_path_flat(pred_cache_root: Path, run_id: str) -> Path:
    # Legacy/flat layout: pred_cache/<run_id>.npy
    return pred_cache_root / f"{run_id}.npy"


def load_or_build_pred(
    row: pd.Series,
    pred_cache_root: Path,
    data_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    val_start: str,
    test_start: str,
    batch_size: int = 256,
    overwrite: bool = False,
) -> np.ndarray:
    run_dir = str(row["run_dir"])
    nwp = str(row["nwp"])
    feat = str(row["feat"])
    keras_path = Path(str(row["keras_path"]))
    rid = run_id_from_row(row)

    # ✅ Always ensure test data is present, even if preds are cached
    key = (nwp, feat)
    if key not in data_cache:
        _, _, _, _, X_test, y_test, _ = data_prep(
            nwp=nwp, target="Q", vars=feat, val_start=val_start, test_start=test_start
        )
        data_cache[key] = (np.asarray(X_test, dtype=float), np.asarray(y_test, dtype=float))

    p_can = pred_path_canonical(pred_cache_root, run_dir)
    p_flat = pred_path_flat(pred_cache_root, rid)

    if not overwrite:
        if p_can.exists():
            return np.asarray(np.load(p_can, allow_pickle=False), dtype=float)
        if p_flat.exists():
            return np.asarray(np.load(p_flat, allow_pickle=False), dtype=float)

    X_test, _y_test = data_cache[key]

    if not keras_path.is_absolute():
        keras_path = Path.cwd() / keras_path

    model = tf.keras.models.load_model(keras_path, compile=False, custom_objects=CUSTOM_OBJECTS)
    yp = model.predict(X_test, batch_size=batch_size, verbose=0)
    yp = np.asarray(yp, dtype=float)

    p_can.parent.mkdir(parents=True, exist_ok=True)
    np.save(p_can, yp)
    try:
        np.save(p_flat, yp)
    except Exception:
        pass

    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    del model
    gc.collect()

    return yp


# -----------------------------
# Metrics (per horizon)
# -----------------------------
def _finite_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    m = _finite_mask(y_true, y_pred)
    if not np.any(m):
        return np.nan
    yt = y_true[m]
    yp = y_pred[m]
    denom = np.sum((yt - np.mean(yt)) ** 2)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((yt - yp) ** 2) / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    m = _finite_mask(y_true, y_pred)
    if not np.any(m):
        return np.nan
    e = y_true[m] - y_pred[m]
    return float(np.sqrt(np.mean(e ** 2)))


def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    m = _finite_mask(y_true, y_pred)
    if not np.any(m):
        return np.nan
    yt = y_true[m]
    yp = y_pred[m]

    mu_t = np.mean(yt)
    mu_p = np.mean(yp)
    sd_t = np.std(yt, ddof=0)
    sd_p = np.std(yp, ddof=0)

    if sd_t <= 0 or sd_p <= 0:
        return np.nan

    r = np.corrcoef(yt, yp)[0, 1]
    alpha = sd_p / sd_t
    beta = mu_p / (mu_t + 1e-12)

    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


# -----------------------------
# Fair within-factor matching
# -----------------------------
NWPS = ["ifs", "gfs", "ukmo", "gem"]
FEATS = ["Qp", "Qpt", "Qpts", "Qptsd"]
ARCHS = ["LSTM", "TCN", "TKAN", "Dense"]  # exclude Dense for Figure 3
HORIZONS = [1, 3, 5]
METRICS = ["NSE", "KGE", "RMSE"]


def collect_block_distributions(df_long: pd.DataFrame, factor: str) -> dict[str, list[float]]:
    """
    df_long columns required:
      model, nwp, feat, tuner, replicate, horizon, metric, value
    factor in {"nwp","feat","model"}
    Returns: level -> list of values (balanced by matched sets).
    """
    out: dict[str, list[float]] = {}

    if factor == "nwp":
        levels = NWPS
        group_keys = ["model", "feat", "tuner", "replicate"]  # hold all else fixed
        level_col = "nwp"
    elif factor == "feat":
        levels = FEATS
        group_keys = ["model", "nwp", "tuner", "replicate"]
        level_col = "feat"
    elif factor == "model":
        levels = ARCHS
        group_keys = ["nwp", "feat", "tuner", "replicate"]
        level_col = "model"
    else:
        raise ValueError("factor must be one of: nwp, feat, model")

    for lv in levels:
        out[lv] = []

    # keep only groups that contain *all* levels for this factor
    g = df_long.groupby(group_keys, dropna=False)

    for _, sub in g:
        present = set(sub[level_col].unique().tolist())
        if not all(lv in present for lv in levels):
            continue
        # balanced: add exactly one value per level from this matched set
        for lv in levels:
            v = sub.loc[sub[level_col] == lv, "value"].values
            # should be exactly one, but take the first finite
            v = v[np.isfinite(v)]
            if v.size:
                out[lv].append(float(v[0]))

    return out


# -----------------------------
# Plot helper (single panel with 3 blocks and gaps)
# -----------------------------
def draw_violin_panel(ax, dist_nwp, dist_feat, dist_arch, title_left: str, y_label: str | None,
                      ylims: tuple[float, float], letter: str):
    # x positions with gaps: 1-4, gap, 6-9, gap, 11-13
    pos_nwp = [1, 2, 3, 4]
    pos_feat = [6, 7, 8, 9]
    pos_arch = [11, 12, 13, 14]

    labels = NWPS + FEATS + ARCHS
    positions = pos_nwp + pos_feat + pos_arch
    label_positions = positions

    # style
    violin_face = "0.55"
    violin_edge = "0.35"
    median_color = "0.10"
    
    def _block_from_pos(pos: float) -> str:
        if pos <= 4:
            return "nwp"
        if 6 <= pos <= 9:
            return "feat"
        return "arch"

    def _plot_one(pos, data):
        if data is None or len(data) == 0:
            return
        blk = _block_from_pos(pos)
        parts = ax.violinplot([data], positions=[pos], widths=0.85,
                            showmeans=False, showmedians=False, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(COLORS[blk])
            b.set_edgecolor(COLORS[blk])
            b.set_alpha(0.45)
            b.set_linewidth(0.8)
        med = np.nanmedian(np.asarray(data, dtype=float))
        ax.scatter([pos], [med],
                s=22,
                color=COLORS[blk],
                edgecolor="0.15",
                linewidth=0.6,
                zorder=3)

    # NWP block
    for p, lv in zip(pos_nwp, NWPS):
        _plot_one(p, dist_nwp.get(lv, []))

    # Features block
    for p, lv in zip(pos_feat, FEATS):
        _plot_one(p, dist_feat.get(lv, []))

    # Architecture block
    for p, lv in zip(pos_arch, ARCHS):
        _plot_one(p, dist_arch.get(lv, []))

    # separators between blocks
    ax.axvline(5.0, color="0.7", lw=0.9, ls="--", zorder=0)
    ax.axvline(10.0, color="0.7", lw=0.9, ls="--", zorder=0)

    # group labels (inside, near top)
    y0, y1 = ylims
    y_text = y1 - 0.03 * (y1 - y0)
    ax.text(np.mean(pos_nwp),  y_text, "NWP",          ha="center", va="top", fontsize=11, color="0.25")  # was 9
    ax.text(np.mean(pos_feat), y_text, "Features",     ha="center", va="top", fontsize=11, color="0.25")  # was 9
    ax.text(np.mean(pos_arch), y_text, "Architecture", ha="center", va="top", fontsize=11, color="0.25")  # was 9

    # axes
    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(*ylims)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", color="0.90", lw=0.7)
    ax.grid(False, axis="x")

    # Title anchored left, with panel letter
    ax.set_title(f"{title_left}", loc="center", fontsize=13)  # was loc="left", fontsize=11

    if y_label:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    # make spines clean
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_csv", type=str, default="analysis_out_v4/tables/runs_kept_v4.csv")
    ap.add_argument("--pred_cache", type=str, default="pred_cache_v4")
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument("--val_start", type=str, default="2023-01-01")
    ap.add_argument("--test_start", type=str, default="2023-10-01")
    ap.add_argument("--overwrite_cache", action="store_true")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    runs_csv = Path(args.runs_csv)
    pred_cache = Path(args.pred_cache)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred_cache.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(runs_csv)

    required = {"model", "nwp", "feat", "replicate", "tuner", "run_dir", "keras_path"}
    miss = required - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns in runs CSV: {sorted(miss)}")

    # Keep only requested architectures for Figure 3
    df = df[df["model"].isin(ARCHS)].copy()
    df["replicate"] = df["replicate"].astype(int)
    df["nwp"] = df["nwp"].astype(str)
    df["feat"] = df["feat"].astype(str)
    df["tuner"] = df["tuner"].astype(str)

    if len(df) == 0:
        raise RuntimeError("No runs left after filtering to architectures LSTM/TCN/TKAN.")

    # Data cache per (nwp, feat)
    data_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    # Compute metrics per run for horizons 1,3,5
    rows = []
    print(f"[INFO] Computing per-run metrics for {len(df)} runs (v4-kept, filtered to {ARCHS})...")

    for i, row in df.iterrows():
        yp = load_or_build_pred(
            row=row,
            pred_cache_root=pred_cache,
            data_cache=data_cache,
            val_start=args.val_start,
            test_start=args.test_start,
            batch_size=args.batch_size,
            overwrite=args.overwrite_cache,
        )

        # load matching y_test for this (nwp, feat)
        X_test, y_test = data_cache[(row["nwp"], row["feat"])]

        # sanity
        if yp.shape != y_test.shape:
            raise RuntimeError(f"Prediction shape mismatch for {run_id_from_row(row)}: yp {yp.shape} vs y_test {y_test.shape}")

        for h in HORIZONS:
            yt = y_test[:, h - 1]
            yh = yp[:, h - 1]
            rows.append({
                "model": row["model"],
                "nwp": row["nwp"],
                "feat": row["feat"],
                "tuner": row["tuner"],
                "replicate": int(row["replicate"]),
                "horizon": h,
                "NSE": nse(yt, yh),
                "KGE": kge(yt, yh),
                "RMSE": rmse(yt, yh),
            })

        if (len(rows) // len(HORIZONS)) % 50 == 0:
            done = (len(rows) // len(HORIZONS))
            print(f"[INFO] processed {done}/{len(df)}")

    dfm = pd.DataFrame(rows)

    # Long format for matching logic
    df_long = dfm.melt(
        id_vars=["model", "nwp", "feat", "tuner", "replicate", "horizon"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )

    # Precompute distributions per (metric, horizon) for each factor block
    dist = {}  # dist[(metric,h)] = (dist_nwp, dist_feat, dist_arch)
    for met in METRICS:
        for h in HORIZONS:
            sub = df_long[(df_long["metric"] == met) & (df_long["horizon"] == h)].copy()

            dist_nwp  = collect_block_distributions(sub, factor="nwp")
            dist_feat = collect_block_distributions(sub, factor="feat")
            dist_arch = collect_block_distributions(sub, factor="model")

            dist[(met, h)] = (dist_nwp, dist_feat, dist_arch)

    # Row-wise y-limits
    # NSE fixed; KGE & RMSE based on pooled values across horizons/categories
    def pooled_vals(met: str) -> np.ndarray:
        vals = []
        for h in HORIZONS:
            d_nwp, d_feat, d_arch = dist[(met, h)]
            for dct in [d_nwp, d_feat, d_arch]:
                for lv, arr in dct.items():
                    vals.extend([v for v in arr if np.isfinite(v)])
        return np.asarray(vals, dtype=float)

    ylims = {}

    # NSE
    ylims["NSE"] = (0.0, 1.05)

    # KGE
    kge_vals = pooled_vals("KGE")
    if kge_vals.size:
        lo = np.nanpercentile(kge_vals, 1)
        hi = np.nanpercentile(kge_vals, 99)
        lo = float(min(lo, 0.0))  # allow negative range
        hi = float(max(hi, 1.0))
        pad = 0.05 * (hi - lo + 1e-9)
        ylims["KGE"] = (lo - pad, hi + pad)
    else:
        ylims["KGE"] = (0.0, 1.05)

    # RMSE
    rmse_vals = pooled_vals("RMSE")
    if rmse_vals.size:
        hi = float(np.nanpercentile(rmse_vals, 99.5))
        hi = hi * 1.08
        ylims["RMSE"] = (0.0, hi)
    else:
        ylims["RMSE"] = (0.0, 1.0)

    # Plot layout: 3x3 (rows metrics, cols horizons)
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


    fig, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)

    letters = list("abcdefghi")
    idx = 0

    for r, met in enumerate(["NSE", "KGE", "RMSE"]):
        for c, h in enumerate([1, 3, 5]):
            ax = axes[r, c]
            d_nwp, d_feat, d_arch = dist[(met, h)]

            day_txt = "day" if h == 1 else "days"
            title_left = f"{met} | H={h} {day_txt}"

            ylab = met if c == 0 else None
            draw_violin_panel(
                ax=ax,
                dist_nwp=d_nwp,
                dist_feat=d_feat,
                dist_arch=d_arch,
                title_left=title_left,
                y_label=ylab,
                ylims=ylims[met],
                letter=letters[idx],
            )
            '''
            ax.text(-0.12, 1.05, letters[idx],
                    transform=ax.transAxes,
                    fontsize=12, fontweight="bold",
                    va="top", ha="left",
                    clip_on=False)
            '''
            idx += 1

    out_png = outdir / "Figure3_v4.png"
    out_pdf = outdir / "Figure3_v4.pdf"
    fig.savefig(out_png, dpi=1200)
    fig.savefig(out_pdf)
    plt.close(fig)

    # Optional: write small “publishable” summaries
    # counts per level per block, per metric/horizon
    sum_rows = []
    for met in METRICS:
        for h in HORIZONS:
            d_nwp, d_feat, d_arch = dist[(met, h)]
            for block, dct in [("NWP", d_nwp), ("Features", d_feat), ("Architecture", d_arch)]:
                for lv, arr in dct.items():
                    a = np.asarray(arr, dtype=float)
                    a = a[np.isfinite(a)]
                    if a.size == 0:
                        continue
                    sum_rows.append({
                        "metric": met,
                        "horizon": h,
                        "block": block,
                        "level": lv,
                        "n": int(a.size),
                        "median": float(np.nanmedian(a)),
                        "q25": float(np.nanpercentile(a, 25)),
                        "q75": float(np.nanpercentile(a, 75)),
                    })
    df_sum = pd.DataFrame(sum_rows).sort_values(["metric", "horizon", "block", "level"])
    df_sum.to_csv(outdir / "Figure3_v4_summary.csv", index=False)

    print(f"[DONE] Saved: {out_png}")
    print(f"[DONE] Saved: {out_pdf}")
    print(f"[DONE] Summary table: {outdir / 'Figure3_v4_summary.csv'}")
    print(f"[INFO] Using pred_cache at: {pred_cache}")


if __name__ == "__main__":
    main()