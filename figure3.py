"""
Figure 3 — Factor-wise performance distributions at selected horizons.

Reads:
  - analysis_out_v4/tables/runs_kept_v4.csv  (the v4-kept ensemble)
  - pred_cache_v4/                           (cached predictions written by
                                              full_pipeline_analysis.py)

For each (metric, horizon), computes a matched-set distribution per workflow
factor level (NWP, feature configuration, architecture). Produces a 3 x 3 grid
of violin panels and a summary CSV.

Prerequisite: run full_pipeline_analysis.py first to populate pred_cache_v4/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import data_prep
import analysis_utils as au


# Color palette
COLORS = {
    "nwp":   "#8B55A0A0",  # purple
    "feat":  "#3C81B39E",  # blue
    "arch":  "#E69D009D",  # orange-yellow
}


# Factor level definitions
NWPS = ["ifs", "gfs", "ukmo", "gem"]
FEATS = ["Qp", "Qpt", "Qpts", "Qptsd"]
ARCHS = ["LSTM", "TCN", "TKAN", "Dense"]
HORIZONS = [1, 3, 5]
METRICS = ["NSE", "KGE", "RMSE"]


# -----------------------------
# Fair within-factor matching
# -----------------------------
def collect_block_distributions(df_long: pd.DataFrame, factor: str) -> dict[str, list[float]]:
    """
    Build a balanced distribution per factor level.

    df_long must contain columns:
        model, nwp, feat, tuner, replicate, horizon, metric, value

    For factor in {"nwp", "feat", "model"}: keep only groups (defined by all
    other factors fixed) that contain every level of the chosen factor, then
    take one value per level from each matched set. This gives a balanced
    comparison across levels of the chosen factor.
    """
    out: dict[str, list[float]] = {}

    if factor == "nwp":
        levels = NWPS
        group_keys = ["model", "feat", "tuner", "replicate"]
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

    g = df_long.groupby(group_keys, dropna=False)

    for _, sub in g:
        present = set(sub[level_col].unique().tolist())
        if not all(lv in present for lv in levels):
            continue
        for lv in levels:
            v = sub.loc[sub[level_col] == lv, "value"].values
            v = v[np.isfinite(v)]
            if v.size:
                out[lv].append(float(v[0]))

    return out


# -----------------------------
# Plot helper (single panel with 3 blocks and gaps)
# -----------------------------
def draw_violin_panel(ax, dist_nwp, dist_feat, dist_arch, title: str,
                      y_label: str | None, ylims: tuple[float, float]):
    # x positions with gaps: 1-4, gap, 6-9, gap, 11-14
    pos_nwp = [1, 2, 3, 4]
    pos_feat = [6, 7, 8, 9]
    pos_arch = [11, 12, 13, 14]

    labels = NWPS + FEATS + ARCHS
    positions = pos_nwp + pos_feat + pos_arch

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

    for p, lv in zip(pos_nwp, NWPS):
        _plot_one(p, dist_nwp.get(lv, []))
    for p, lv in zip(pos_feat, FEATS):
        _plot_one(p, dist_feat.get(lv, []))
    for p, lv in zip(pos_arch, ARCHS):
        _plot_one(p, dist_arch.get(lv, []))

    # block separators
    ax.axvline(5.0, color="0.7", lw=0.9, ls="--", zorder=0)
    ax.axvline(10.0, color="0.7", lw=0.9, ls="--", zorder=0)

    # group labels (inside, near top)
    y0, y1 = ylims
    y_text = y1 - 0.03 * (y1 - y0)
    ax.text(np.mean(pos_nwp),  y_text, "NWP",          ha="center", va="top", fontsize=11, color="0.25")
    ax.text(np.mean(pos_feat), y_text, "Features",     ha="center", va="top", fontsize=11, color="0.25")
    ax.text(np.mean(pos_arch), y_text, "Architecture", ha="center", va="top", fontsize=11, color="0.25")

    # axes
    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(*ylims)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", color="0.90", lw=0.7)
    ax.grid(False, axis="x")

    ax.set_title(title, loc="center", fontsize=13)
    ax.set_ylabel(y_label if y_label else "")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_csv", type=str, default="analysis_out_v4/tables/runs_kept_v4.csv")
    ap.add_argument("--pred_cache", type=str, default="pred_cache_v4",
                    help="Directory with cached predictions (populated by full_pipeline_analysis.py).")
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument("--val_start", type=str, default="2023-01-01")
    ap.add_argument("--test_start", type=str, default="2023-10-01")
    args = ap.parse_args()

    runs_csv = Path(args.runs_csv)
    pred_cache = Path(args.pred_cache)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(runs_csv)

    required = {"model", "nwp", "feat", "replicate", "tuner", "run_id"}
    miss = required - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns in runs CSV: {sorted(miss)}")

    # Keep only the architectures used in Figure 3
    df = df[df["model"].isin(ARCHS)].copy()
    df["replicate"] = df["replicate"].astype(int)
    df["nwp"] = df["nwp"].astype(str)
    df["feat"] = df["feat"].astype(str)
    df["tuner"] = df["tuner"].astype(str)

    if len(df) == 0:
        raise RuntimeError(f"No runs left after filtering to architectures {ARCHS}.")

    # y_test cache per (nwp, feat) — at most 4 x 4 = 16 unique combinations
    y_test_cache: dict[tuple[str, str], np.ndarray] = {}

    rows = []
    n_missing_pred = 0
    n_shape_fail = 0

    print(f"[INFO] Reading cached predictions from {pred_cache} for {len(df)} runs "
          f"(v4-kept, filtered to {ARCHS})...")

    for _, row in df.iterrows():
        # Load y_test once per (nwp, feat)
        key = (str(row["nwp"]), str(row["feat"]))
        if key not in y_test_cache:
            _, _, _, _, _, y_test, _ = data_prep(
                nwp=key[0], target="Q", vars=key[1],
                val_start=args.val_start, test_start=args.test_start,
            )
            y_test_cache[key] = np.asarray(y_test, dtype=float)

        y_test = y_test_cache[key]

        # Load cached prediction
        yp = au.load_prediction(str(pred_cache), str(row["run_id"]))
        if yp is None:
            n_missing_pred += 1
            continue
        yp = np.asarray(yp, dtype=float)

        if yp.shape != y_test.shape:
            n_shape_fail += 1
            continue

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
                "NSE": au.nse(yt, yh),
                "KGE": au.kge(yt, yh),
                "RMSE": au.rmse(yt, yh),
            })

    if n_missing_pred:
        print(f"[WARN] missing cached prediction for {n_missing_pred} runs "
              f"(run full_pipeline_analysis.py to populate {pred_cache}).")
    if n_shape_fail:
        print(f"[WARN] prediction-shape mismatch for {n_shape_fail} runs.")

    dfm = pd.DataFrame(rows)
    if len(dfm) == 0:
        raise RuntimeError(
            f"No usable predictions found in {pred_cache}. "
            f"Run full_pipeline_analysis.py first."
        )

    # Long format for matching logic
    df_long = dfm.melt(
        id_vars=["model", "nwp", "feat", "tuner", "replicate", "horizon"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )

    # Precompute matched distributions per (metric, horizon) for each factor block
    dist = {}  # dist[(metric, h)] = (dist_nwp, dist_feat, dist_arch)
    for met in METRICS:
        for h in HORIZONS:
            sub = df_long[(df_long["metric"] == met) & (df_long["horizon"] == h)].copy()
            dist[(met, h)] = (
                collect_block_distributions(sub, factor="nwp"),
                collect_block_distributions(sub, factor="feat"),
                collect_block_distributions(sub, factor="model"),
            )

    # Row-wise y-limits: NSE fixed; KGE & RMSE based on pooled values
    def pooled_vals(met: str) -> np.ndarray:
        vals = []
        for h in HORIZONS:
            d_nwp, d_feat, d_arch = dist[(met, h)]
            for dct in [d_nwp, d_feat, d_arch]:
                for _, arr in dct.items():
                    vals.extend([v for v in arr if np.isfinite(v)])
        return np.asarray(vals, dtype=float)

    ylims = {"NSE": (0.0, 1.05)}

    kge_vals = pooled_vals("KGE")
    if kge_vals.size:
        lo = float(min(np.nanpercentile(kge_vals, 1), 0.0))
        hi = float(max(np.nanpercentile(kge_vals, 99), 1.0))
        pad = 0.05 * (hi - lo + 1e-9)
        ylims["KGE"] = (lo - pad, hi + pad)
    else:
        ylims["KGE"] = (0.0, 1.05)

    rmse_vals = pooled_vals("RMSE")
    if rmse_vals.size:
        ylims["RMSE"] = (0.0, float(np.nanpercentile(rmse_vals, 99.5)) * 1.08)
    else:
        ylims["RMSE"] = (0.0, 1.0)

    # Plot layout: 3 (metrics) x 3 (horizons)
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

    for r, met in enumerate(["NSE", "KGE", "RMSE"]):
        for c, h in enumerate([1, 3, 5]):
            ax = axes[r, c]
            d_nwp, d_feat, d_arch = dist[(met, h)]

            day_txt = "day" if h == 1 else "days"
            title = f"{met} | H={h} {day_txt}"
            ylab = met if c == 0 else None

            draw_violin_panel(
                ax=ax,
                dist_nwp=d_nwp,
                dist_feat=d_feat,
                dist_arch=d_arch,
                title=title,
                y_label=ylab,
                ylims=ylims[met],
            )

    out_png = outdir / "Figure3_v4.png"
    out_pdf = outdir / "Figure3_v4.pdf"
    fig.savefig(out_png, dpi=1200)
    fig.savefig(out_pdf)
    plt.close(fig)

    # Summary table: counts and IQR per level per block, per metric/horizon
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
    print(f"[INFO] Used pred_cache at: {pred_cache}")


if __name__ == "__main__":
    main()
