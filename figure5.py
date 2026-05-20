from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# -------------------------
# Labels / markers
# -------------------------
def nice_tuner_name(t: str) -> str:
    t = str(t).lower()
    return {
        "bayesian": "BayesOpt",
        "hyperband": "Hyperband",
        "evol": "Diff. Evolution",
        "random": "RandomSearch",
    }.get(t, t)


def marker_for(t: str) -> str:
    t = str(t).lower()
    return {
        "random": "D",
        "bayesian": "o",
        "hyperband": "s",
        "evol": "^",
    }.get(t, "o")


# -------------------------
# Ellipse shading
# -------------------------
def add_cov_ellipse(ax, x, y, n_std=1.0, facecolor="none", alpha=0.10):
    """
    Add covariance ellipse (~n_std) in (x,y) space.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 6:
        return

    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(vals, 0.0))
    cx, cy = float(np.mean(x)), float(np.mean(y))

    ell = Ellipse((cx, cy), width=width, height=height, angle=theta,
                  facecolor=facecolor, edgecolor="none", alpha=alpha)
    ax.add_patch(ell)


# -------------------------
# Pareto + smoothing
# -------------------------
def pareto_mask_min2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Pareto-efficient mask for minimizing (x,y).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    mask = np.zeros_like(good, dtype=bool)
    if not np.any(good):
        return mask

    idx = np.where(good)[0]
    order = idx[np.argsort(x[idx], kind="mergesort")]  # increasing x
    best_y = np.inf
    for j in order:
        if y[j] < best_y - 1e-12:
            mask[j] = True
            best_y = y[j]
    return mask


def smooth_monotone(x: np.ndarray, y: np.ndarray, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Light smoothing: running median + enforce non-increasing y with increasing x.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) <= 2:
        return x, y

    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    half = w // 2

    y_med = y.copy()
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        y_med[i] = np.median(y[lo:hi])

    y_mono = np.minimum.accumulate(y_med)
    return x, y_mono


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points_csv", type=str, default="figures/Figure5_points.csv")
    ap.add_argument("--outdir", type=str, default="figures")

    ap.add_argument("--trim", type=float, default=0.05,
                    help="Tail trim fraction for thresholds (e.g., 0.05 => keep 5–95%).")
    ap.add_argument("--trim_rmse", action="store_true",
                    help="Also trim RMSE using the same tail fraction.")
    ap.add_argument("--min_minutes", type=float, default=10.0,
                    help="Drop runs with cost < min_minutes (likely failed/empty tuning).")

    # IMPORTANT CHANGE:
    # compute trim thresholds on the FULL dataset (before min_minutes filter),
    # then apply min_minutes + those thresholds.
    ap.add_argument("--trim_reference", choices=["full", "post_min"], default="full",
                    help="Where to compute quantile thresholds: full dataset (default) or after min_minutes filter.")

    ap.add_argument("--h_label", type=str, default="H=1")

    # shading default ON, can disable
    ap.add_argument("--no_shade_groups", action="store_true",
                    help="Disable tuner-group ellipse shading in panel (b).")

    ap.add_argument("--pareto_smooth", action="store_true",
                    help="Light smoothing of Pareto curve.")
    ap.add_argument("--pareto_window", type=int, default=3)

    args = ap.parse_args()

    points_csv = Path(args.points_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df0 = pd.read_csv(points_csv)

    needed = {"tuner", "cost_hours", "rmse"}
    missing = needed - set(df0.columns)
    if missing:
        raise RuntimeError(f"{points_csv} missing columns: {sorted(missing)}")

    df0 = df0.copy()
    df0["tuner"] = df0["tuner"].astype(str).str.lower()
    df0["cost_hours"] = pd.to_numeric(df0["cost_hours"], errors="coerce")
    df0["rmse"] = pd.to_numeric(df0["rmse"], errors="coerce")

    df0 = df0[np.isfinite(df0["cost_hours"]) & np.isfinite(df0["rmse"])].copy()
    if len(df0) < 20:
        raise RuntimeError("Too few valid points in points_csv.")

    q = float(args.trim)

    # Choose where thresholds are computed
    if args.trim_reference == "full":
        ref = df0
    else:
        min_hours = float(args.min_minutes) / 60.0
        ref = df0[df0["cost_hours"] >= min_hours].copy()

    cost_lo = ref["cost_hours"].quantile(q)
    cost_hi = ref["cost_hours"].quantile(1.0 - q)

    if args.trim_rmse:
        rmse_lo = ref["rmse"].quantile(q)
        rmse_hi = ref["rmse"].quantile(1.0 - q)
    else:
        rmse_lo, rmse_hi = -np.inf, np.inf

    # Apply filters: (1) min_minutes, then (2) quantile thresholds from ref
    min_hours = float(args.min_minutes) / 60.0
    df = df0[df0["cost_hours"] >= min_hours].copy()

    m = (
        (df["cost_hours"] >= cost_lo) & (df["cost_hours"] <= cost_hi) &
        (df["rmse"] >= rmse_lo) & (df["rmse"] <= rmse_hi)
    )
    df_t = df[m].copy()

    if len(df_t) < 40:
        raise RuntimeError(f"Too few points after filtering (n={len(df_t)}). "
                           f"Try smaller trim or trim_reference=post_min.")

    # Canonical tuner order
    order = [t for t in ["random", "bayesian", "hyperband", "evol"] if t in df_t["tuner"].unique()]

    # Style
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": True,   # was False
        "axes.spines.right": True, # was False
    })

    fig = plt.figure(figsize=(14.5, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5]) # was 1, 1.35

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # panel labels
    ax1.text(-0.06, 1.05, "a",
            transform=ax1.transAxes,
            fontsize=16, fontweight="bold",
            va="top", ha="left", clip_on=False)

    ax2.text(-0.06, 1.05, "b",
            transform=ax2.transAxes,
            fontsize=16, fontweight="bold",
            va="top", ha="left", clip_on=False)

    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    color_map = {t: cycle[i % len(cycle)] for i, t in enumerate(["random", "bayesian", "hyperband", "evol"])}

    # -------------------------
    # (a) Violin of cost by tuner
    # -------------------------
    data = [df_t.loc[df_t["tuner"] == t, "cost_hours"].values for t in order]
    positions = np.arange(1, len(order) + 1)

    vp = ax1.violinplot(
        data, positions=positions, widths=0.85,
        showmeans=False, showmedians=False, showextrema=False
    )
    for b in vp["bodies"]:
        b.set_facecolor("0.65")
        b.set_edgecolor("0.35")
        b.set_alpha(0.20)
        b.set_linewidth(0.9)

    # symmetric jitter
    rng = np.random.default_rng(42)
    jitter = 0.10 # was 0.11

    for i, t in enumerate(order, start=1):
        vals = df_t.loc[df_t["tuner"] == t, "cost_hours"].values
        if vals.size == 0:
            continue
        xj = i + rng.uniform(-jitter, jitter, size=vals.size)
        ax1.scatter(
            xj, vals, s=18, alpha=0.48,
            marker=marker_for(t),
            color=color_map[t],
            edgecolors="none"
        )
        med = float(np.nanmedian(vals))
        ax1.plot([i - 0.20, i + 0.20], [med, med], lw=2.0, color=color_map[t])

    ax1.set_xticks(positions)
    ax1.set_xticklabels([nice_tuner_name(t) for t in order], rotation=0, ha="center") # was rotation=15, ha="right"
    ax1.set_ylabel("Compute cost (GPU-hours)")
    ax1.set_title("Compute cost by tuning algorithm", loc="center") # was loc="left"
    ax1.grid(True, axis="y", color="0.90", lw=0.9)
    ax1.grid(False, axis="x")

    # lock y-limits to trimmed range (+ small padding) to ensure the “zoom”
    y0, y1 = float(cost_lo), float(cost_hi)
    pad = 0.04 * max(1e-9, (y1 - y0))
    ax1.set_ylim(max(0.0, y0 - pad), y1 + pad)

    # -------------------------
    # (b) Scatter: RMSE (x) vs cost (y) + shaded groups + Pareto
    # -------------------------
    shade = (not args.no_shade_groups)
    if shade:
        for t in order:
            sub = df_t[df_t["tuner"] == t]
            add_cov_ellipse(
                ax2,
                sub["rmse"].values,
                sub["cost_hours"].values,
                n_std=2.0,
                facecolor=color_map[t],
                alpha=0.10,
            )

    for t in order:
        sub = df_t[df_t["tuner"] == t]
        ax2.scatter(
            sub["rmse"].values,
            sub["cost_hours"].values,
            s=40, alpha=0.48,
            marker=marker_for(t),
            color=color_map[t],
            label=nice_tuner_name(t),
        )

    # Pareto front (minimize rmse and cost)
    x = df_t["rmse"].to_numpy()
    y = df_t["cost_hours"].to_numpy()
    pm = pareto_mask_min2(x, y)
    pareto = df_t.loc[pm].sort_values("rmse")

    px = pareto["rmse"].to_numpy()
    py = pareto["cost_hours"].to_numpy()

    if args.pareto_smooth and len(px) >= 4:
        px, py = smooth_monotone(px, py, window=args.pareto_window)

    ax2.plot(px, py, lw=2.2, ls="--", color="C3", label="Pareto front")

    ax2.set_xlabel(r"RMSE (m$^3$/s)")
    ax2.set_ylabel("Compute cost (GPU-hours)")
    ax2.set_title(f"Cost–performance trade-off (lower RMSE is better) | {args.h_label}", loc="center") # was loc="left"
    ax2.grid(True, color="0.90", lw=0.9)
    ax2.legend(frameon=False, loc="upper right")

    # match y-limits to panel (a) for direct comparability
    ax2.set_ylim(ax1.get_ylim())

    out_png = outdir / "Figure5_v4_cost_vs_rmse_trim_shaded.png"
    out_pdf = outdir / "Figure5_v4_cost_vs_rmse_trim_shaded.pdf"
    fig.savefig(out_png, dpi=1200)
    fig.savefig(out_pdf)
    plt.close(fig)

    df_t.to_csv(outdir / "Figure5_v4_points_trimmed.csv", index=False)

    print(f"[INFO] raw points: {len(df0)}")
    print(f"[INFO] kept after <{args.min_minutes:.0f} min + trim({args.trim_reference}, {q:.2f}): {len(df_t)}")
    print(f"[INFO] cost kept in [{cost_lo:.3f}, {cost_hi:.3f}] hours; rmse kept in [{rmse_lo:.3f}, {rmse_hi:.3f}]")
    print(f"[DONE] Saved: {out_png}")
    print(f"[DONE] Saved: {out_pdf}")
    print(f"[DONE] Saved trimmed points: {outdir / 'Figure5_v4_points_trimmed.csv'}")


if __name__ == "__main__":
    main()