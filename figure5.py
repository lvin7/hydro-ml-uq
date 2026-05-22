"""
Figure 5 — Cost vs. RMSE Pareto trade-off across HPO algorithms.

Two stages, controlled by a cache:

  1. Compute stage: for each retained pipeline, look up the cached test-set
     prediction (written by full_pipeline_analysis.py), compute RMSE at the
     chosen horizon, and read the tuning cost from tuning_summary.json. Save
     the per-pipeline (cost, RMSE) table to figures/Figure5_points.csv.

  2. Plot stage: read Figure5_points.csv, apply quantile trimming, draw the
     compute-cost violin (panel a) and the cost-vs-RMSE scatter with Pareto
     front (panel b), and save Figure5_v4_cost_vs_rmse_trim_shaded.{png,pdf}.

The compute stage is skipped automatically if Figure5_points.csv already
exists. Pass --recompute to force re-computation.

Prerequisite for the compute stage: run full_pipeline_analysis.py first to
populate pred_cache_v4/ with the test-set predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from data_utils import data_prep
import analysis_utils as au


# =========================================================
# Labels / markers
# =========================================================
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


# =========================================================
# Helpers (compute stage)
# =========================================================
def _resolve_path(p: str | Path) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else (Path.cwd() / p)


def _safe_read_json(p: Path) -> dict:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def rmse_at_horizon(y_test: np.ndarray, y_pred: np.ndarray, horizon: int = 1, agg: str = "single") -> float:
    """RMSE at a specific horizon (1-indexed), or mean RMSE across horizons."""
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    H = y_test.shape[1]

    if agg == "mean":
        return float(np.nanmean([au.rmse(y_test[:, h], y_pred[:, h]) for h in range(H)]))

    h = int(horizon)
    if h < 1 or h > H:
        raise ValueError(f"horizon must be in [1, {H}]")
    return au.rmse(y_test[:, h - 1], y_pred[:, h - 1])


def load_cost_hours(row: pd.Series) -> float | None:
    """Read total tuning time (in hours) from tuning_summary.json."""
    if "tuning_summary_path" in row.index and pd.notna(row["tuning_summary_path"]):
        p = _resolve_path(row["tuning_summary_path"])
    else:
        p = _resolve_path(Path(str(row["run_dir"])) / "tuning_summary.json")

    if not p.exists():
        return None

    d = _safe_read_json(p)
    if "total_tuning_time(min)" in d:
        try:
            return float(d["total_tuning_time(min)"]) / 60.0
        except Exception:
            return None

    for k in ["total_tuning_time_min", "total_tuning_minutes", "total_time_min"]:
        if k in d:
            try:
                return float(d[k]) / 60.0
            except Exception:
                return None

    return None


# =========================================================
# Compute stage — build Figure5_points.csv from cached predictions
# =========================================================
def compute_points(args) -> pd.DataFrame:
    runs_csv = Path(args.runs_csv)
    pred_cache = Path(args.pred_cache)

    df = pd.read_csv(runs_csv)

    required = {"model", "nwp", "feat", "replicate", "tuner", "run_dir", "run_id"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"runs_csv missing columns: {sorted(missing)}")

    df["tuner"] = df["tuner"].astype(str)
    df = df[df["tuner"].isin(args.tuners)].copy()
    if len(df) == 0:
        raise RuntimeError("No runs left after filtering by --tuners.")

    if args.max_models and args.max_models > 0:
        df = df.head(int(args.max_models)).copy()

    # Cache y_test per (nwp, feat) — at most 4 x 4 = 16 unique combinations.
    y_test_cache: dict[tuple[str, str], np.ndarray] = {}

    rows = []
    n_missing_cost = 0
    n_missing_pred = 0
    n_shape_fail = 0

    print(f"[INFO] Reading cached predictions from {pred_cache} for {len(df)} runs...")

    for _, row in df.iterrows():
        cost_h = load_cost_hours(row)
        if cost_h is None or not np.isfinite(cost_h):
            n_missing_cost += 1
            continue

        # Load y_test once per (nwp, feat)
        key = (str(row["nwp"]), str(row["feat"]))
        if key not in y_test_cache:
            _, _, _, _, _, y_test, _ = data_prep(
                nwp=key[0],
                target="Q",
                vars=key[1],
                val_start=args.val_start,
                test_start=args.test_start,
            )
            y_test_cache[key] = np.asarray(y_test, dtype=float)

        y_test = y_test_cache[key]

        # Load cached prediction
        y_pred = au.load_prediction(str(pred_cache), str(row["run_id"]))
        if y_pred is None:
            n_missing_pred += 1
            continue

        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.shape != y_test.shape:
            n_shape_fail += 1
            continue

        r = rmse_at_horizon(y_test, y_pred, horizon=args.horizon, agg=args.rmse_agg)
        rows.append({
            "tuner": str(row["tuner"]),
            "tuner_name": nice_tuner_name(row["tuner"]),
            "cost_hours": float(cost_h),
            "rmse": float(r),
        })

    dff = pd.DataFrame(rows)

    print(f"[INFO] usable points: {len(dff)}")
    if n_missing_cost:
        print(f"[WARN] missing tuning cost for {n_missing_cost} runs.")
    if n_missing_pred:
        print(f"[WARN] missing cached prediction for {n_missing_pred} runs "
              f"(run full_pipeline_analysis.py to populate {pred_cache}).")
    if n_shape_fail:
        print(f"[WARN] prediction-shape mismatch for {n_shape_fail} runs.")

    if len(dff) < 10:
        raise RuntimeError(
            f"Too few usable points to build Figure 5 (got {len(dff)}). "
            f"Ensure {pred_cache} is populated by running full_pipeline_analysis.py first."
        )

    return dff


# =========================================================
# Plot stage — covariance ellipse, Pareto, smoothing
# =========================================================
def add_cov_ellipse(ax, x, y, n_std=1.0, facecolor="none", alpha=0.10):
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


def pareto_mask_min2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    mask = np.zeros_like(good, dtype=bool)
    if not np.any(good):
        return mask

    idx = np.where(good)[0]
    order = idx[np.argsort(x[idx], kind="mergesort")]
    best_y = np.inf
    for j in order:
        if y[j] < best_y - 1e-12:
            mask[j] = True
            best_y = y[j]
    return mask


def smooth_monotone(x: np.ndarray, y: np.ndarray, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
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


def plot_pareto(df0: pd.DataFrame, args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df0 = df0.copy()
    df0["tuner"] = df0["tuner"].astype(str).str.lower()
    df0["cost_hours"] = pd.to_numeric(df0["cost_hours"], errors="coerce")
    df0["rmse"] = pd.to_numeric(df0["rmse"], errors="coerce")
    df0 = df0[np.isfinite(df0["cost_hours"]) & np.isfinite(df0["rmse"])].copy()

    if len(df0) < 20:
        raise RuntimeError("Too few valid points to plot.")

    q = float(args.trim)

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

    min_hours = float(args.min_minutes) / 60.0
    df = df0[df0["cost_hours"] >= min_hours].copy()

    m = (
        (df["cost_hours"] >= cost_lo) & (df["cost_hours"] <= cost_hi) &
        (df["rmse"] >= rmse_lo) & (df["rmse"] <= rmse_hi)
    )
    df_t = df[m].copy()

    if len(df_t) < 40:
        raise RuntimeError(
            f"Too few points after filtering (n={len(df_t)}). "
            f"Try a smaller --trim or --trim_reference post_min."
        )

    order = [t for t in ["random", "bayesian", "hyperband", "evol"] if t in df_t["tuner"].unique()]

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })

    fig = plt.figure(figsize=(14.5, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

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

    rng = np.random.default_rng(42)
    jitter = 0.10

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
    ax1.set_xticklabels([nice_tuner_name(t) for t in order], rotation=0, ha="center")
    ax1.set_ylabel("Compute cost (GPU-hours)")
    ax1.set_title("Compute cost by tuning algorithm", loc="center")
    ax1.grid(True, axis="y", color="0.90", lw=0.9)
    ax1.grid(False, axis="x")

    y0, y1 = float(cost_lo), float(cost_hi)
    pad = 0.04 * max(1e-9, (y1 - y0))
    ax1.set_ylim(max(0.0, y0 - pad), y1 + pad)

    # -------------------------
    # (b) Scatter: RMSE vs cost + Pareto
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
    ax2.set_title(f"Cost–performance trade-off (lower RMSE is better) | {args.h_label}", loc="center")
    ax2.grid(True, color="0.90", lw=0.9)
    ax2.legend(frameon=False, loc="upper right")
    ax2.set_ylim(ax1.get_ylim())

    out_png = outdir / "Figure5_v4_cost_vs_rmse_trim_shaded.png"
    out_pdf = outdir / "Figure5_v4_cost_vs_rmse_trim_shaded.pdf"
    fig.savefig(out_png, dpi=1200)
    fig.savefig(out_pdf)
    plt.close(fig)

    df_t.to_csv(outdir / "Figure5_v4_points_trimmed.csv", index=False)

    print(f"[INFO] raw points: {len(df0)}")
    print(f"[INFO] kept after <{args.min_minutes:.0f} min + trim({args.trim_reference}, {q:.2f}): {len(df_t)}")
    print(f"[INFO] cost kept in [{cost_lo:.3f}, {cost_hi:.3f}] hours; "
          f"rmse kept in [{rmse_lo:.3f}, {rmse_hi:.3f}]")
    print(f"[DONE] Saved: {out_png}")
    print(f"[DONE] Saved: {out_pdf}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    # Shared
    ap.add_argument("--outdir", type=str, default="figures",
                    help="Directory for outputs (CSV and figure files).")
    ap.add_argument("--points_csv", type=str, default="figures/Figure5_points.csv",
                    help="Per-pipeline (cost, RMSE) cache. Computed automatically if missing.")
    ap.add_argument("--recompute", action="store_true",
                    help="Force re-computation of points_csv even if it exists.")

    # Compute stage
    ap.add_argument("--runs_csv", type=str, default="analysis_out_v4/tables/runs_kept_v4.csv")
    ap.add_argument("--pred_cache", type=str, default="pred_cache_v4",
                    help="Directory with cached predictions (populated by full_pipeline_analysis.py).")
    ap.add_argument("--val_start", type=str, default="2023-01-01")
    ap.add_argument("--test_start", type=str, default="2023-10-01")
    ap.add_argument("--tuners", nargs="+", default=["random", "bayesian", "hyperband", "evol"])
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--rmse_agg", choices=["single", "mean"], default="single")
    ap.add_argument("--max_models", type=int, default=0,
                    help="Limit number of models processed (0 = all). Useful for quick tests.")

    # Plot stage
    ap.add_argument("--trim", type=float, default=0.05,
                    help="Tail trim fraction for cost thresholds (e.g., 0.05 keeps 5–95%).")
    ap.add_argument("--trim_rmse", action="store_true",
                    help="Also trim RMSE using the same tail fraction.")
    ap.add_argument("--min_minutes", type=float, default=10.0,
                    help="Drop runs with cost < min_minutes (likely failed/empty tuning).")
    ap.add_argument("--trim_reference", choices=["full", "post_min"], default="full",
                    help="Where to compute quantile thresholds: full dataset or after min_minutes filter.")
    ap.add_argument("--h_label", type=str, default="H=1")
    ap.add_argument("--no_shade_groups", action="store_true",
                    help="Disable tuner-group ellipse shading in panel (b).")
    ap.add_argument("--pareto_smooth", action="store_true",
                    help="Light smoothing of the Pareto curve.")
    ap.add_argument("--pareto_window", type=int, default=3)

    args = ap.parse_args()

    points_csv = Path(args.points_csv)
    points_csv.parent.mkdir(parents=True, exist_ok=True)

    # ---- Compute or load points ----
    if points_csv.exists() and not args.recompute:
        print(f"[INFO] Using cached points file: {points_csv}")
        df_points = pd.read_csv(points_csv)
    else:
        if points_csv.exists():
            print(f"[INFO] --recompute set; overwriting {points_csv}")
        else:
            print(f"[INFO] No cached points file found; computing from cached predictions.")
        df_points = compute_points(args)
        df_points.to_csv(points_csv, index=False)
        print(f"[DONE] Saved points table: {points_csv}")

    # ---- Plot ----
    plot_pareto(df_points, args)


if __name__ == "__main__":
    main()
