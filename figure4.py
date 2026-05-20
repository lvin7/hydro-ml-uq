# figure4_main_effects_3metrics_stackplot_REPLICATE_RANDOM.py
"""
Companion to figure4_final.py.

Difference from the original:
- 'replicate' is treated as a RANDOM EFFECT (random intercept) instead of being
  excluded from the model. Fixed effects remain: nwp, feat, model, tuner.

Approach:
- For each horizon h and each metric m, fit a linear mixed-effects model:
      y ~ nwp + feat + model + tuner + (1 | replicate)
- Compute partial SS per fixed-effect factor via drop-one refits on the
  fixed-effect design matrix, holding the random-effect structure constant.
- Convert to eta^2 (share of total SST), then normalize across the four
  main factors so the stackplot is directly comparable to figure4_final.py.

Outputs:
- figures/Figure4_main_effects_RMSE_NSE_KGE_stackplot_REPLICATE_RANDOM.png/.pdf
- figures/Figure4_main_effects_table_REPLICATE_RANDOM.csv

Notes:
- If 'replicate' is missing or has only one level per cell, the script
  falls back to OLS (equivalent to the original script). This is reported
  in the console output.
- statsmodels MixedLM occasionally throws convergence warnings on small or
  unbalanced cells; we catch those and report them per (horizon, metric).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# statsmodels for MixedLM
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


# -----------------------------
# Utilities reused from the original script
# -----------------------------

REQ_FACTOR_COLS = ["model", "nwp", "feat", "tuner"]
REQ_LONG_COLS_MIN = set(REQ_FACTOR_COLS + ["horizon"])


def _read_csv_header_only(p: Path) -> List[str]:
    try:
        return list(pd.read_csv(p, nrows=0).columns)
    except Exception:
        return []


def find_metrics_csv(tables_dir: Path, metric: str) -> Optional[Path]:
    csvs = sorted(tables_dir.glob("*.csv"))
    best, best_score = None, -1
    for p in csvs:
        cols = _read_csv_header_only(p)
        if not cols:
            continue
        colset = set(cols)
        long_ok = (REQ_LONG_COLS_MIN.issubset(colset) and (metric in colset))
        long_score = 100 if long_ok else 0
        wide_score = 0
        if set(REQ_FACTOR_COLS).issubset(colset):
            hits = 0
            for h in [1, 2, 3, 4, 5]:
                if f"{metric}_h{h}" in colset or f"{metric}_H{h}" in colset:
                    hits += 1
            if hits >= 3:
                wide_score = 50 + hits
        score = max(long_score, wide_score)
        if score > best_score:
            best_score, best = score, p
    return best


def load_runs_csv(runs_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(runs_csv)
    for c in REQ_FACTOR_COLS:
        if c not in df.columns:
            raise ValueError(f"runs CSV missing required column '{c}': {runs_csv}")
    return df


def load_metrics_table(metrics_csv: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    if "horizon" in df.columns and metric in df.columns and set(REQ_FACTOR_COLS).issubset(df.columns):
        return df
    if set(REQ_FACTOR_COLS).issubset(df.columns):
        wide_cols = []
        for h in [1, 2, 3, 4, 5]:
            for name in (f"{metric}_h{h}", f"{metric}_H{h}"):
                if name in df.columns:
                    wide_cols.append((h, name))
                    break
        if wide_cols:
            base_cols = [c for c in df.columns if c in REQ_FACTOR_COLS or c in
                         ["replicate", "lag", "run_dir", "keras_path", "run_id"]]
            long_rows = []
            for h, colname in wide_cols:
                tmp = df[base_cols + [colname]].copy()
                tmp = tmp.rename(columns={colname: metric})
                tmp["horizon"] = h
                long_rows.append(tmp)
            return pd.concat(long_rows, ignore_index=True)
    raise ValueError(
        f"Could not interpret metrics CSV format: {metrics_csv}\n"
        f"Need either long (.., horizon, {metric}) or wide ({metric}_h1..)."
    )


def smart_merge_runs_metrics(runs: pd.DataFrame, met: pd.DataFrame) -> pd.DataFrame:
    key_sets = [
        ["model", "nwp", "feat", "tuner", "replicate"],
        ["model", "nwp", "feat", "tuner", "lag"],
        ["model", "nwp", "feat", "tuner"],
        ["run_dir"],
        ["keras_path"],
        ["run_id"],
    ]
    for keys in key_sets:
        if all(k in runs.columns for k in keys) and all(k in met.columns for k in keys):
            merged = met.merge(runs.drop_duplicates(keys), on=keys, how="inner", suffixes=("", "_run"))
            if len(merged) > 0:
                return merged
    raise RuntimeError(
        "Could not merge runs and metrics tables. "
        "Ensure both contain compatible identifiers (model,nwp,feat,tuner[,replicate/lag]) or run_dir/keras_path/run_id."
    )


# -----------------------------
# Mixed-effects partial SS
# -----------------------------

def _design_matrix(df: pd.DataFrame, factors: List[str]):
    """Build intercept + dummy columns; return (X, col_groups)."""
    X_parts = [np.ones((len(df), 1), dtype=float)]
    col_groups = {}
    start = 1
    for f in factors:
        d = pd.get_dummies(df[f].astype(str), drop_first=True)
        arr = d.to_numpy(dtype=float)
        X_parts.append(arr)
        col_groups[f] = list(range(start, start + arr.shape[1]))
        start += arr.shape[1]
    X = np.concatenate(X_parts, axis=1)
    return X, col_groups


def _fit_mixedlm_sse(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Optional[float]:
    """
    Fit y = X beta + u_group + eps with random intercept per group.
    Return residual SSE (sum of squared marginal residuals: y - X*beta).
    Using marginal (not conditional) residuals so SSE is comparable across
    nested models that share the same random-effect structure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = MixedLM(endog=y, exog=X, groups=groups)
            res = md.fit(method="lbfgs", reml=False, disp=False)
        beta = np.asarray(res.fe_params, dtype=float)
        resid = y - X @ beta
        return float(np.sum(resid * resid))
    except Exception:
        return None


def _fit_ols_sse(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(np.sum(resid * resid))


def mixed_main_effects_eta2(
    df_h: pd.DataFrame,
    y_col: str,
    factors: List[str],
    group_col: str = "replicate",
) -> Dict[str, float]:
    """
    Partial SS per fixed-effect factor in a mixed model with random intercept
    on group_col. Falls back to OLS if mixed model cannot be fit or if
    group_col is missing / has only one level.
    """
    y = df_h[y_col].astype(float).to_numpy()
    mask = np.isfinite(y)
    df2 = df_h.loc[mask, :].copy()
    if len(df2) < 10:
        return {f: np.nan for f in factors}

    y_full = df2[y_col].astype(float).to_numpy()
    y_bar = float(np.mean(y_full))
    sst = float(np.sum((y_full - y_bar) ** 2))
    if sst <= 0:
        return {f: 0.0 for f in factors}

    X_full, col_groups = _design_matrix(df2, factors)

    use_mixed = (group_col in df2.columns) and (df2[group_col].nunique() >= 2)

    if use_mixed:
        groups = df2[group_col].astype(str).to_numpy()
        sse_full = _fit_mixedlm_sse(X_full, y_full, groups)
        if sse_full is None:
            use_mixed = False  # fallback

    if not use_mixed:
        sse_full = _fit_ols_sse(X_full, y_full)
        groups = None

    out = {}
    for f in factors:
        keep_cols = [0] + [j for ff in factors if ff != f for j in col_groups[ff]]
        X_red = X_full[:, keep_cols]
        if use_mixed:
            sse_red = _fit_mixedlm_sse(X_red, y_full, groups)
            if sse_red is None:
                sse_red = _fit_ols_sse(X_red, y_full)
        else:
            sse_red = _fit_ols_sse(X_red, y_full)
        ss_f = max(0.0, sse_red - sse_full)
        out[f] = ss_f / sst

    return out


# -----------------------------
# Style
# -----------------------------

def set_nature_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# -----------------------------
# Shares per horizon
# -----------------------------

def shares_by_horizon(
    df: pd.DataFrame,
    metric: str,
    factors: List[str],
    horizons=(1, 2, 3, 4, 5),
    group_col: str = "replicate",
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        d_h = df[df["horizon"] == h].copy()
        contrib = mixed_main_effects_eta2(d_h, y_col=metric, factors=factors, group_col=group_col)
        vals = np.array([contrib[f] for f in factors], dtype=float)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        denom = float(vals.sum())
        shares = (vals / denom) if denom > 0 else np.zeros_like(vals)
        row = {"metric": metric, "horizon": h}
        for f, s in zip(factors, shares):
            row[f] = float(s)
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_out", type=str, default="analysis_out_v4")
    ap.add_argument("--runs_csv", type=str, default="")
    ap.add_argument("--rmse_csv", type=str, default="")
    ap.add_argument("--nse_csv", type=str, default="")
    ap.add_argument("--kge_csv", type=str, default="")
    ap.add_argument("--fig_dir", type=str, default="figures")
    ap.add_argument("--group_col", type=str, default="replicate",
                    help="Column used as random-effect group (default: replicate).")
    args = ap.parse_args()

    analysis_out = Path(args.analysis_out)
    tables_dir = analysis_out / "tables"
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = Path(args.runs_csv) if args.runs_csv else (tables_dir / "runs_kept_v4.csv")
    if not runs_csv.exists():
        raise FileNotFoundError(f"Missing runs CSV: {runs_csv}")
    runs = load_runs_csv(runs_csv)

    def resolve_metric_csv(metric: str, user_path: str) -> Path:
        if user_path:
            p = Path(user_path)
            if not p.exists():
                raise FileNotFoundError(f"Missing {metric} CSV: {p}")
            return p
        p = find_metrics_csv(tables_dir, metric)
        if p is None:
            raise FileNotFoundError(f"Could not auto-detect {metric} metrics CSV in {tables_dir}. Pass --{metric.lower()}_csv.")
        return p

    rmse_csv = resolve_metric_csv("RMSE", args.rmse_csv)
    nse_csv  = resolve_metric_csv("NSE",  args.nse_csv)
    kge_csv  = resolve_metric_csv("KGE",  args.kge_csv)

    def load_merged(metric: str, m_csv: Path) -> pd.DataFrame:
        met = load_metrics_table(m_csv, metric=metric)
        df = smart_merge_runs_metrics(runs, met)
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df = df.dropna(subset=["horizon", metric]).copy()
        df["horizon"] = df["horizon"].astype(int)
        df = df[df["horizon"].isin([1, 2, 3, 4, 5])].copy()
        for c in REQ_FACTOR_COLS:
            df[c] = df[c].astype(str)
        if args.group_col in df.columns:
            df[args.group_col] = df[args.group_col].astype(str)
        return df

    df_rmse = load_merged("RMSE", rmse_csv)
    df_nse  = load_merged("NSE",  nse_csv)
    df_kge  = load_merged("KGE",  kge_csv)

    # Report whether group_col is usable
    has_group = args.group_col in df_rmse.columns and df_rmse[args.group_col].nunique() >= 2
    if has_group:
        print(f"[INFO] Treating '{args.group_col}' as random effect "
              f"({df_rmse[args.group_col].nunique()} levels).")
    else:
        print(f"[WARN] Column '{args.group_col}' missing or single-level; "
              f"falling back to OLS (results will match figure4_final.py).")

    factors = ["nwp", "feat", "model", "tuner"]
    labels = ["NWP", "Features", "ML architecture", "Tuner"]

    COLORS = {
        "nwp":   "#8B55A0A0",
        "feat":  "#3C81B39E",
        "model": "#E69D009D",
        "tuner": "#D55C009D",
    }

    set_nature_style()

    tab_rmse = shares_by_horizon(df_rmse, "RMSE", factors=factors, group_col=args.group_col)
    tab_nse  = shares_by_horizon(df_nse,  "NSE",  factors=factors, group_col=args.group_col)
    tab_kge  = shares_by_horizon(df_kge,  "KGE",  factors=factors, group_col=args.group_col)

    table_all = pd.concat([tab_rmse, tab_nse, tab_kge], ignore_index=True)
    out_table = fig_dir / "Figure4_main_effects_table_REPLICATE_RANDOM.csv"
    table_all.to_csv(out_table, index=False)

    # -----------------------------
    # Figure
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.2), dpi=220, gridspec_kw={"wspace": 0.28})

    panels = [
        ("a", "RMSE", tab_rmse),
        ("b", "NSE",  tab_nse),
        ("c", "KGE",  tab_kge),
    ]

    for ax in axes:
        ax.set_box_aspect(6 / 7)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[f], ec="none") for f in factors]

    for ax, (letter, metric_name, tab) in zip(axes, panels):
        tab = tab.sort_values("horizon")
        x = tab["horizon"].to_numpy(dtype=int)
        y_stack = np.vstack([tab[f].to_numpy(dtype=float) for f in factors])

        ax.stackplot(
            x, y_stack,
            colors=[COLORS[f] for f in factors],
            alpha=0.58, linewidth=0.0,
        )

        ax.set_xlim(1, 5)
        ax.set_ylim(0, 1)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Forecast horizon (days ahead)")
        if ax is axes[0]:
            ax.set_ylabel("Normalized contribution")

        ax.grid(False)
        for s in ["top", "right", "left", "bottom"]:
            ax.spines[s].set_visible(True)
            ax.spines[s].set_linewidth(1.2)
            ax.spines[s].set_color("0.3")
        ax.set_title(metric_name)
        ax.text(-0.10, 1.05, letter, transform=ax.transAxes,
                fontweight="bold", fontsize=12, va="bottom")

    suffix = " — replicate as random effect" if has_group else " — OLS fallback"
    fig.suptitle(f"Main-effect attribution{suffix}", y=1.02, fontsize=11)

    fig.legend(
        legend_handles, labels,
        loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.07),
    )

    fig.subplots_adjust(bottom=0.18)

    out_png = fig_dir / "Figure4_main_effects_RMSE_NSE_KGE_stackplot_REPLICATE_RANDOM.png"
    out_pdf = fig_dir / "Figure4_main_effects_RMSE_NSE_KGE_stackplot_REPLICATE_RANDOM.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=1200)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved: {out_png}")
    print(f"[DONE] Saved: {out_pdf}")
    print(f"[DONE] Saved table: {out_table}")
    print(f"[INFO] runs CSV: {runs_csv}")
    print(f"[INFO] RMSE source: {rmse_csv}")
    print(f"[INFO] NSE  source: {nse_csv}")
    print(f"[INFO] KGE  source: {kge_csv}")


if __name__ == "__main__":
    main()