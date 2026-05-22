
"""
Figure 4 (refined, as requested):

Three panels (a,b,c): RMSE, NSE, KGE
Each panel shows NORMALIZED attribution vs horizon (H=1..5) as a stackplot
(Main contributors normalized: NWP, Features, ML architecture, Tuner; replicates omitted)

Color coding (requested):
- NWP: purple
- Features: blue
- ML architecture: orange-yellow-ish
- Tuner: orange-red

Legend: BELOW the 3 plots (single shared legend)

Outputs:
- figures/Figure4_main_effects_RMSE_NSE_KGE_stackplot.png/.pdf
- figures/Figure4_main_effects_table.csv   (shares per metric per horizon)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

        long_ok = REQ_LONG_COLS_MIN.issubset(colset) and metric in colset
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

    # long
    if "horizon" in df.columns and metric in df.columns and set(REQ_FACTOR_COLS).issubset(df.columns):
        return df

    # wide -> long
    if set(REQ_FACTOR_COLS).issubset(df.columns):
        wide_cols = []
        for h in [1, 2, 3, 4, 5]:
            for name in (f"{metric}_h{h}", f"{metric}_H{h}"):
                if name in df.columns:
                    wide_cols.append((h, name))
                    break

        if wide_cols:
            base_cols = [
                c
                for c in df.columns
                if c in REQ_FACTOR_COLS
                or c in ["replicate", "lag", "run_dir", "keras_path", "run_id"]
            ]
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
            merged = met.merge(
                runs.drop_duplicates(keys),
                on=keys,
                how="inner",
                suffixes=("", "_run"),
            )
            if len(merged) > 0:
                return merged

    raise RuntimeError(
        "Could not merge runs and metrics tables. "
        "Ensure both contain compatible identifiers (model,nwp,feat,tuner[,replicate/lag]) or run_dir/keras_path/run_id."
    )


# -----------------------------
# ANOVA-style main effects
# -----------------------------

def _fit_ols_sse(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(np.sum(resid * resid))


def anova_main_effects_eta2(df_h: pd.DataFrame, y_col: str, factors: List[str]) -> Dict[str, float]:
    """
    Main-effect contributions via drop-one-factor partial SS:
      SS_factor = SSE(reduced) - SSE(full)
      eta2_factor = SS_factor / SST
    """
    y = df_h[y_col].astype(float).to_numpy()
    mask = np.isfinite(y)
    df2 = df_h.loc[mask, :].copy()
    y_full = df2[y_col].astype(float).to_numpy()

    if len(y_full) < 10:
        return {f: np.nan for f in factors}

    y_bar = float(np.mean(y_full))
    sst = float(np.sum((y_full - y_bar) ** 2))
    if sst <= 0:
        return {f: 0.0 for f in factors}

    # full design: intercept + dummies for all factors
    X_parts = [np.ones((len(df2), 1), dtype=float)]
    col_groups = {}
    start = 1
    for f in factors:
        d = pd.get_dummies(df2[f].astype(str), drop_first=True)
        arr = d.to_numpy(dtype=float)
        X_parts.append(arr)
        col_groups[f] = list(range(start, start + arr.shape[1]))
        start += arr.shape[1]

    X_full = np.concatenate(X_parts, axis=1)
    sse_full = _fit_ols_sse(X_full, y_full)

    out = {}
    for f in factors:
        keep_cols = [0] + [j for ff in factors if ff != f for j in col_groups[ff]]
        X_red = X_full[:, keep_cols]
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
# Shares per horizon (stackplot payload)
# -----------------------------

def shares_by_horizon(
    df: pd.DataFrame,
    metric: str,
    factors: List[str],
    horizons=(1, 2, 3, 4, 5),
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        d_h = df[df["horizon"] == h].copy()
        contrib = anova_main_effects_eta2(d_h, y_col=metric, factors=factors)

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
            raise FileNotFoundError(
                f"Could not auto-detect {metric} metrics CSV in {tables_dir}. "
                f"Pass --{metric.lower()}_csv."
            )
        return p

    rmse_csv = resolve_metric_csv("RMSE", args.rmse_csv)
    nse_csv = resolve_metric_csv("NSE", args.nse_csv)
    kge_csv = resolve_metric_csv("KGE", args.kge_csv)

    def load_merged(metric: str, m_csv: Path) -> pd.DataFrame:
        met = load_metrics_table(m_csv, metric=metric)
        df = smart_merge_runs_metrics(runs, met)

        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df = df.dropna(subset=["horizon", metric]).copy()
        df["horizon"] = df["horizon"].astype(int)
        df = df[df["horizon"].isin([1, 2, 3, 4, 5])].copy()

        for c in REQ_FACTOR_COLS:
            df[c] = df[c].astype(str)

        return df

    df_rmse = load_merged("RMSE", rmse_csv)
    df_nse = load_merged("NSE", nse_csv)
    df_kge = load_merged("KGE", kge_csv)

    factors = ["nwp", "feat", "model", "tuner"]
    labels = ["NWP", "Features", "ML architecture", "Tuner"]

    # Requested palette
    COLORS = {
        "nwp":   "#8B55A0A0",  # purple
        "feat":  "#3C81B39E",  # blue
        "model": "#E69D009D",  # orange-yellow
        "tuner": "#D55C009D",  # orange-red
    }

    set_nature_style()

    # Compute per-horizon shares (this is what you want for the stackplots)
    tab_rmse = shares_by_horizon(df_rmse, "RMSE", factors=factors)
    tab_nse = shares_by_horizon(df_nse, "NSE", factors=factors)
    tab_kge = shares_by_horizon(df_kge, "KGE", factors=factors)

    table_all = pd.concat([tab_rmse, tab_nse, tab_kge], ignore_index=True)
    out_table = fig_dir / "Figure4_main_effects_table.csv"
    table_all.to_csv(out_table, index=False)

    # -----------------------------
    # Figure: three panels (a,b,c) with stackplot
    # -----------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.6, 3.2),
        dpi=220,
        gridspec_kw={"wspace": 0.28},
    )

    panels = [
        ("a", "RMSE", tab_rmse),
        ("b", "NSE",  tab_nse),
        ("c", "KGE",  tab_kge),
    ]

    # Keep a nice 1:1.5 feel per panel
    for ax in axes:
        ax.set_box_aspect(6 / 7)

    # Shared legend handles
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[f], ec="none")
        for f in factors
    ]

    for ax, (letter, metric_name, tab) in zip(axes, panels):
        tab = tab.sort_values("horizon")
        x = tab["horizon"].to_numpy(dtype=int)
        y_stack = np.vstack([tab[f].to_numpy(dtype=float) for f in factors])

        ax.stackplot(
            x,
            y_stack,
            colors=[COLORS[f] for f in factors],
            alpha=0.58,
            linewidth=0.0,
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
        ax.text(-0.10, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=12, va="bottom")

    # Legend BELOW all panels (single shared legend)
    fig.legend(
        legend_handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.07),
    )

    # leave room for legend at bottom
    fig.subplots_adjust(bottom=0.18)

    out_png = fig_dir / "Figure4_main_effects_RMSE_NSE_KGE_stackplot.png"
    out_pdf = fig_dir / "Figure4_main_effects_RMSE_NSE_KGE_stackplot.pdf"
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
