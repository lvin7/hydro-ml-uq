import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# -----------------------------
# Run folder parsing
# models/<Model>/nwp=.../feat=.../lag=.../<tuner>/
# -----------------------------
RUN_RE = re.compile(
    r"""
    (?P<model>[^/\\]+)
    [\\/]+nwp=(?P<nwp>[^/\\]+)
    [\\/]+feat=(?P<feat>[^/\\]+)
    [\\/]+lag=(?P<lag>\d+)
    [\\/]+(?P<tuner>[^/\\]+)
    $""",
    re.VERBOSE,
)

@dataclass(frozen=True)
class RunRecord:
    model: str
    nwp: str
    feat: str
    replicate: int   # lag folder label used as replicate
    tuner: str
    run_dir: Path
    keras_path: Path
    scalers_path: Path
    summary_path: Path
    tuning_time_min: float

    @property
    def run_id(self) -> str:
        return f"{self.model}_{self.nwp}_{self.feat}_{self.tuner}_{self.replicate}"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def scan_runs(root: str,
              models: Optional[List[str]] = None,
              nwps: Optional[List[str]] = None,
              feats: Optional[List[str]] = None,
              replicates: Optional[List[int]] = None,
              tuners: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Scan for runs that have exactly what we need:
      - model-<Model>.keras
      - model-<Model>.scalers.pkl
      - tuning_summary.json
    """
    root = Path(root)
    rows = []

    for d in root.rglob("*"):
        if not d.is_dir():
            continue

        rel = str(d.relative_to(root)).replace("\\", "/")
        m = RUN_RE.search(rel)
        if not m:
            continue

        model = m.group("model")
        nwp = m.group("nwp")
        feat = m.group("feat")
        rep = int(m.group("lag"))
        tuner = m.group("tuner")

        if models is not None and model not in set(models):
            continue
        if nwps is not None and nwp not in set(nwps):
            continue
        if feats is not None and feat not in set(feats):
            continue
        if replicates is not None and rep not in set(replicates):
            continue
        if tuners is not None and tuner not in set(tuners):
            continue

        keras_path = d / f"model-{model}.keras"
        scalers_path = d / f"model-{model}.scalers.pkl"
        summary_path = d / "tuning_summary.json"

        if not (keras_path.exists() and scalers_path.exists() and summary_path.exists()):
            continue

        summ = _read_json(summary_path)
        tmin = summ.get("total_tuning_time(min)", np.nan)

        rows.append({
            "model": model,
            "nwp": nwp,
            "feat": feat,
            "replicate": rep,
            "tuner": tuner,
            "run_dir": str(d),
            "keras_path": str(keras_path),
            "scalers_path": str(scalers_path),
            "tuning_summary_path": str(summary_path),
            "total_tuning_time_min": tmin,
            "run_id": f"{model}_{nwp}_{feat}_{tuner}_{rep}"
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No runs found containing (.keras, .scalers.pkl, tuning_summary.json) in expected folder structure.")
    return df


# -----------------------------
# Custom objects (TCN/TKAN/Loss)
# -----------------------------
def build_custom_objects():
    """
    Import your custom objects from your codebase.
    Adjust imports if your files differ.
    """
    from tcn import TCN
    from tkan import TKAN
    try:
        from models import PinballLoss
    except Exception:
        PinballLoss = None

    custom = {
        "TCN": TCN,
        "Custom>TCN": TCN,
        "TKAN": TKAN,
        "Custom>TKAN": TKAN,
    }
    if PinballLoss is not None:
        custom["PinballLoss"] = PinballLoss
        custom["Custom>PinballLoss"] = PinballLoss
    return custom


def load_model_for_inference(path: str, custom_objects: dict):
    return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)


# -----------------------------
# Data prep cache
# -----------------------------
def build_data_cache(df_runs: pd.DataFrame,
                     target: str,
                     val_start: str,
                     test_start: str):
    """
    Cache train/val/test splits per (nwp, feat).
    NOTE: lag/replicate does not affect data in your current training code,
    so it should not affect analysis data either.
    """
    from data_utils import data_prep

    cache = {}
    keys = df_runs[["nwp", "feat"]].drop_duplicates().itertuples(index=False, name=None)
    for (nwp, feat) in keys:
        X_train, y_train, X_val, y_val, X_test, y_test, _scalers = data_prep(
            nwp=nwp, target=target, vars=feat,
            val_start=val_start, test_start=test_start,
        )
        cache[(nwp, feat)] = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }
    return cache


# -----------------------------
# Prediction caching
# -----------------------------
def pred_cache_path(pred_cache_dir: str, run_id: str) -> Path:
    d = Path(pred_cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"pred__{run_id}.npz"


def save_prediction(pred_cache_dir: str, run_id: str, y_pred: np.ndarray):
    p = pred_cache_path(pred_cache_dir, run_id)
    np.savez_compressed(p, y_pred=y_pred)


def load_prediction(pred_cache_dir: str, run_id: str) -> Optional[np.ndarray]:
    p = pred_cache_path(pred_cache_dir, run_id)
    if not p.exists():
        return None
    try:
        z = np.load(p, allow_pickle=False)
        return z["y_pred"]
    except Exception:
        return None


# -----------------------------
# Metrics (deterministic per model)
# -----------------------------
def _safe_mask(a, b):
    return np.isfinite(a) & np.isfinite(b)


def mae(y, yhat) -> float:
    m = _safe_mask(y, yhat)
    return float(np.mean(np.abs(yhat[m] - y[m]))) if np.any(m) else np.nan


def rmse(y, yhat) -> float:
    m = _safe_mask(y, yhat)
    return float(np.sqrt(np.mean((yhat[m] - y[m]) ** 2))) if np.any(m) else np.nan


def mape(y, yhat, eps=1e-6) -> float:
    m = _safe_mask(y, yhat) & (np.abs(y) > eps)
    return float(np.mean(np.abs((yhat[m] - y[m]) / y[m]))) if np.any(m) else np.nan


def max_err(y, yhat) -> float:
    m = _safe_mask(y, yhat)
    return float(np.max(np.abs(yhat[m] - y[m]))) if np.any(m) else np.nan


def pearson_r(y, yhat) -> float:
    m = _safe_mask(y, yhat)
    if not np.any(m):
        return np.nan
    yy = y[m]; hh = yhat[m]
    if np.std(yy) == 0 or np.std(hh) == 0:
        return np.nan
    return float(np.corrcoef(yy, hh)[0, 1])


def nse(y, yhat) -> float:
    m = _safe_mask(y, yhat)
    if not np.any(m):
        return np.nan
    yy = y[m]; hh = yhat[m]
    denom = np.sum((yy - np.mean(yy)) ** 2)
    if denom == 0:
        return np.nan
    return float(1.0 - np.sum((yy - hh) ** 2) / denom)


def kge(y, yhat) -> float:
    """
    Kling-Gupta Efficiency (2009)
    KGE = 1 - sqrt( (r-1)^2 + (alpha-1)^2 + (beta-1)^2 )
    alpha = std(sim)/std(obs)
    beta  = mean(sim)/mean(obs)
    """
    m = _safe_mask(y, yhat)
    if not np.any(m):
        return np.nan
    yy = y[m]; hh = yhat[m]
    r = pearson_r(yy, hh)
    if not np.isfinite(r):
        return np.nan
    sy = np.std(yy); sh = np.std(hh)
    my = np.mean(yy); mh = np.mean(hh)
    if sy == 0 or my == 0:
        return np.nan
    alpha = sh / sy
    beta = mh / my
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def ve(y, yhat, eps=1e-12) -> float:
    """
    Volume Error (relative): (sum(sim) - sum(obs)) / sum(obs)
    """
    m = _safe_mask(y, yhat)
    if not np.any(m):
        return np.nan
    yy = y[m]; hh = yhat[m]
    denom = np.sum(yy)
    if np.abs(denom) < eps:
        return np.nan
    return float((np.sum(hh) - np.sum(yy)) / denom)


def atpe_2pct(y, yhat, peak_mask) -> float:
    """
    ATPE-2%: fraction of peak points where |error| <= 2% of observed.
    peak_mask: boolean mask over time indices (same length as y)
    """
    m = _safe_mask(y, yhat) & peak_mask
    if not np.any(m):
        return np.nan
    tol = 0.02 * np.abs(y[m])
    ok = np.abs(yhat[m] - y[m]) <= tol
    return float(np.mean(ok))


def dt_peak(y, yhat) -> float:
    """
    dTpeak: time index difference between global observed peak and predicted peak.
    """
    m = _safe_mask(y, yhat)
    if not np.any(m):
        return np.nan
    yy = y[m]; hh = yhat[m]
    # Map back to original indices
    idx = np.where(m)[0]
    t_obs = idx[int(np.argmax(yy))]
    t_sim = idx[int(np.argmax(hh))]
    return float(t_sim - t_obs)


def _robust_sigma_mad(e: np.ndarray, eps: float = 1e-12) -> float:
    """
    Robust scale estimate using MAD:
      sigma = 1.4826 * median(|e - median(e)|)
    """
    e = np.asarray(e)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return np.nan
    med = np.median(e)
    mad = np.median(np.abs(e - med))
    sigma = 1.4826 * mad
    return float(max(sigma, eps))


def scas(y, yhat, *, max_iter: int = 60, tol: float = 1e-6) -> float:
    """
    Self-Consistent Agreement Score (SCAS).

    Solve tau = f(tau), where
      f(tau) = fraction(|e| <= (1 - tau) * sigma),
    with sigma a robust scale of residuals (MAD).
    """
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)

    m = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return np.nan

    e = (y - yhat)[m]
    if e.size == 0:
        return np.nan

    sigma = _robust_sigma_mad(e)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan

    ae = np.abs(e)

    def f(tau: float) -> float:
        thr = (1.0 - tau) * sigma
        # if thr is ~0, only exact matches count
        return float(np.mean(ae <= thr))

    def g(tau: float) -> float:
        return f(tau) - tau

    a, b = 0.0, 1.0
    ga, gb = g(a), g(b)

    # Edge cases: no sign change -> return best feasible endpoint
    # If g(a) <= 0, fixed point is at 0 (or negative, but tau in [0,1])
    if ga <= 0.0:
        return 0.0
    # If g(b) >= 0, fixed point is at 1 (rare; would mean many exact zeros)
    if gb >= 0.0:
        return 1.0

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        gm = g(mid)

        if abs(gm) < tol or (b - a) < tol:
            return float(mid)

        # g is typically decreasing; maintain bracket where sign changes
        if gm > 0.0:
            a = mid
        else:
            b = mid

    return float(0.5 * (a + b))



def compute_metrics_per_horizon(y_true_2d: np.ndarray,
                                y_pred_2d: np.ndarray,
                                peak_q: float = 0.90) -> Dict[int, Dict[str, float]]:
    """
    Return dict keyed by horizon index h (0..H-1), each containing metric values.
    Peaks are defined by observed >= quantile(peak_q) for each horizon separately.
    """
    H = y_true_2d.shape[1]
    out = {}
    for h in range(H):
        y = y_true_2d[:, h]
        yhat = y_pred_2d[:, h]
        m = _safe_mask(y, yhat)
        if not np.any(m):
            out[h] = {k: np.nan for k in ["MAE","RMSE","MAPE","MAX","r","NSE","KGE","SCAS","ATPE2","dTpeak","VE"]}
            continue

        yy = y[m]
        thr = np.quantile(yy, peak_q) if len(yy) > 0 else np.nan
        peak_mask = np.zeros_like(y, dtype=bool)
        if np.isfinite(thr):
            peak_mask = (y >= thr)

        out[h] = {
            "MAE": mae(y, yhat),
            "RMSE": rmse(y, yhat),
            "MAPE": mape(y, yhat),
            "MAX": max_err(y, yhat),
            "r": pearson_r(y, yhat),
            "NSE": nse(y, yhat),
            "KGE": kge(y, yhat),
            "SCAS": scas(y, yhat),
            "ATPE2": atpe_2pct(y, yhat, peak_mask),
            "dTpeak": dt_peak(y, yhat),
            "VE": ve(y, yhat),
        }
    return out


# -----------------------------
# Ensemble probabilistic metrics (per subset)
# -----------------------------
def interval_score(y, lo, hi, alpha=0.05) -> float:
    """
    Winkler interval score for central (1-alpha) interval.
    """
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(m):
        return np.nan
    yy = y[m]; L = lo[m]; U = hi[m]
    width = U - L
    under = (L - yy) * (yy < L)
    over = (yy - U) * (yy > U)
    return float(np.mean(width + (2.0/alpha)*(under + over)))


def picp(y, lo, hi) -> float:
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(m):
        return np.nan
    ok = (y[m] >= lo[m]) & (y[m] <= hi[m])
    return float(np.mean(ok))


def crps_from_samples(y: np.ndarray, samples: np.ndarray) -> float:
    """
    CRPS for an empirical ensemble via energy score identity:
      CRPS = E|X - y| - 0.5 E|X - X'|
    samples: shape (M, N) for a single horizon
    y: shape (N,)
    """
    # finite mask per time
    if samples.ndim != 2:
        raise ValueError("samples must be (M, N)")
    M, N = samples.shape
    y = np.asarray(y).reshape(-1)
    if len(y) != N:
        raise ValueError("y must have length N matching samples second dim")

    # Mask times where all ensemble members finite and y finite
    m_time = np.isfinite(y)
    m_time &= np.all(np.isfinite(samples), axis=0)
    if not np.any(m_time):
        return np.nan

    S = samples[:, m_time]  # (M, Nt)
    yt = y[m_time]          # (Nt,)

    # E|X - y|
    term1 = np.mean(np.abs(S - yt[None, :]))

    # E|X - X'| computed by pairwise differences (O(M^2)) — M can be up to 1024.
    # Use a cheaper approximation: sample K pairs.
    M = S.shape[0]
    K = min(5000, M * 20)  # adjustable
    rng = np.random.default_rng(123)
    i1 = rng.integers(0, M, size=K)
    i2 = rng.integers(0, M, size=K)
    term2 = np.mean(np.abs(S[i1] - S[i2]))

    return float(term1 - 0.5 * term2)


def ensemble_summary(y_true_2d: np.ndarray,
                     preds_stack_3d: np.ndarray,
                     alpha: float = 0.05) -> Dict[int, Dict[str, float]]:
    """
    y_true_2d: (N,H)
    preds_stack_3d: (M,N,H) for subset of M models
    Return per horizon: CRPS, IntervalScore, PICP
    """
    M, N, H = preds_stack_3d.shape
    out = {}
    lo_q = alpha/2 * 100
    hi_q = (1 - alpha/2) * 100
    for h in range(H):
        y = y_true_2d[:, h]
        S = preds_stack_3d[:, :, h]  # (M,N)

        lo = np.nanpercentile(S, lo_q, axis=0)
        hi = np.nanpercentile(S, hi_q, axis=0)

        out[h] = {
            "CRPS": crps_from_samples(y, S),
            "IntervalScore": interval_score(y, lo, hi, alpha=alpha),
            "PICP": picp(y, lo, hi),
        }
    return out


# -----------------------------
# Plot helpers
# -----------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_violin(df: pd.DataFrame, metric: str, horizon: int, out_png: str, title: str):
    """
    df has columns: metric values + cap label
    """

    caps = sorted(df["cap"].unique(), key=lambda x: int(x.replace("%","")))
    data = [df[(df["cap"] == c) & (df["horizon"] == horizon)][metric].dropna().values for c in caps]

    plt.figure(figsize=(10, 5))
    plt.violinplot(data, showmeans=True, showmedians=False, showextrema=True)
    plt.xticks(range(1, len(caps)+1), caps)
    plt.title(title)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_unified_ts(y_obs: np.ndarray,
                    mean_pred: np.ndarray,
                    lo: np.ndarray,
                    hi: np.ndarray,
                    out_png: str,
                    title: str):

    plt.figure(figsize=(16, 6))
    plt.plot(y_obs, linewidth=2, label="Observed")
    plt.plot(mean_pred, linewidth=2, label="Ensemble mean")
    plt.fill_between(np.arange(len(y_obs)), lo, hi, alpha=0.25, label="95% band")
    plt.title(title)
    plt.xlabel("Test timestep index")
    plt.ylabel("Discharge")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

import numpy as np
import pandas as pd

def save_metric_vs_horizon_violin(
    df: pd.DataFrame,
    metric: str,
    out_png: str,
    title: str,
    horizons=(1,2,3,4,5),
    show_means=True
):
    """
    One figure per metric: 5 violins (horizons 1..5) for top-cut models.
    df must already be filtered to the desired subset.
    """

    data = []
    labels = []
    for h in horizons:
        v = df[df["horizon"] == h][metric].dropna().values
        if v.size == 0:
            v = np.array([np.nan])  # keep slot to preserve x-axis
        data.append(v)
        labels.append(str(h))

    # If everything is empty, skip
    if all((np.asarray(d).size == 0 or np.all(np.isnan(d))) for d in data):
        return

    plt.figure(figsize=(10, 5))
    plt.violinplot(data, showmeans=show_means, showmedians=not show_means, showextrema=True)
    plt.xticks(range(1, len(labels)+1), labels)
    plt.xlabel("Forecast horizon (days ahead)")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def save_grouped_boxplot(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    out_png: str,
    title: str,
    dropna=True,
    showfliers=False,
    max_groups=30
):
    """
    Boxplot of metric grouped by group_col (e.g. tuner, nwp, model, feat).
    """

    d = df[[group_col, metric]].copy()
    if dropna:
        d = d.dropna()

    # order groups by median (best on top depends on metric direction; we do median sort)
    meds = d.groupby(group_col)[metric].median().sort_values(ascending=False)
    groups = meds.index.tolist()[:max_groups]

    data = [d[d[group_col] == g][metric].values for g in groups]

    # skip if empty
    if len(data) == 0 or all(len(x) == 0 for x in data):
        return

    plt.figure(figsize=(max(10, 0.6*len(groups)), 5))
    plt.boxplot(data, labels=groups, showfliers=showfliers)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


# -----------------------------
# ANOVA-style attribution (variance decomposition)
# -----------------------------
def _anova_eta2_table(df: pd.DataFrame, y_col: str, factors: list) -> pd.DataFrame:
    """
    Compute simple ANOVA-style attribution using one-way sums of squares per factor:
      eta^2 = SS_factor / SS_total
    This is main-effects only and works without external stats packages.

    df: rows are observations (models)
    y_col: response variable
    factors: list of categorical factor column names
    """
    d = df[factors + [y_col]].dropna().copy()
    if len(d) < 10:
        return pd.DataFrame({"factor": factors, "eta2": np.nan, "ss": np.nan, "df": np.nan})

    y = d[y_col].values.astype(float)
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean)**2)
    if ss_total <= 0:
        return pd.DataFrame({"factor": factors, "eta2": 0.0, "ss": 0.0, "df": 0})

    rows = []
    for f in factors:
        # group means
        grp = d.groupby(f)[y_col]
        n_g = grp.size()
        mu_g = grp.mean()
        ss_f = np.sum(n_g * (mu_g - y_mean)**2)
        df_f = int(mu_g.shape[0] - 1)
        eta2 = float(ss_f / ss_total)
        rows.append({"factor": f, "eta2": eta2, "ss": float(ss_f), "df": df_f})

    out = pd.DataFrame(rows).sort_values("eta2", ascending=False).reset_index(drop=True)
    out["ss_total"] = float(ss_total)
    out["n"] = int(len(d))
    return out


def attribution_by_horizon(
    df_metrics: pd.DataFrame,
    metric: str,
    factors: list,
    horizons=(1,2,3,4,5)
) -> pd.DataFrame:
    """
    Returns long table:
      horizon, factor, eta2, ss, df, ss_total, n
    """
    out = []
    for h in horizons:
        dh = df_metrics[df_metrics["horizon"] == h].copy()
        tab = _anova_eta2_table(dh, metric, factors)
        tab.insert(0, "horizon", h)
        tab.insert(1, "metric", metric)
        out.append(tab)
    return pd.concat(out, ignore_index=True)


def save_attribution_stacked_area(
    df_attr: pd.DataFrame,
    out_png: str,
    title: str,
    factor_order: list = None
):
    """
    df_attr: output of attribution_by_horizon() filtered to one metric.
    Produces stacked area chart: x=horizon, y=cumulative eta2 contributions.
    """

    d = df_attr.copy()
    # pivot to horizon x factor
    piv = d.pivot_table(index="horizon", columns="factor", values="eta2", aggfunc="first").fillna(0.0)

    if factor_order is None:
        # order by average contribution
        factor_order = piv.mean(axis=0).sort_values(ascending=False).index.tolist()

    piv = piv[factor_order]

    x = piv.index.values
    ys = [piv[c].values for c in piv.columns]

    plt.figure(figsize=(9, 5))
    plt.stackplot(x, ys, labels=piv.columns)
    plt.xticks(x, [str(int(v)) for v in x])
    plt.ylim(0, 1)
    plt.xlabel("Forecast horizon (days ahead)")
    plt.ylabel("Explained variance fraction (η², main effects)")
    plt.title(title)
    plt.legend(loc="upper left", fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


# ========== v3 ===========
# -----------------------------
# Grouped violin plots
# -----------------------------
def save_grouped_violin(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    out_png: str,
    title: str,
    *,
    horizon: int | None = None,
    order_by_median_desc: bool = True,
    max_groups: int = 30,
):
    """
    Violin plot of metric grouped by group_col.
    Optionally filter to a single horizon. Uses df already filtered to top75.
    """
    import matplotlib.pyplot as plt

    d = df[[group_col, metric] + (["horizon"] if "horizon" in df.columns else [])].copy()
    if horizon is not None:
        d = d[d["horizon"] == horizon]
    d = d.dropna(subset=[group_col, metric])

    if len(d) == 0:
        return

    meds = d.groupby(group_col)[metric].median()
    meds = meds.sort_values(ascending=not order_by_median_desc)
    groups = meds.index.tolist()[:max_groups]

    data = [d[d[group_col] == g][metric].values for g in groups]
    if len(data) == 0 or all(len(x) == 0 for x in data):
        return

    plt.figure(figsize=(max(10, 0.6 * len(groups)), 5))
    plt.violinplot(data, showmeans=True, showmedians=False, showextrema=True)
    plt.xticks(range(1, len(groups) + 1), [str(g) for g in groups], rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


# -----------------------------
# Publishable compact group tables
# -----------------------------
def group_compact_table(
    df: pd.DataFrame,
    group_col: str,
    metrics: list[str],
    *,
    horizon: int | None = None,
) -> pd.DataFrame:
    """
    Returns compact table per group level:
      n, median, q25, q75 for each metric.
    Optionally filter to one horizon.
    """
    d = df.copy()
    if horizon is not None:
        d = d[d["horizon"] == horizon].copy()

    out_rows = []
    for lvl, g in d.groupby(group_col):
        row = {"level": str(lvl), "n": int(g.shape[0])}
        for m in metrics:
            x = g[m].dropna().values
            if x.size == 0:
                row[f"{m}_median"] = np.nan
                row[f"{m}_q25"] = np.nan
                row[f"{m}_q75"] = np.nan
            else:
                row[f"{m}_median"] = float(np.median(x))
                row[f"{m}_q25"] = float(np.quantile(x, 0.25))
                row[f"{m}_q75"] = float(np.quantile(x, 0.75))
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    # sort by NSE median if present
    if "NSE_median" in out.columns:
        out = out.sort_values("NSE_median", ascending=False, na_position="last")
    return out


# -----------------------------
# Normalized attribution helper
# -----------------------------
def normalize_attribution(df_attr: pd.DataFrame) -> pd.DataFrame:
    """
    df_attr: columns include [horizon, factor, eta2]
    Returns same rows with an extra column eta2_norm where each horizon sums to 1
    over included factors (main effects only).
    """
    d = df_attr.copy()
    denom = d.groupby("horizon")["eta2"].transform("sum")
    d["eta2_norm"] = np.where(denom > 0, d["eta2"] / denom, np.nan)
    return d


def save_attribution_stacked_area_norm(
    df_attr_norm: pd.DataFrame,
    out_png: str,
    title: str,
    factor_order: list[str] | None = None,
):
    """
    df_attr_norm must contain columns: horizon, factor, eta2_norm
    """
    import matplotlib.pyplot as plt

    d = df_attr_norm.copy()
    piv = d.pivot_table(index="horizon", columns="factor", values="eta2_norm", aggfunc="first").fillna(0.0)

    if factor_order is None:
        factor_order = piv.mean(axis=0).sort_values(ascending=False).index.tolist()
    piv = piv[factor_order]

    x = piv.index.values
    ys = [piv[c].values for c in piv.columns]

    plt.figure(figsize=(9, 5))
    plt.stackplot(x, ys, labels=piv.columns)
    plt.xticks(x, [str(int(v)) for v in x])
    plt.ylim(0, 1)
    plt.xlabel("Forecast horizon (days ahead)")
    plt.ylabel("Normalized explained variance (η² / Ση²)")
    plt.title(title)
    plt.legend(loc="upper left", fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


# -----------------------------
# OLS-based attribution with interactions (drop-one / semi-partial)
# -----------------------------
def _one_hot(df: pd.DataFrame, col: str, drop_first: bool = True) -> pd.DataFrame:
    return pd.get_dummies(df[col].astype("category"), prefix=col, drop_first=drop_first)


def _ols_sse(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(np.sum(resid ** 2))


def ols_attribution(
    df: pd.DataFrame,
    y_col: str,
    main_factors: list[str],
    interactions: list[tuple[str, str]] | None = None,
    *,
    drop_first: bool = True,
) -> dict:
    """
    Fit OLS with one-hot main effects (and optional pairwise interactions).
    Compute:
      - R2_full
      - contributions per TERM via drop-one: (SSE_reduced - SSE_full) / SST

    Returns dict with:
      terms: list[str]
      contrib: dict term->fraction_of_total_variance_explained_by_that_term
      R2_full, unexplained
      sum_contrib
    """
    d = df[main_factors + [y_col]].dropna().copy()
    if len(d) < 20:
        return {"terms": [], "contrib": {}, "R2_full": np.nan, "unexplained": np.nan, "sum_contrib": np.nan}

    y = d[y_col].values.astype(float)
    ymean = float(np.mean(y))
    sst = float(np.sum((y - ymean) ** 2))
    if sst <= 0:
        return {"terms": [], "contrib": {}, "R2_full": 0.0, "unexplained": 1.0, "sum_contrib": 0.0}

    # Build term matrices
    term_mats = {}
    for f in main_factors:
        term_mats[f] = _one_hot(d, f, drop_first=drop_first)

    if interactions:
        for a, b in interactions:
            A = term_mats[a]
            B = term_mats[b]
            # interaction columns = all pairwise products of dummy columns
            cols = {}
            for ca in A.columns:
                for cb in B.columns:
                    cols[f"{ca}:{cb}"] = (A[ca].values * B[cb].values)
            term_mats[f"{a}*{b}"] = pd.DataFrame(cols)

    # Full X = intercept + all term columns
    X_parts = [np.ones((len(d), 1))]
    term_cols = {}
    for t, mat in term_mats.items():
        if mat.shape[1] == 0:
            continue
        term_cols[t] = (len(np.hstack(X_parts)), len(np.hstack(X_parts)) + mat.shape[1])
        X_parts.append(mat.values.astype(float))

    X_full = np.hstack(X_parts)
    sse_full = _ols_sse(X_full, y)
    r2_full = 1.0 - (sse_full / sst)
    r2_full = float(np.clip(r2_full, 0.0, 1.0))

    contrib = {}
    for t, (i0, i1) in term_cols.items():
        # reduced = drop these columns
        keep = np.ones(X_full.shape[1], dtype=bool)
        keep[i0:i1] = False
        X_red = X_full[:, keep]
        sse_red = _ols_sse(X_red, y)
        delta = (sse_red - sse_full) / sst
        contrib[t] = float(max(0.0, delta))  # numerical safety

    sum_contrib = float(sum(contrib.values()))
    unexplained = float(max(0.0, 1.0 - r2_full))
    return {
        "terms": list(contrib.keys()),
        "contrib": contrib,
        "R2_full": r2_full,
        "unexplained": unexplained,
        "sum_contrib": sum_contrib,
    }


def ols_attribution_by_horizon(
    df_metrics: pd.DataFrame,
    metric: str,
    main_factors: list[str],
    interactions: list[tuple[str, str]] | None,
    horizons=(1,2,3,4,5),
) -> pd.DataFrame:
    """
    Returns long table:
      horizon, term, frac (of total variance), kind(main/interaction), R2_full, unexplained
    """
    rows = []
    for h in horizons:
        dh = df_metrics[df_metrics["horizon"] == h].copy()
        res = ols_attribution(dh, metric, main_factors, interactions)
        for term, frac in res["contrib"].items():
            rows.append({
                "horizon": int(h),
                "metric": metric,
                "term": term,
                "frac": frac,
                "kind": "interaction" if ("*" in term or ":" in term) and ("*" in term) else ("interaction" if "*" in term else "main"),
                "R2_full": res["R2_full"],
                "unexplained": res["unexplained"],
            })
        # Also store totals row for convenience
        rows.append({
            "horizon": int(h),
            "metric": metric,
            "term": "__TOTAL__",
            "frac": float(res["sum_contrib"]),
            "kind": "total_explained",
            "R2_full": res["R2_full"],
            "unexplained": res["unexplained"],
        })
    return pd.DataFrame(rows)


# -----------------------------
# Simple radar chart (use sparingly)
# -----------------------------
def save_radar_chart(
    df_summary: pd.DataFrame,
    label_col: str,
    metric_cols: list[str],
    out_png: str,
    title: str,
    *,
    higher_is_better: dict[str, bool] | None = None,
):
    """
    df_summary: one row per group (e.g. per tuner) with already aggregated metrics.
    We normalize each metric to [0,1] across groups (invert if lower is better).
    """

    d = df_summary.copy()
    if len(d) < 2:
        return

    higher_is_better = higher_is_better or {}
    M = []
    for m in metric_cols:
        x = d[m].astype(float).values
        # invert if lower is better
        hib = higher_is_better.get(m, True)
        if not hib:
            x = -x
        # min-max normalize
        mn, mx = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
            z = np.zeros_like(x)
        else:
            z = (x - mn) / (mx - mn)
        M.append(z)

    M = np.vstack(M).T  # (G, K)
    labels = d[label_col].astype(str).values
    K = len(metric_cols)

    angles = np.linspace(0, 2*np.pi, K, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for i in range(M.shape[0]):
        vals = M[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=labels[i])
        ax.fill(angles, vals, alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_cols)
    ax.set_ylim(0, 1)
    plt.title(title, y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()
