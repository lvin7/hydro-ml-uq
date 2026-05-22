"""
Utilities used by full_pipeline_analysis.py and generating figures.

Contents
--------
- Run-folder scanning (scan_runs, RUN_RE)
- Custom-object handling for model loading (build_custom_objects, load_model_for_inference)
- Test-set data preparation cache (build_data_cache)
- Prediction caching (save_prediction, load_prediction, pred_cache_path)
- Deterministic metrics (mae, rmse, mape, max_err, pearson_r, nse, kge, ve,
  atpe_2pct, dt_peak, scas) and per-horizon helper (compute_metrics_per_horizon)
- Probabilistic metrics (crps_from_samples, interval_score, picp) and per-horizon
  helper (ensemble_summary)
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf


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
    Scan for run directories that contain the three artifacts we need:
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
        raise RuntimeError(
            "No runs found containing (.keras, .scalers.pkl, tuning_summary.json) "
            "in expected folder structure."
        )
    return df


# -----------------------------
# Custom objects (TCN/TKAN/Loss)
# -----------------------------
def build_custom_objects() -> dict:
    """
    Build the custom_objects dict needed by tf.keras.models.load_model
    for models that contain TCN/TKAN layers and/or PinballLoss.
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
    """Load a saved Keras model for inference (compile=False)."""
    return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)


# -----------------------------
# Data prep cache
# -----------------------------
def build_data_cache(df_runs: pd.DataFrame,
                     target: str,
                     val_start: str,
                     test_start: str) -> dict:
    """
    Build a cache of train/val/test splits keyed by (nwp, feat).

    Lag/replicate does not affect data preparation in the training code,
    so it is not part of the cache key.
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
    """Return the cached-prediction path for a given run_id."""
    d = Path(pred_cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"pred__{run_id}.npz"


def save_prediction(pred_cache_dir: str, run_id: str, y_pred: np.ndarray) -> None:
    """Save a model's test-set prediction to the prediction cache."""
    p = pred_cache_path(pred_cache_dir, run_id)
    np.savez_compressed(p, y_pred=y_pred)


def load_prediction(pred_cache_dir: str, run_id: str) -> Optional[np.ndarray]:
    """Load a cached prediction. Returns None if it does not exist."""
    p = pred_cache_path(pred_cache_dir, run_id)
    if not p.exists():
        return None
    try:
        z = np.load(p, allow_pickle=False)
        return z["y_pred"]
    except Exception:
        return None


# -----------------------------
# Deterministic metrics (per model)
# -----------------------------
def _safe_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def mae(y, yhat) -> float:
    m = _safe_mask(y, yhat)
    return float(np.mean(np.abs(yhat[m] - y[m]))) if np.any(m) else np.nan


def rmse(y, yhat) -> float:
    m = _safe_mask(y, yhat)
    return float(np.sqrt(np.mean((yhat[m] - y[m]) ** 2))) if np.any(m) else np.nan


def mape(y, yhat, eps: float = 1e-6) -> float:
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
    """Nash-Sutcliffe Efficiency."""
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
    Kling-Gupta Efficiency (Gupta et al. 2009).

        KGE = 1 - sqrt( (r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2 )

    where alpha = std(sim) / std(obs) and beta = mean(sim) / mean(obs).
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


def ve(y, yhat, eps: float = 1e-12) -> float:
    """Relative volume error: (sum(sim) - sum(obs)) / sum(obs)."""
    m = _safe_mask(y, yhat)
    if not np.any(m):
        return np.nan
    yy = y[m]; hh = yhat[m]
    denom = np.sum(yy)
    if np.abs(denom) < eps:
        return np.nan
    return float((np.sum(hh) - np.sum(yy)) / denom)


def compute_metrics_per_horizon(y_true_2d: np.ndarray,
                                y_pred_2d: np.ndarray,
                                peak_q: float = 0.90) -> Dict[int, Dict[str, float]]:
    """
    Compute deterministic metrics for each horizon independently.

    Returns a dict keyed by horizon index h (0..H-1).
    Peaks are defined per horizon as observed >= quantile(peak_q).
    """
    H = y_true_2d.shape[1]
    out = {}
    metric_keys = ["MAE", "RMSE", "MAPE", "MAX", "r", "NSE", "KGE", "VE"]

    for h in range(H):
        y = y_true_2d[:, h]
        yhat = y_pred_2d[:, h]
        m = _safe_mask(y, yhat)
        if not np.any(m):
            out[h] = {k: np.nan for k in metric_keys}
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
            "VE": ve(y, yhat),
        }
    return out


# -----------------------------
# Probabilistic metrics (per ensemble subset)
# -----------------------------
def interval_score(y, lo, hi, alpha: float = 0.05) -> float:
    """Winkler interval score for the central (1 - alpha) interval."""
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(m):
        return np.nan
    yy = y[m]; L = lo[m]; U = hi[m]
    width = U - L
    under = (L - yy) * (yy < L)
    over = (yy - U) * (yy > U)
    return float(np.mean(width + (2.0 / alpha) * (under + over)))


def picp(y, lo, hi) -> float:
    """Prediction-interval coverage probability."""
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(m):
        return np.nan
    ok = (y[m] >= lo[m]) & (y[m] <= hi[m])
    return float(np.mean(ok))


def crps_from_samples(y: np.ndarray, samples: np.ndarray) -> float:
    """
    CRPS for an empirical ensemble via the energy-score identity:

        CRPS = E|X - y| - 0.5 * E|X - X'|

    The second term is estimated by sampling pairs (K = min(5000, 20*M)).

    samples: (M, N) for a single horizon.
    y: (N,).
    """
    if samples.ndim != 2:
        raise ValueError("samples must be (M, N)")
    M, N = samples.shape
    y = np.asarray(y).reshape(-1)
    if len(y) != N:
        raise ValueError("y must have length N matching samples second dim")

    m_time = np.isfinite(y) & np.all(np.isfinite(samples), axis=0)
    if not np.any(m_time):
        return np.nan

    S = samples[:, m_time]
    yt = y[m_time]

    term1 = np.mean(np.abs(S - yt[None, :]))

    K = min(5000, M * 20)
    rng = np.random.default_rng(123)
    i1 = rng.integers(0, M, size=K)
    i2 = rng.integers(0, M, size=K)
    term2 = np.mean(np.abs(S[i1] - S[i2]))

    return float(term1 - 0.5 * term2)


def ensemble_summary(y_true_2d: np.ndarray,
                     preds_stack_3d: np.ndarray,
                     alpha: float = 0.05) -> Dict[int, Dict[str, float]]:
    """
    Probabilistic metrics per horizon: CRPS, IntervalScore, PICP.

    y_true_2d: (N, H).
    preds_stack_3d: (M, N, H) — ensemble predictions for the M-member subset.
    """
    M, N, H = preds_stack_3d.shape
    out = {}
    lo_q = alpha / 2 * 100
    hi_q = (1 - alpha / 2) * 100
    for h in range(H):
        y = y_true_2d[:, h]
        S = preds_stack_3d[:, :, h]
        lo = np.nanpercentile(S, lo_q, axis=0)
        hi = np.nanpercentile(S, hi_q, axis=0)
        out[h] = {
            "CRPS": crps_from_samples(y, S),
            "IntervalScore": interval_score(y, lo, hi, alpha=alpha),
            "PICP": picp(y, lo, hi),
        }
    return out
