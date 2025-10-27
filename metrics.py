import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error

# -----------------------
# Metrics helper functions
# -----------------------

def plot_loss(history, ylim=None, save_path=''):
    plt.figure(figsize=(14, 7))
    plt.title('Model loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    if ylim!=None:
      plt.ylim(0, ylim)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig(f'{save_path}/loss.png')


def scatter_plot(y_pred, y_true, model_name='_ Model', save_path=''):
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true) & ~np.isinf(y_pred) & ~np.isinf(y_true)
    y_pred_clean = y_pred[mask].reshape(-1, y_pred.shape[1])
    y_obs_clean = y_true[mask].reshape(-1, y_true.shape[1])

    y_pred_flat = y_pred_clean.flatten()
    y_obs_flat = y_obs_clean.flatten()

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f'{model_name} Scatter Plot', fontsize=16, y=0.8, x=0.4)
    ax_scatter = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3)

    for i in range(5):
        ax_scatter.scatter(y_obs_clean[:, i], y_pred_clean[:, i],
                            label=f'{i + 1} day(s) ahead', alpha=0.3, s=25)

    ax_scatter.set_xlabel('Observed discharge (m³/s)', fontsize=14)
    ax_scatter.set_ylabel('Predicted discharge (m³/s)', fontsize=14)
    max_val = max(y_obs_clean.max(), y_pred_clean.max())
    ax_scatter.plot([0, max_val], [0, max_val], 'k-', label='45 degree line')

    model = LinearRegression()
    model.fit(y_obs_flat.reshape(-1, 1), y_pred_flat)
    predicted_regression = model.predict(y_obs_flat.reshape(-1, 1))
    ax_scatter.plot(y_obs_flat, predicted_regression, 'r--', lw=2, label='Mean Regression Line')
    ax_scatter.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{save_path}/scatter.png')
    plt.close()


def scatter_plot_1dah(y_pred, y_true, model_name='_Model', save_path=''):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true[:,0], y_pred[:,0], alpha=0.5)

    reg = LinearRegression().fit(y_true[:, [0]], y_pred[:, 0])
    y_line = reg.predict(y_true[:, [0]])

    plt.plot(y_true[:,0], y_line, 'r--', lw=2, label='Regression')
    plt.plot(y_true[:,0], y_true[:,0], 'k-', lw=1, label='1:1')
    plt.legend()
    plt.title(model_name)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{save_path}/scatter_1dah.png')
    plt.close()


def metrics_table(y_pred, y_true):
    threshold = np.percentile(y_true, 75)
    metrics = []

    for i in range(y_pred.shape[1]):
        forecast = y_pred[:, i]
        observed = y_true[:, i]
        mask = observed >= threshold
        observed_peaks = observed[mask]
        forecast_peaks = forecast[mask]

        full_metrics = {
            'Day Ahead': f'{i + 1} Day(s)',
            'MAE (Full)': mean_absolute_error(observed, forecast),
            'MAPE (Full)': mean_absolute_percentage_error(observed, forecast),
            'RMSE (Full)': np.sqrt(mean_squared_error(observed, forecast)),
            'Max Error (Full)': np.max(np.abs(observed - forecast)),
            'NSE (Full)': r2_score(observed, forecast)
        }

        peak_metrics = {
            'MAE (Peaks)': mean_absolute_error(observed_peaks, forecast_peaks),
            'MAPE (Peaks)': mean_absolute_percentage_error(observed_peaks, forecast_peaks),
            'RMSE (Peaks)': np.sqrt(mean_squared_error(observed_peaks, forecast_peaks)),
            'Max Error (Peaks)': np.max(np.abs(observed_peaks - forecast_peaks)),
            'NSE (Peaks)': r2_score(observed_peaks, forecast_peaks)
        }

        metrics.append({**full_metrics, **peak_metrics})

    return pd.DataFrame(metrics)


# ====================================== ADDITIONAL METRICS FOR LATER USE ==================================

# -----------------------
# Helpers
# -----------------------

def _nan_mask2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mask where both a and b are finite."""
    return np.isfinite(a) & np.isfinite(b)

def _nan_mask3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b) & np.isfinite(c)

def _as_2d(x: np.ndarray) -> np.ndarray:
    """Ensure (N, H) shape (H=1 if originally 1D)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    return x

def _pinball_mat(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    e = y - q
    return np.maximum(tau * e, (tau - 1.0) * e)

# -----------------------
# Probabilistic metrics
# -----------------------

def picp(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    """Prediction Interval Coverage Probability: P(q_lo <= y <= q_hi)."""
    yt, lo, hi = _as_2d(y_true), _as_2d(q_lo), _as_2d(q_hi)
    m = _nan_mask3(yt, lo, hi)
    if not np.any(m):
        return np.nan
    ok = (yt[m] >= lo[m]) & (yt[m] <= hi[m])
    return float(np.mean(ok))

def mpiw(q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    lo, hi = _as_2d(q_lo), _as_2d(q_hi)
    m = _nan_mask2(lo, hi)
    if not np.any(m):
        return np.nan
    return float(np.mean(hi[m] - lo[m]))

def interval_score(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray, alpha: float = 0.1) -> float:
    """
    Interval (Winkler) score for a (1-alpha) central interval.
    Lower is better. Rewards narrow intervals that cover the truth.
    """
    yt, lo, hi = _as_2d(y_true), _as_2d(q_lo), _as_2d(q_hi)
    m = _nan_mask3(yt, lo, hi)
    if not np.any(m):
        return np.nan
    y = yt[m]; L = lo[m]; U = hi[m]
    width = U - L
    under = (L - y) * (y < L)
    over  = (y - U) * (y > U)
    return float(np.mean(width + (2.0/alpha)*(under + over)))

def crps_from_quantiles(y_true: np.ndarray, q_grid: np.ndarray, q_values: np.ndarray) -> float:
    """
    Approximate CRPS by integrating pinball loss over a set of quantiles.
    q_grid: (K,) monotonically increasing in (0,1)
    q_values: (N, K) or (N, H, K)
    """
    yt = _as_2d(y_true)  # (N,H)
    qv = np.asarray(q_values)
    if qv.ndim == 2:  # (N,K) -> (N,1,K)
        qv = qv[:, None, :]
    if qv.ndim != 3 or qv.shape[2] != len(q_grid):
        raise ValueError("q_values must be (N,K) or (N,H,K) matching len(q_grid).")
    # trapezoidal integration over tau in (0,1) of pinball
    taus = np.asarray(q_grid).astype(float)
    taus = np.clip(taus, 1e-6, 1-1e-6)
    # compute pinball for each tau
    losses = []
    for k, tau in enumerate(taus):
        losses.append(_pinball_mat(yt, qv[..., k], tau))
    losses = np.stack(losses, axis=-1)  # (N,H,K)
    # integrate over tau
    crps = np.trapezoid(losses, taus, axis=-1)  # (N,H)
    m = np.isfinite(crps)
    return float(np.mean(crps[m])) if np.any(m) else np.nan


# -----------------------
# MC‑Dropout utilities
# -----------------------

def mc_mean_q(y_mc: np.ndarray, lower: float = 5.0, upper: float = 95.0):
    """
    Given MC samples y_mc with shape (T, N) or (T, N, H),
    return (mean, q_lo, q_hi) with shapes (N,) or (N,H).
    """
    y_mc = np.asarray(y_mc)
    mean = np.mean(y_mc, axis=0)
    q_lo = np.percentile(y_mc, lower, axis=0)
    q_hi = np.percentile(y_mc, upper, axis=0)
    return mean, q_lo, q_hi


# -----------------------
# Aggregation helpers
# -----------------------

def per_horizon(func, y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute metric per horizon step + overall mean when H>1.
    Returns dict: {"overall": ..., "h0": ..., "h1": ...}
    """
    yt, yp = _as_2d(y_true), _as_2d(y_pred)
    H = yt.shape[1]
    out = {}
    vals = []
    for h in range(H):
        v = func(yt[:, h], yp[:, h])
        out[f"h{h}"] = v
        vals.append(v)
    out["overall"] = float(np.nanmean(vals))
    return out

def per_horizon_interval(metric_func, y_true, q_lo, q_hi, **kwargs):
    yt, lo, hi = _as_2d(y_true), _as_2d(q_lo), _as_2d(q_hi)
    H = yt.shape[1]
    out = {}; vals = []
    for h in range(H):
        v = metric_func(yt[:, h], lo[:, h], hi[:, h], **kwargs)
        out[f"h{h}"] = v
        vals.append(v)
    out["overall"] = float(np.nanmean(vals))
    return out


# -----------------------
# Convenience: compute_all
# -----------------------

def compute_all(y_true: np.ndarray,
                y_pred_mean,
                q_lo,
                q_hi,
                y_mc,
                quantiles,
                q_values,
                alpha=0.1):
    """
    Compute a compact suite of metrics.
    Provide either (q_lo, q_hi) or (y_mc) or (quantiles & q_values) for probabilistic metrics.
    """
    out = {}

    # Probabilistic via intervals
    if q_lo is not None and q_hi is not None:
        out["picp"] = picp(y_true, q_lo, q_hi)
        out["mpiw"] = mpiw(q_lo, q_hi)
        out["interval_score"] = interval_score(y_true, q_lo, q_hi, alpha=alpha)

    # Probabilistic via MC
    if y_mc is not None:
        mean, lo, hi = mc_mean_q(y_mc, lower=alpha*50, upper=100 - alpha*50)
        out["picp_mc"] = picp(y_true, lo, hi)
        out["mpiw_mc"] = mpiw(lo, hi)
        out["interval_score_mc"] = interval_score(y_true, lo, hi, alpha=alpha)

    # CRPS via quantiles
    if quantiles is not None and q_values is not None:
        out["crps_q"] = crps_from_quantiles(y_true, np.asarray(quantiles), np.asarray(q_values))

    return out