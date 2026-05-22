"""
Figure 2 for the paper.

Three panels in one figure:

  (a) Full test period: observed discharge, ensemble mean (H=1) with 95% and
      50% predictive bands. CRPS and PICP95 annotated.
  (b) Zoomed event window around two prolonged-flood peaks: observed discharge
      and ensemble forecasts at H=1, with 95% and 50% bands.
  (c) Six fan charts (3 issue offsets x 2 peaks). Shared y-axis across all six.

Date alignment:
  - dates_test is the verification date of H=1 (i.e. y_test[i, 0] is observed
    discharge on dates_test[i]).
  - For horizon h, the verification date is dates_test + (h-1) days.

Central line throughout is the ENSEMBLE MEAN (not the median).

Prerequisite: run full_pipeline_analysis.py first to populate pred_cache_v4/
with the test-set predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_utils import (
    load_exog_data,
    load_target_data,
    clean_data,
    sync_data,
    merge_exog,
    scale_data,
    FEATURE_MAP,
    DEFAULT_HORIZONS,
    SUFFIX_MAP,
    file_path as DATA_FILE_PATH,
)
import analysis_utils as au


# =========================================================
# I/O
# =========================================================
def load_runs_table(runs_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(runs_csv)
    required = {"model", "nwp", "feat", "replicate", "tuner", "run_id"}
    miss = required - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns in runs CSV: {sorted(miss)}")
    return df.copy()


# =========================================================
# Data preparation (carries dates through)
# =========================================================
def prepare_data_with_dates(
    data,
    target_scaled,
    target,
    lag,
    horizon,
    val_index,
    test_index,
    use_q=False,
    seasonality=False,
):
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if len(data) < horizon:
        raise ValueError("Provided data has fewer lead DataFrames than the requested horizon")

    ws = lag + horizon

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    d_train, d_val, d_test = [], [], []

    n = len(target)

    for i in range(n - ws - horizon):
        end_ix = i + ws

        try:
            seq_x = data[0].iloc[i + horizon : end_ix + 1, :]
        except Exception:
            break

        for j in range(1, horizon):
            row_index = end_ix + j
            if row_index >= len(data[j]):
                seq_x = None
                break
            row = data[j].iloc[[row_index], :].values
            seq_x = np.vstack((seq_x, row))

        if seasonality:
            seas = np.sin(
                data[0].index[i:end_ix].dayofyear.values.reshape(-1, 1) / 365.25 * 2 * np.pi
            )
            seq_x = np.hstack((seq_x, seas))

        if seq_x is None:
            break

        if use_q:
            q_vals = target_scaled.iloc[i:end_ix].values.reshape(-1, 1)
            seq_x = np.hstack((seq_x, q_vals))

        if end_ix + horizon > n:
            break

        seq_y = target.iloc[end_ix : end_ix + horizon].squeeze()
        seq_date = target.index[end_ix]

        if end_ix >= test_index:
            X_test.append(seq_x)
            y_test.append(seq_y)
            d_test.append(seq_date)
        elif end_ix >= val_index:
            X_val.append(seq_x)
            y_val.append(seq_y)
            d_val.append(seq_date)
        else:
            X_train.append(seq_x)
            y_train.append(seq_y)
            d_train.append(seq_date)

    return (
        np.array(X_train),
        np.array(y_train),
        pd.DatetimeIndex(d_train),
        np.array(X_val),
        np.array(y_val),
        pd.DatetimeIndex(d_val),
        np.array(X_test),
        np.array(y_test),
        pd.DatetimeIndex(d_test),
    )


def data_prep_with_dates(
    nwp,
    target,
    file_path=DATA_FILE_PATH,
    vars="Qpt",
    horizons=DEFAULT_HORIZONS,
    lag=3,
    datetime_col_index=0,
    val_start="2023-01-01",
    test_start="2023-10-01",
    suffix_map=SUFFIX_MAP,
):
    """Same as data_utils.data_prep but also returns aligned test-set dates."""
    variables = FEATURE_MAP[vars]["variables"]

    exog_dfs = load_exog_data(file_path, nwp, variables, horizons, datetime_col_index)
    target_df = load_target_data(file_path, target, datetime_col_index)

    for var in variables:
        for h in horizons:
            exog_dfs[var][h] = clean_data(exog_dfs[var][h])
    target_df = clean_data(target_df)

    exog_dfs, target_df = sync_data(exog_dfs, target_df)
    exogs = merge_exog(exog_dfs, variables, horizons, suffix_map)
    exog_scaled, endo_scaled, _scaler = scale_data(exogs, target_df, val_start)

    out = prepare_data_with_dates(
        exog_scaled,
        endo_scaled,
        target_df,
        lag,
        len(horizons),
        val_index=np.where(exogs[0].index == val_start)[0][0],
        test_index=np.where(exogs[0].index == test_start)[0][0],
        use_q=FEATURE_MAP[vars]["use_q"],
        seasonality=FEATURE_MAP[vars]["seasonality"],
    )

    _, _, _, _, _, _, _, y_test, d_test = out
    return np.asarray(y_test, dtype=float), pd.DatetimeIndex(d_test)


# =========================================================
# Alignments
# =========================================================
def discharge_timeline_and_dates(
    y_true: np.ndarray, dates_h1: pd.DatetimeIndex
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Build a continuous discharge timeline y_disc[t] indexed by integer t and
    a parallel DatetimeIndex dates_disc[t].

    Convention:
      y_true[i, 0] verifies on dates_h1[i]
      y_disc[t]   is observed discharge on dates_disc[t]
      dates_disc[t] = dates_h1[0] + (t - 1) days   (i.e. dates_disc[1] == dates_h1[0])

    So y_disc[i+1] == y_true[i, 0] and y_disc[i+h] is the H=h verification target.
    """
    y_true = np.asarray(y_true, dtype=float)
    N, H = y_true.shape

    y_disc = np.full(N + H, np.nan, dtype=float)
    y_disc[1 : N + 1] = y_true[:, 0]
    for k in range(1, H):
        y_disc[N + k] = y_true[N - 1, k]

    dates_disc = [dates_h1[0] - pd.Timedelta(days=1)]
    dates_disc.extend(list(dates_h1))
    for k in range(1, H):
        dates_disc.append(dates_h1[-1] + pd.Timedelta(days=k))

    return y_disc, pd.DatetimeIndex(dates_disc)


def horizon_dates(dates_h1: pd.DatetimeIndex, h: int) -> pd.DatetimeIndex:
    """Verification dates for horizon h: dates_h1 shifted by (h-1) days."""
    return pd.DatetimeIndex(dates_h1 + pd.to_timedelta(h - 1, unit="D"))


# =========================================================
# Peak picking & event window
# =========================================================
def local_peaks(y: np.ndarray, thr: float) -> list[int]:
    out = []
    for i in range(1, len(y) - 1):
        if np.isfinite(y[i]) and y[i] > thr and y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            out.append(i)
    return out


def pick_two_distinct_peaks(
    y_disc: np.ndarray, t_start: int, peak_thr: float, dip_thr: float
) -> tuple[int, int]:
    cand = [t for t in local_peaks(y_disc, peak_thr) if t >= t_start]
    if len(cand) < 2:
        ids = np.where(np.isfinite(y_disc))[0]
        ids = ids[ids >= t_start]
        top2 = ids[np.argsort(y_disc[ids])[-2:]]
        top2 = np.sort(top2)
        return int(top2[0]), int(top2[1])

    cand_sorted = sorted(cand, key=lambda t: y_disc[t], reverse=True)
    p1 = cand_sorted[0]

    best = None
    for t in cand_sorted[1:]:
        a, b = (p1, t) if p1 < t else (t, p1)
        seg = y_disc[a : b + 1]
        if np.any(np.isfinite(seg)) and np.nanmin(seg) < dip_thr:
            best = t
            break

    if best is None:
        best = cand_sorted[1]

    p2 = best
    if p2 < p1:
        p1, p2 = p2, p1
    return int(p1), int(p2)


def build_event_window_include_p2(
    y_disc: np.ndarray, p1: int, p2: int, max_len: int = 40
) -> tuple[int, int]:
    ws = max(0, p1 - 12)
    we = p2 + 6
    if we - ws > max_len:
        ws = max(0, we - max_len)
    we = min(we, len(y_disc) - 1)
    return int(ws), int(we)


# =========================================================
# Volumetric error over the event window
# =========================================================
def volumetric_error_event(
    preds: np.ndarray, y_disc: np.ndarray, ws: int, we: int, Hmax: int = 5
) -> dict[int, float]:
    preds = np.asarray(preds, dtype=float)
    y_disc = np.asarray(y_disc, dtype=float)

    M, N, H = preds.shape
    Huse = min(int(Hmax), H)
    out = {}
    t = np.arange(ws, we + 1)

    for h in range(1, Huse + 1):
        i = t - h
        m = (
            (i >= 0)
            & (i < N)
            & (t >= 0)
            & (t < len(y_disc))
            & np.isfinite(y_disc[t])
        )
        if not np.any(m):
            out[h] = np.nan
            continue

        obs = y_disc[t[m]]
        pred_mean = np.nanmean(preds[:, i[m], h - 1], axis=0)

        denom = np.nansum(obs)
        if denom == 0 or not np.isfinite(denom):
            out[h] = np.nan
        else:
            out[h] = float(100.0 * np.nansum(pred_mean - obs) / denom)

    return out


# =========================================================
# Fan chart payload (uses ensemble MEAN as central line)
# =========================================================
def fan_payload(
    preds: np.ndarray,
    y_true: np.ndarray,
    y_disc: np.ndarray,
    dates_disc: pd.DatetimeIndex,
    t0: int,
    past_days: int = 3,
    leads: int = 5,
):
    """
    Fan chart payload for an issue at discharge-timeline index t0.

    Alignment:
      preds[:, t0, h-1] verifies y_disc[t0 + h]   (and y_true[t0, h-1] is the same)
    """
    preds = np.asarray(preds, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_disc = np.asarray(y_disc, dtype=float)

    M, N, H = preds.shape
    leads = int(min(leads, H))
    i0 = int(t0)
    if not (0 <= i0 < N):
        return None

    # Past observed window (and corresponding x positions in days relative to issue)
    t_hist = np.arange(t0 - past_days, t0 + 1)
    x_hist = t_hist - t0  # -past_days .. 0

    obs_hist = np.full_like(t_hist, np.nan, dtype=float)
    for k, t in enumerate(t_hist):
        if 0 <= t < len(y_disc):
            obs_hist[k] = y_disc[t]

    samples = preds[:, i0, :leads]  # (M, leads)
    q025, q25, q75, q975 = np.nanquantile(samples, [0.025, 0.25, 0.75, 0.975], axis=0)
    mean_line = np.nanmean(samples, axis=0)  # central line = ensemble MEAN

    obs_now = y_disc[t0] if (0 <= t0 < len(y_disc)) else np.nan

    # Make bands start at x=0 with the observed "now" value (degenerate band)
    x_future = np.arange(0, leads + 1)
    q025 = np.r_[obs_now, q025]
    q25 = np.r_[obs_now, q25]
    q75 = np.r_[obs_now, q75]
    q975 = np.r_[obs_now, q975]
    mean_line = np.r_[obs_now, mean_line]

    # Observed future truth at +1..+leads (no duplicate at 0)
    obs_future_x = np.arange(1, leads + 1)
    obs_future = np.asarray(y_true[i0, :leads], dtype=float)

    issue_date = dates_disc[t0] if (0 <= t0 < len(dates_disc)) else None

    return {
        "x_hist": x_hist,
        "obs_hist": obs_hist,
        "x_future": x_future,
        "q025": q025,
        "q25": q25,
        "mean": mean_line,
        "q75": q75,
        "q975": q975,
        "obs_future_x": obs_future_x,
        "obs_future": obs_future,
        "issue_date": issue_date,
    }


def shared_ylim(payloads: list[dict], pad_frac: float = 0.06) -> Optional[tuple[float, float]]:
    vals = []
    for p in payloads:
        if p is None:
            continue
        for key in ["obs_hist", "q025", "q975", "obs_future"]:
            a = np.asarray(p[key])
            if np.any(np.isfinite(a)):
                vals.append(np.nanmin(a))
                vals.append(np.nanmax(a))
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return None
    lo, hi = float(min(vals)), float(max(vals))
    pad = pad_frac * (hi - lo + 1e-9)
    return lo - pad, hi + pad


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_csv", type=str, default="analysis_out_v4/tables/runs_kept_v4.csv")
    ap.add_argument("--pred_cache", type=str, default="pred_cache_v4",
                    help="Directory with cached predictions (populated by full_pipeline_analysis.py).")
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument("--file_path", type=str, default=DATA_FILE_PATH)
    ap.add_argument("--val_start", type=str, default="2023-01-01")
    ap.add_argument("--test_start", type=str, default="2023-10-01")
    ap.add_argument("--lag", type=int, default=3)

    ap.add_argument("--event_tstart", type=int, default=50)
    ap.add_argument("--peak_threshold", type=float, default=200.0)
    ap.add_argument("--dip_threshold", type=float, default=110.0)
    ap.add_argument("--max_event_len", type=int, default=40)

    ap.add_argument("--fan_past_days", type=int, default=3)
    ap.add_argument("--fan_leads", type=int, default=5)

    args = ap.parse_args()

    runs_csv = Path(args.runs_csv)
    pred_cache = Path(args.pred_cache)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # -----------------------------
    # Load runs and stack predictions
    # -----------------------------
    df_runs = load_runs_table(runs_csv)

    # y_test + dates cache per (nwp, feat) — at most 16 unique combinations
    yd_cache: dict[tuple[str, str], tuple[np.ndarray, pd.DatetimeIndex]] = {}

    preds_list = []
    y_ref = None
    dates_ref = None
    n_missing_pred = 0
    n_shape_fail = 0

    print(f"[INFO] Reading cached predictions from {pred_cache} for {len(df_runs)} runs.")

    for _, row in df_runs.iterrows():
        key = (str(row["nwp"]), str(row["feat"]))
        if key not in yd_cache:
            y_k, d_k = data_prep_with_dates(
                nwp=key[0],
                target="Q",
                file_path=args.file_path,
                vars=key[1],
                lag=args.lag,
                val_start=args.val_start,
                test_start=args.test_start,
            )
            yd_cache[key] = (y_k, d_k)

        y_test_k, d_test_k = yd_cache[key]

        yp = au.load_prediction(str(pred_cache), str(row["run_id"]))
        if yp is None:
            n_missing_pred += 1
            continue
        yp = np.asarray(yp, dtype=float)

        if yp.shape != y_test_k.shape:
            n_shape_fail += 1
            continue

        if y_ref is None:
            y_ref = y_test_k.copy()
            dates_ref = d_test_k.copy()
        else:
            if y_test_k.shape != y_ref.shape or not np.allclose(y_test_k, y_ref, equal_nan=True):
                print(f"[WARN] y_test differs for key={key}. Using first y_test.")
            if len(d_test_k) != len(dates_ref) or not np.all(d_test_k == dates_ref):
                print(f"[WARN] dates_test differs for key={key}.")

        preds_list.append(yp)

    if n_missing_pred:
        print(f"[WARN] missing cached prediction for {n_missing_pred} runs "
              f"(run full_pipeline_analysis.py to populate {pred_cache}).")
    if n_shape_fail:
        print(f"[WARN] prediction-shape mismatch for {n_shape_fail} runs.")

    if not preds_list:
        raise RuntimeError(
            f"No cached predictions found in {pred_cache}. "
            f"Run full_pipeline_analysis.py first."
        )

    preds = np.stack(preds_list, axis=0)  # (M, N, H)
    y_test = np.asarray(y_ref, dtype=float)
    dates_test = pd.DatetimeIndex(dates_ref)

    M, N, H = preds.shape
    print(f"[INFO] Loaded ensemble: M={M}, N={N}, H={H}")
    print(f"[INFO] dates_test[0] = {dates_test[0]}, dates_test[-1] = {dates_test[-1]}")

    # Continuous discharge timeline (for peak picking and fan-chart history)
    y_disc, dates_disc = discharge_timeline_and_dates(y_test, dates_test)

    # -----------------------------
    # Panel (a) — full test, H=1
    # -----------------------------
    ens_h1 = preds[:, :, 0]
    mean_h1 = np.nanmean(ens_h1, axis=0)
    lo50, hi50 = np.nanquantile(ens_h1, [0.25, 0.75], axis=0)
    lo95, hi95 = np.nanquantile(ens_h1, [0.025, 0.975], axis=0)

    crps1 = au.crps_from_samples(y_test[:, 0], ens_h1)
    picp95 = au.picp(y_test[:, 0], lo95, hi95)

    # -----------------------------
    # Event window (peaks on y_disc; dates_disc[p] is the date of peak)
    # -----------------------------
    p1, p2 = pick_two_distinct_peaks(
        y_disc,
        t_start=args.event_tstart,
        peak_thr=args.peak_threshold,
        dip_thr=args.dip_threshold,
    )
    ws_t, we_t = build_event_window_include_p2(y_disc, p1, p2, max_len=args.max_event_len)
    ve_event = volumetric_error_event(preds, y_disc, ws_t, we_t, Hmax=min(5, H))

    d_win_lo = dates_disc[ws_t]
    d_win_hi = dates_disc[we_t]
    d_p1 = dates_disc[p1]
    d_p2 = dates_disc[p2]

    # -----------------------------
    # Fan-chart payloads (issue offsets in days before each peak)
    # -----------------------------
    issue_offsets = [3, 2, 1]
    payloads_p1, payloads_p2 = [], []
    for d in issue_offsets:
        pl1 = fan_payload(
            preds, y_test, y_disc, dates_disc,
            t0=p1 - d, past_days=args.fan_past_days, leads=args.fan_leads,
        )
        pl2 = fan_payload(
            preds, y_test, y_disc, dates_disc,
            t0=p2 - d, past_days=args.fan_past_days, leads=args.fan_leads,
        )
        if pl1 is None or pl2 is None:
            raise RuntimeError(
                "Fan chart issue time outside available range. "
                "Adjust thresholds or offsets."
            )
        payloads_p1.append(pl1)
        payloads_p2.append(pl2)

    # SHARED y-limits across all six fan panels (consistency)
    fan_ylim = shared_ylim(payloads_p1 + payloads_p2)

    # -----------------------------
    # Figure layout
    # -----------------------------
    fig = plt.figure(figsize=(17, 8.4))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[3.25, 1.45],
        height_ratios=[1, 1],
        left=0.06, right=0.99, top=0.94, bottom=0.10,
        wspace=0.22, hspace=0.32,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c_container = fig.add_subplot(gs[:, 1])
    ax_c_container.axis("off")

    gs_c = gs[:, 1].subgridspec(nrows=3, ncols=2, wspace=0.18, hspace=0.34)
    ax_c = [[fig.add_subplot(gs_c[r, c]) for c in range(2)] for r in range(3)]

    # Colour scheme — outer band blue, inner band orange, central line darker blue
    obs_color = "black"
    central_color = "#1f77b4"
    outer_color = "#1f77b4"
    inner_color = "#ff7f0e"

    # =========================================================
    # (a) Full test period, h=1
    # =========================================================
    ax_a.plot(dates_test, y_test[:, 0], color=obs_color, lw=1.25, label="Observed")
    ax_a.plot(dates_test, mean_h1, color=central_color, lw=1.6, label="Ensemble mean")
    ax_a.fill_between(dates_test, lo95, hi95, color=outer_color, alpha=0.20, label="95% band")
    ax_a.fill_between(dates_test, lo50, hi50, color=inner_color, alpha=0.30, label="50% band")
    ax_a.set_title("Full test period — ensemble mean and predictive intervals (h=1)")
    ax_a.set_xlabel("Date")
    ax_a.set_ylabel(r"Discharge (m$^3$/s)")
    ax_a.legend(loc="upper right", frameon=False)
    ax_a.text(
        0.5, 0.95,
        f"CRPS (h=1): {crps1:.2f} m$^3$/s \nPICP$_{{95}}$ (h=1): {picp95*100:.1f}%",
        transform=ax_a.transAxes, va="top", ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )
    ax_a.text(-0.06, 1.04, "a", transform=ax_a.transAxes,
              fontweight="bold", fontsize=18, va="bottom")
    ax_a.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # =========================================================
    # (b) Zoomed event window with h=1 only
    # =========================================================
    t_win = np.arange(ws_t, we_t + 1)
    d_win = dates_disc[t_win]
    y_obs_win = y_disc[t_win]
    ax_b.plot(d_win, y_obs_win, color=obs_color, lw=1.25)

    # H=1 forecast at verification dates
    d_h1 = horizon_dates(dates_test, 1)
    keep = (d_h1 >= d_win_lo) & (d_h1 <= d_win_hi)
    lo95_b = np.nanquantile(preds[:, :, 0], 0.025, axis=0)
    hi95_b = np.nanquantile(preds[:, :, 0], 0.975, axis=0)
    lo50_b = np.nanquantile(preds[:, :, 0], 0.25, axis=0)
    hi50_b = np.nanquantile(preds[:, :, 0], 0.75, axis=0)
    mean_h1_b = np.nanmean(preds[:, :, 0], axis=0)

    ax_b.plot(d_h1[keep], mean_h1_b[keep], color=central_color, lw=1.8)
    ax_b.fill_between(d_h1[keep], lo95_b[keep], hi95_b[keep],
                      color=outer_color, alpha=0.20, linewidth=0)
    ax_b.fill_between(d_h1[keep], lo50_b[keep], hi50_b[keep],
                      color=inner_color, alpha=0.30, linewidth=0)

    # Mark peaks
    for k, dp in enumerate([d_p1, d_p2], start=1):
        if d_win_lo <= dp <= d_win_hi:
            ax_b.axvline(dp, color="0.55", lw=0.9, ls="--")
            t_peak = p1 if k == 1 else p2
            yy = y_disc[t_peak]
            if np.isfinite(yy):
                ax_b.text(dp + pd.Timedelta(hours=8), yy,
                          f"Peak {k}", fontsize=11, color="0.30", va="bottom")

    ax_b.set_xlim(d_win_lo, d_win_hi)
    ax_b.set_title("Zoomed: Prolonged high-flow event (h=1)")
    ax_b.set_xlabel("Date")
    ax_b.set_ylabel(r"Discharge (m$^3$/s)")
    ax_b.text(-0.06, 1.04, "b", transform=ax_b.transAxes,
              fontweight="bold", fontsize=18, va="bottom")
    ax_b.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # =========================================================
    # (c) Fan charts, shared y-limits across all six panels
    # =========================================================
    ax_c[0][0].set_title("Peak 1", fontsize=12)
    ax_c[0][1].set_title("Peak 2", fontsize=12)

    for r, d in enumerate(issue_offsets):
        for col, pl in enumerate([payloads_p1[r], payloads_p2[r]]):
            ax = ax_c[r][col]
            for s in ["top", "right", "left", "bottom"]:
                ax.spines[s].set_visible(True)
                ax.spines[s].set_linewidth(0.6)
                ax.spines[s].set_color("0.3")

            ax.axvline(0, color="0.65", lw=0.9)

            ax.fill_between(pl["x_future"], pl["q025"], pl["q975"],
                            color=outer_color, alpha=0.20, linewidth=0)
            ax.fill_between(pl["x_future"], pl["q25"], pl["q75"],
                            color=inner_color, alpha=0.30, linewidth=0)
            ax.plot(pl["x_future"], pl["mean"], color=central_color, lw=1.5)

            x_all = np.concatenate([pl["x_hist"], pl["obs_future_x"]])
            y_all = np.concatenate([pl["obs_hist"], pl["obs_future"]])
            ax.plot(x_all, y_all, color=obs_color, lw=0.95, marker="o", ms=2.4, alpha=0.9)

            ax.set_xlim(-args.fan_past_days, args.fan_leads)
            if fan_ylim is not None:
                ax.set_ylim(*fan_ylim)

            ax.text(0.05, 0.92, f"t = peak − {d} day(s)",
                    transform=ax.transAxes, ha="left", va="top", fontsize=11)

            if r < 2:
                ax.set_xticklabels([])
            if col == 1:
                ax.set_yticklabels([])

    ax_c[2][0].set_xlabel("Days relative to prediction")
    ax_c[2][1].set_xlabel("Days relative to prediction")
    ax_c[1][0].set_ylabel(r"Discharge (m$^3$/s)")

    ax_c_container.text(-0.06, 1.04, "c", transform=ax_c_container.transAxes,
                        fontweight="bold", fontsize=18, va="bottom")

    # =========================================================
    # Save
    # =========================================================
    out_png = outdir / "Figure2_paper.png"
    out_pdf = outdir / "Figure2_paper.pdf"
    fig.savefig(out_png, dpi=1200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved: {out_png}")
    print(f"[DONE] Saved: {out_pdf}")
    print(f"[INFO] Peaks: p1={p1} ({d_p1.date()}), p2={p2} ({d_p2.date()})")
    print(f"[INFO] Window: {d_win_lo.date()} → {d_win_hi.date()}")
    print("\n[EVENT VE% over prolonged high-flow window]")
    for h in range(1, min(5, H) + 1):
        print(f"  h={h}: {ve_event[h]:+.2f}%")


if __name__ == "__main__":
    main()
