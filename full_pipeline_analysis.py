"""
Full-pipeline analysis script.

Produces, under analysis_out_v4/:
  - tables/runs_scanned.csv
  - tables/metrics_per_model_per_horizon.csv
  - tables/runs_kept_v4.csv
  - tables/metrics_kept_v4_per_model_per_horizon.csv
  - tables/v4_selection_summary.csv
  - tables/v4_selection_flags.csv
  - tables/ensemble_deterministic_metrics_mean_v4.csv
  - tables/ensemble_deterministic_metrics_mean_with_IQR_v4.csv
  - tables/ensemble_prob_metrics_v4.csv
  - tables/overall_metrics_kept_v4_by_run.csv
  - tables/group_compact_overall_kept_v4__by_{factor}.csv
  - tables/group_compact_by_horizon_kept_v4__by_{factor}.csv
  - plots/violin__{metric}__kept_v4__h1to5.png

Cached predictions are written to pred_cache_v4/ and reused by the figure
scripts (figure2.py, figure3.py, figure5.py).
"""

from __future__ import annotations

import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K

import analysis_utils as au


# -----------------------------
# Pretty labels (full words)
# -----------------------------
FACTOR_LABEL = {
    "model": "model architecture",
    "nwp": "data source",
    "feat": "features",
    "tuner": "tuning algorithm",
    "replicate": "replicate",
}


# -----------------------------
# Small plotting utilities
# -----------------------------
def save_violin_by_horizon(df: pd.DataFrame, metric: str, out_png: Path, title: str, horizons=(1, 2, 3, 4, 5)):
    d = df[["horizon", metric]].dropna()
    data = [d[d["horizon"] == h][metric].values for h in horizons]
    if all(len(x) == 0 for x in data):
        return
    plt.figure(figsize=(9, 4.5))
    plt.violinplot(data, showmeans=True, showmedians=False, showextrema=True)
    plt.xticks(range(1, len(horizons) + 1), [str(h) for h in horizons])
    plt.xlabel("forecast horizon (days ahead)")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def group_compact_table(df: pd.DataFrame, group_col: str, metrics: list[str], horizon: int | None = None) -> pd.DataFrame:
    d = df.copy()
    if horizon is not None:
        d = d[d["horizon"] == horizon].copy()
    rows = []
    for lvl, g in d.groupby(group_col):
        row = {"level": str(lvl), "n": int(g.shape[0])}
        for m in metrics:
            x = g[m].dropna().values if m in g.columns else np.array([])
            if x.size == 0:
                row[f"{m}_median"] = np.nan
                row[f"{m}_q25"] = np.nan
                row[f"{m}_q75"] = np.nan
                row[f"{m}_IQR"] = ""
            else:
                q25, q50, q75 = np.quantile(x, [0.25, 0.50, 0.75])
                row[f"{m}_median"] = float(q50)
                row[f"{m}_q25"] = float(q25)
                row[f"{m}_q75"] = float(q75)
                row[f"{m}_IQR"] = f"{q25:.3f}–{q75:.3f}"
        rows.append(row)
    out = pd.DataFrame(rows)
    if "NSE_median" in out.columns:
        out = out.sort_values("NSE_median", ascending=False, na_position="last")
    return out


def add_iqr_cols_per_horizon(df_det: pd.DataFrame, df_metrics: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    df_det: ensemble mean metrics per horizon.
    df_metrics: per-model per-horizon metrics (kept subset).
    Adds q25/q50/q75 and a formatted IQR string per metric, per horizon.
    """
    out = df_det.copy()
    for h in out["horizon"].tolist():
        dh = df_metrics[df_metrics["horizon"] == h]
        for m in cols:
            if m not in dh.columns:
                continue
            x = dh[m].dropna().values
            if x.size == 0:
                out.loc[out["horizon"] == h, f"{m}_q25"] = np.nan
                out.loc[out["horizon"] == h, f"{m}_q50"] = np.nan
                out.loc[out["horizon"] == h, f"{m}_q75"] = np.nan
                out.loc[out["horizon"] == h, f"{m}_IQR"] = ""
            else:
                q25, q50, q75 = np.quantile(x, [0.25, 0.50, 0.75])
                out.loc[out["horizon"] == h, f"{m}_q25"] = float(q25)
                out.loc[out["horizon"] == h, f"{m}_q50"] = float(q50)
                out.loc[out["horizon"] == h, f"{m}_q75"] = float(q75)
                out.loc[out["horizon"] == h, f"{m}_IQR"] = f"{q25:.3f}–{q75:.3f}"
    return out


# -----------------------------
# v4 selection: union-of-worst quartiles computed on the full ensemble
# -----------------------------
def v4_keep_ids(df_overall: pd.DataFrame, q=0.25):
    """
    df_overall columns: run_id, NSE_mean, KGE_mean, RMSE_mean.
    Worst sets are computed on the FULL ensemble. Each set is exactly floor(N*q)
    with deterministic tie-breaking by run_id.
    """
    d = df_overall.copy()
    N = len(d)
    k = int(np.floor(N * q))
    d = d.sort_values("run_id").reset_index(drop=True)

    worst_nse = set(d.sort_values(["NSE_mean", "run_id"], ascending=[True, True]).head(k)["run_id"])
    worst_kge = set(d.sort_values(["KGE_mean", "run_id"], ascending=[True, True]).head(k)["run_id"])
    worst_rmse = set(d.sort_values(["RMSE_mean", "run_id"], ascending=[False, True]).head(k)["run_id"])

    removed_1 = worst_nse
    removed_2 = worst_kge - removed_1
    removed_3 = worst_rmse - (removed_1 | worst_kge)
    removed_union = worst_nse | worst_kge | worst_rmse
    keep_ids = set(d["run_id"]) - removed_union

    summary = pd.DataFrame([
        {"step": 1, "metric": "NSE",  "removed_this_step": len(removed_1),
         "removed_cumulative": len(removed_1), "pct_removed_cum": 100 * len(removed_1) / N},
        {"step": 2, "metric": "KGE",  "removed_this_step": len(removed_2),
         "removed_cumulative": len(removed_1 | worst_kge), "pct_removed_cum": 100 * len(removed_1 | worst_kge) / N},
        {"step": 3, "metric": "RMSE", "removed_this_step": len(removed_3),
         "removed_cumulative": len(removed_union), "pct_removed_cum": 100 * len(removed_union) / N},
        {"step": 4, "metric": "FINAL", "removed_this_step": 0,
         "removed_cumulative": len(removed_union), "pct_removed_cum": 100 * len(removed_union) / N,
         "kept": len(keep_ids), "pct_kept": 100 * len(keep_ids) / N},
    ])

    flagged = d[["run_id", "NSE_mean", "KGE_mean", "RMSE_mean"]].copy()
    flagged["worst25_NSE"] = flagged["run_id"].isin(worst_nse)
    flagged["worst25_KGE"] = flagged["run_id"].isin(worst_kge)
    flagged["worst25_RMSE"] = flagged["run_id"].isin(worst_rmse)
    flagged["removed_union"] = ~flagged["run_id"].isin(keep_ids)

    return keep_ids, summary, flagged


def main():
    # -----------------------------
    # Config
    # -----------------------------
    ROOT_MODELS = "final-models"
    OUTDIR = Path("analysis_out_v4")
    PLOTS = OUTDIR / "plots"
    TABLES = OUTDIR / "tables"
    PRED_CACHE = Path("pred_cache_v4")
    OUTDIR.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    PRED_CACHE.mkdir(exist_ok=True)

    TARGET = "Q"
    VAL_START = "2023-01-01"
    TEST_START = "2023-10-01"
    HORIZON = 5
    PEAK_Q = 0.90
    ALPHA = 0.05  # 95% interval band
    Q_WORST = 0.25

    # deterministic metrics where we want IQR across models
    DET_IQR_METRICS = ["NSE", "KGE", "RMSE", "MAE", "MAPE", "r"]

    # -----------------------------
    # Verify required analysis_utils hooks exist
    # -----------------------------
    required = [
        "scan_runs", "build_data_cache", "build_custom_objects",
        "load_model_for_inference", "load_prediction", "save_prediction",
        "compute_metrics_per_horizon", "ensemble_summary",
    ]
    missing = [f for f in required if not hasattr(au, f)]
    if missing:
        raise RuntimeError(f"analysis_utils.py is missing required functions: {missing}")

    # -----------------------------
    # 0) Load or compute df_runs / df_metrics
    # -----------------------------
    df_runs_path = TABLES / "runs_scanned.csv"
    df_metrics_path = TABLES / "metrics_per_model_per_horizon.csv"

    if df_runs_path.exists():
        df_runs = pd.read_csv(df_runs_path)
    else:
        df_runs = au.scan_runs(ROOT_MODELS)
        df_runs.to_csv(df_runs_path, index=False)

    # data cache + y_test reference check
    data_cache = au.build_data_cache(df_runs, target=TARGET, val_start=VAL_START, test_start=TEST_START)
    first_key = next(iter(data_cache.keys()))
    _, y_test_ref = data_cache[first_key]["test"]
    for k, v in data_cache.items():
        yk = v["test"][1]
        if yk.shape != y_test_ref.shape or not np.allclose(yk, y_test_ref, equal_nan=True):
            raise RuntimeError(f"y_test mismatch across data_prep calls at key={k}.")

    if df_metrics_path.exists():
        df_metrics = pd.read_csv(df_metrics_path)
    else:
        custom_objects = au.build_custom_objects()
        rows = []

        for i, row in df_runs.iterrows():
            run_id = row["run_id"]
            X_test, y_test = data_cache[(row["nwp"], row["feat"])]["test"]

            y_pred = au.load_prediction(str(PRED_CACHE), run_id)
            if y_pred is None:
                model = au.load_model_for_inference(row["keras_path"], custom_objects=custom_objects)
                y_pred = model.predict(X_test, batch_size=1024, verbose=0)
                au.save_prediction(str(PRED_CACHE), run_id, y_pred)
                del model
                K.clear_session()
                gc.collect()

            per_h = au.compute_metrics_per_horizon(y_test, y_pred, peak_q=PEAK_Q)

            for h in range(HORIZON):
                rec = {
                    "run_id": run_id,
                    "model": row["model"],
                    "nwp": row["nwp"],
                    "feat": row["feat"],
                    "tuner": row["tuner"],
                    "replicate": int(row["replicate"]),
                    "horizon": h + 1,
                }
                rec.update(per_h[h])
                rows.append(rec)

            if (i + 1) % 50 == 0:
                print(f"[INFO] processed {i + 1}/{len(df_runs)}")

        df_metrics = pd.DataFrame(rows)
        df_metrics.to_csv(df_metrics_path, index=False)

    # -----------------------------
    # 1) v4 selection on FULL ensemble (1024)
    # -----------------------------
    df_overall = (df_metrics.groupby("run_id", as_index=False)
                  .agg(NSE_mean=("NSE", "mean"), KGE_mean=("KGE", "mean"), RMSE_mean=("RMSE", "mean")))
    keep_ids, sel_summary, sel_flagged = v4_keep_ids(df_overall, q=Q_WORST)

    sel_summary.to_csv(TABLES / "v4_selection_summary.csv", index=False)
    sel_flagged.to_csv(TABLES / "v4_selection_flags.csv", index=False)

    df_runs_kept = df_runs[df_runs["run_id"].isin(keep_ids)].copy()
    df_metrics_kept = df_metrics[df_metrics["run_id"].isin(keep_ids)].copy()

    print(f"[INFO] v4 kept: {len(keep_ids)}/{len(df_overall)} = {100 * len(keep_ids) / len(df_overall):.1f}%")

    df_runs_kept.to_csv(TABLES / "runs_kept_v4.csv", index=False)
    df_metrics_kept.to_csv(TABLES / "metrics_kept_v4_per_model_per_horizon.csv", index=False)

    # -----------------------------
    # 2) Ensemble metrics (deterministic mean + probabilistic interval)
    # -----------------------------
    keep_list = list(keep_ids)
    preds = []
    for rid in keep_list:
        yp = au.load_prediction(str(PRED_CACHE), rid)
        if yp is None:
            raise RuntimeError(f"Missing cached prediction for run_id={rid}.")
        preds.append(yp)
    preds = np.stack(preds, axis=0)  # (M, N, H)

    # deterministic ensemble mean evaluated as a single forecast
    det_rows = []
    for h in range(HORIZON):
        y_obs = y_test_ref[:, [h]]
        mean_pred = np.nanmean(preds[:, :, h], axis=0).reshape(-1, 1)
        per = au.compute_metrics_per_horizon(y_obs, mean_pred, peak_q=PEAK_Q)[0]
        det_rows.append({"horizon": h + 1, **per, "n_models": preds.shape[0]})

    df_det_ens = pd.DataFrame(det_rows)
    df_det_ens.to_csv(TABLES / "ensemble_deterministic_metrics_mean_v4.csv", index=False)

    # IQR across models (kept subset) for deterministic metrics
    df_det_ens_iqr = add_iqr_cols_per_horizon(
        df_det_ens, df_metrics_kept,
        cols=[m for m in DET_IQR_METRICS if m in df_metrics_kept.columns],
    )
    df_det_ens_iqr.to_csv(TABLES / "ensemble_deterministic_metrics_mean_with_IQR_v4.csv", index=False)

    # probabilistic interval metrics via ensemble_summary
    ens = au.ensemble_summary(y_test_ref, preds, alpha=ALPHA)
    prob_rows = [{"horizon": h + 1, **ens[h], "n_models": preds.shape[0]} for h in range(HORIZON)]
    df_prob = pd.DataFrame(prob_rows)

    # robust column naming for interval-score
    if "IntervalSc" not in df_prob.columns:
        if "IntervalScore" in df_prob.columns:
            df_prob = df_prob.rename(columns={"IntervalScore": "IntervalSc"})
        elif "interval_score" in df_prob.columns:
            df_prob = df_prob.rename(columns={"interval_score": "IntervalSc"})

    prob_cols = [c for c in ["horizon", "CRPS", "IntervalSc", "PICP", "MPIW", "n_models"] if c in df_prob.columns]
    df_prob = df_prob[prob_cols]
    df_prob.to_csv(TABLES / "ensemble_prob_metrics_v4.csv", index=False)

    # -----------------------------
    # 3) Violin plots vs horizon (kept subset)
    # -----------------------------
    violin_metrics = [m for m in ["NSE", "KGE", "RMSE", "MAE", "MAPE", "r", "SCAS"] if m in df_metrics_kept.columns]
    for m in violin_metrics:
        save_violin_by_horizon(
            df_metrics_kept, metric=m,
            out_png=PLOTS / f"violin__{m}__kept_v4__h1to5.png",
            title=f"{m} | kept subset (v4 union-of-worst-25% removal) | distribution across horizons (1–5 days ahead)",
        )

    # -----------------------------
    # 4) Grouped compact tables (publishable)
    # -----------------------------
    headline = [m for m in ["NSE", "KGE", "RMSE", "MAE", "MAPE", "r"] if m in df_metrics_kept.columns]

    overall_run = (df_metrics_kept
        .groupby(["run_id", "model", "nwp", "feat", "tuner", "replicate"], as_index=False)
        .agg({m: "mean" for m in headline})
    )
    overall_run.to_csv(TABLES / "overall_metrics_kept_v4_by_run.csv", index=False)

    for f in ["tuner", "model", "nwp", "feat", "replicate"]:
        # overall table
        tab = group_compact_table(overall_run, group_col=f, metrics=headline, horizon=None)
        tab.insert(0, "factor", FACTOR_LABEL.get(f, f))
        tab.to_csv(TABLES / f"group_compact_overall_kept_v4__by_{f}.csv", index=False)

        # per-horizon table
        tabs = []
        for h in range(1, HORIZON + 1):
            th = group_compact_table(df_metrics_kept, group_col=f, metrics=headline, horizon=h)
            th.insert(0, "horizon", h)
            th.insert(1, "factor", FACTOR_LABEL.get(f, f))
            tabs.append(th)
        pd.concat(tabs, ignore_index=True).to_csv(TABLES / f"group_compact_by_horizon_kept_v4__by_{f}.csv", index=False)

    print("\n[DONE] v4 outputs written to:")
    print("  ", str(OUTDIR))
    print("   Tables:", str(TABLES))
    print("   Plots :", str(PLOTS))
    print("   Pred cache:", str(PRED_CACHE))


if __name__ == "__main__":
    main()
