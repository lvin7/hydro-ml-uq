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
def save_violin_by_horizon(df: pd.DataFrame, metric: str, out_png: Path, title: str, horizons=(1,2,3,4,5)):
    d = df[["horizon", metric]].dropna()
    data = [d[d["horizon"] == h][metric].values for h in horizons]
    if all(len(x) == 0 for x in data):
        return
    plt.figure(figsize=(9, 4.5))
    plt.violinplot(data, showmeans=True, showmedians=False, showextrema=True)
    plt.xticks(range(1, len(horizons)+1), [str(h) for h in horizons])
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
    # prefer sort by NSE if present
    if "NSE_median" in out.columns:
        out = out.sort_values("NSE_median", ascending=False, na_position="last")
    return out


def add_iqr_cols_per_horizon(df_det: pd.DataFrame, df_metrics: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    df_det: ensemble mean metrics per horizon
    df_metrics: per-model per-horizon metrics (kept subset)
    Adds q25/q50/q75 and formatted IQR string per metric, per horizon.
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


def save_stack_main_inter_unexp(df_main_sum: pd.DataFrame, df_ols_totals: pd.DataFrame, out_png: Path, title: str):
    """
    df_main_sum columns: horizon, explained_main (sum eta2 main effects)
    df_ols_totals columns: horizon, R2_full (explained main+interactions)
    Plot stack: main (absolute), interactions(total), unexplained.
    """
    d = df_main_sum.merge(df_ols_totals[["horizon", "R2_full"]], on="horizon", how="left").copy()
    d["explained_main"] = d["explained_main"].fillna(0.0)
    d["R2_full"] = d["R2_full"].fillna(0.0)

    d["explained_interactions"] = np.clip(d["R2_full"] - d["explained_main"], 0.0, 1.0)
    d["unexplained"] = np.clip(1.0 - d["R2_full"], 0.0, 1.0)

    x = d["horizon"].values
    y1 = d["explained_main"].values
    y2 = d["explained_interactions"].values
    y3 = d["unexplained"].values

    plt.figure(figsize=(9, 5))
    plt.stackplot(x, y1, y2, y3, labels=["Main effects", "Interactions (total)", "Unexplained"])
    plt.xticks(x, [str(int(v)) for v in x])
    plt.ylim(0, 1)
    plt.xlabel("forecast horizon (days ahead)")
    plt.ylabel("variance fraction")
    plt.title(title)
    plt.legend(loc="upper left", fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()

    # explained-only normalized
    d["explained_total"] = np.clip(d["R2_full"], 1e-12, 1.0)
    d["main_norm"] = d["explained_main"] / d["explained_total"]
    d["inter_norm"] = d["explained_interactions"] / d["explained_total"]

    plt.figure(figsize=(9, 5))
    plt.stackplot(x, d["main_norm"].values, d["inter_norm"].values, labels=["Main effects", "Interactions (total)"])
    plt.xticks(x, [str(int(v)) for v in x])
    plt.ylim(0, 1)
    plt.xlabel("forecast horizon (days ahead)")
    plt.ylabel("normalized explained variance")
    plt.title(title + " (normalized explained only)")
    plt.legend(loc="upper left", fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(out_png.with_name(out_png.stem + "__normalized_explained.png"), dpi=170)
    plt.close()


# -----------------------------
# v4 selection: union-of-worst quartiles computed on full ensemble
# -----------------------------
def v4_keep_ids(df_overall: pd.DataFrame, q=0.25):
    """
    df_overall columns: run_id, NSE_mean, KGE_mean, RMSE_mean
    Worst sets computed on FULL ensemble. Size exactly floor(N*q) using deterministic tie-break by run_id.
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
         "removed_cumulative": len(removed_1), "pct_removed_cum": 100*len(removed_1)/N},
        {"step": 2, "metric": "KGE",  "removed_this_step": len(removed_2),
         "removed_cumulative": len(removed_1 | worst_kge), "pct_removed_cum": 100*len(removed_1 | worst_kge)/N},
        {"step": 3, "metric": "RMSE", "removed_this_step": len(removed_3),
         "removed_cumulative": len(removed_union), "pct_removed_cum": 100*len(removed_union)/N},
        {"step": 4, "metric": "FINAL", "removed_this_step": 0,
         "removed_cumulative": len(removed_union), "pct_removed_cum": 100*len(removed_union)/N,
         "kept": len(keep_ids), "pct_kept": 100*len(keep_ids)/N},
    ])

    flagged = d[["run_id","NSE_mean","KGE_mean","RMSE_mean"]].copy()
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
    PRED_CACHE = Path("pred_cache")  # reuse existing cache
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

    # interactions to include (publishable set)
    INTERACTIONS_SELECTED = [("model","nwp"), ("model","feat"), ("nwp","feat"), ("model","tuner")]

    # factors for main effects
    FACTORS_FULL = ["model", "nwp", "feat", "tuner", "replicate"]
    FACTORS_NO_REP = ["model", "nwp", "feat", "tuner"]

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

    # OLS attribution helper (for interactions)
    if not hasattr(au, "ols_attribution_by_horizon"):
        raise RuntimeError("analysis_utils.py missing ols_attribution_by_horizon (v3 patch).")

    # main-effects eta2 helper is optional; we will fallback to OLS main-only if absent
    has_eta2 = hasattr(au, "attribution_by_horizon")

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
                print(f"[INFO] processed {i+1}/{len(df_runs)}")

        df_metrics = pd.DataFrame(rows)
        df_metrics.to_csv(df_metrics_path, index=False)

    # -----------------------------
    # 1) v4 selection on FULL ensemble (1024)
    # -----------------------------
    df_overall = (df_metrics.groupby("run_id", as_index=False)
                  .agg(NSE_mean=("NSE","mean"), KGE_mean=("KGE","mean"), RMSE_mean=("RMSE","mean")))
    keep_ids, sel_summary, sel_flagged = v4_keep_ids(df_overall, q=Q_WORST)

    sel_summary.to_csv(TABLES / "v4_selection_summary.csv", index=False)
    sel_flagged.to_csv(TABLES / "v4_selection_flags.csv", index=False)

    df_runs_kept = df_runs[df_runs["run_id"].isin(keep_ids)].copy()
    df_metrics_kept = df_metrics[df_metrics["run_id"].isin(keep_ids)].copy()

    print(f"[INFO] v4 kept: {len(keep_ids)}/{len(df_overall)} = {100*len(keep_ids)/len(df_overall):.1f}%")

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
    preds = np.stack(preds, axis=0)  # (M,N,H)

    # deterministic ensemble mean evaluated as a single forecast
    det_rows = []
    for h in range(HORIZON):
        y_obs = y_test_ref[:, [h]]
        mean_pred = np.nanmean(preds[:, :, h], axis=0).reshape(-1, 1)
        per = au.compute_metrics_per_horizon(y_obs, mean_pred, peak_q=PEAK_Q)[0]
        det_rows.append({"horizon": h+1, **per, "n_models": preds.shape[0]})

    df_det_ens = pd.DataFrame(det_rows)
    df_det_ens.to_csv(TABLES / "ensemble_deterministic_metrics_mean_v4.csv", index=False)

    # add IQR across models (kept subset) for deterministic metrics
    df_det_ens_iqr = add_iqr_cols_per_horizon(df_det_ens, df_metrics_kept, cols=[m for m in DET_IQR_METRICS if m in df_metrics_kept.columns])
    df_det_ens_iqr.to_csv(TABLES / "ensemble_deterministic_metrics_mean_with_IQR_v4.csv", index=False)

    # probabilistic interval metrics via ensemble_summary
    ens = au.ensemble_summary(y_test_ref, preds, alpha=ALPHA)  # list/dict per horizon
    prob_rows = [{"horizon": h+1, **ens[h], "n_models": preds.shape[0]} for h in range(HORIZON)]
    df_prob = pd.DataFrame(prob_rows)

    # make interval column robust
    if "IntervalSc" not in df_prob.columns:
        if "IntervalScore" in df_prob.columns:
            df_prob = df_prob.rename(columns={"IntervalScore": "IntervalSc"})
        elif "interval_score" in df_prob.columns:
            df_prob = df_prob.rename(columns={"interval_score": "IntervalSc"})

    # keep only what exists
    prob_cols = [c for c in ["horizon", "CRPS", "IntervalSc", "PICP", "MPIW", "n_models"] if c in df_prob.columns]
    df_prob = df_prob[prob_cols]
    df_prob.to_csv(TABLES / "ensemble_prob_metrics_v4.csv", index=False)

    # -----------------------------
    # 3) Violin plots vs horizon (kept subset)
    # -----------------------------
    violin_metrics = [m for m in ["NSE","KGE","RMSE","MAE","MAPE","r","SCAS"] if m in df_metrics_kept.columns]
    for m in violin_metrics:
        save_violin_by_horizon(
            df_metrics_kept, metric=m,
            out_png=PLOTS / f"violin__{m}__kept_v4__h1to5.png",
            title=f"{m} | kept subset (v4 union-of-worst-25% removal) | distribution across horizons (1–5 days ahead)"
        )

    # -----------------------------
    # 4) Grouped compact tables (publishable)
    # -----------------------------
    headline = [m for m in ["NSE","KGE","RMSE","MAE","MAPE","r"] if m in df_metrics_kept.columns]

    # overall per-run mean (for overall grouped summaries)
    overall_run = (df_metrics_kept
        .groupby(["run_id","model","nwp","feat","tuner","replicate"], as_index=False)
        .agg({m: "mean" for m in headline})
    )
    overall_run.to_csv(TABLES / "overall_metrics_kept_v4_by_run.csv", index=False)

    for f in ["tuner","model","nwp","feat","replicate"]:
        # overall table
        tab = group_compact_table(overall_run, group_col=f, metrics=headline, horizon=None)
        tab.insert(0, "factor", FACTOR_LABEL.get(f, f))
        tab.to_csv(TABLES / f"group_compact_overall_kept_v4__by_{f}.csv", index=False)

        # per-horizon table
        tabs = []
        for h in range(1, HORIZON+1):
            th = group_compact_table(df_metrics_kept, group_col=f, metrics=headline, horizon=h)
            th.insert(0, "horizon", h)
            th.insert(1, "factor", FACTOR_LABEL.get(f, f))
            tabs.append(th)
        pd.concat(tabs, ignore_index=True).to_csv(TABLES / f"group_compact_by_horizon_kept_v4__by_{f}.csv", index=False)

    # -----------------------------
    # 5) Attribution: main effects + interactions (with/without replicate)
    # -----------------------------
    def run_attr(metric: str, factors: list[str], tag: str):
        # main effects explained
        if has_eta2:
            df_eta2 = au.attribution_by_horizon(df_metrics_kept, metric=metric, factors=factors, horizons=(1,2,3,4,5))
            df_eta2.to_csv(TABLES / f"attr_eta2_abs__{metric}__{tag}.csv", index=False)
            main_sum = (df_eta2.groupby("horizon", as_index=False)["eta2"].sum()
                        .rename(columns={"eta2": "explained_main"}))
        else:
            # fallback: use OLS main-only drop-one totals as main explained (R2)
            df_ols_main = au.ols_attribution_by_horizon(df_metrics_kept, metric=metric, main_factors=factors, interactions=None, horizons=(1,2,3,4,5))
            df_ols_main.to_csv(TABLES / f"attr_ols__{metric}__main_only__{tag}.csv", index=False)
            totals = df_ols_main[df_ols_main["term"] == "__TOTAL__"][["horizon","R2_full"]].copy()
            main_sum = totals.rename(columns={"R2_full": "explained_main"})

        main_sum.to_csv(TABLES / f"attr_explained_main__{metric}__{tag}.csv", index=False)

        # interactions (selected)
        df_ols_sel = au.ols_attribution_by_horizon(df_metrics_kept, metric=metric, main_factors=factors, interactions=INTERACTIONS_SELECTED, horizons=(1,2,3,4,5))
        df_ols_sel.to_csv(TABLES / f"attr_ols__{metric}__selected_interactions__{tag}.csv", index=False)

        totals_sel = df_ols_sel[df_ols_sel["term"] == "__TOTAL__"][["horizon","R2_full","unexplained","frac"]].copy()
        totals_sel = totals_sel.rename(columns={"frac": "sum_contrib_dropone"})
        totals_sel.to_csv(TABLES / f"attr_ols_totals__{metric}__selected_interactions__{tag}.csv", index=False)

        # plot main/inter/unexplained (absolute + normalized explained)
        save_stack_main_inter_unexp(
            df_main_sum=main_sum,
            df_ols_totals=totals_sel[["horizon","R2_full"]],
            out_png=PLOTS / f"attr_stack__main_inter_unexp__{metric}__{tag}.png",
            title=f"Variance decomposition (main vs interactions) | metric={metric} | kept subset (v4) | factors: {', '.join([FACTOR_LABEL.get(x,x) for x in factors])}"
        )

        # explained summary (publishable)
        expl = main_sum.merge(totals_sel[["horizon","R2_full","unexplained"]], on="horizon", how="left")
        expl["explained_interactions_total"] = np.clip(expl["R2_full"] - expl["explained_main"], 0.0, 1.0)
        expl.to_csv(TABLES / f"attr_explained_summary__{metric}__{tag}.csv", index=False)

    # run for core metrics (as you want: horizon-wise)
    for met in [m for m in ["NSE","RMSE","KGE"] if m in df_metrics_kept.columns]:
        run_attr(met, FACTORS_FULL, tag="with_replicate")
        run_attr(met, FACTORS_NO_REP, tag="no_replicate")

    print("\n[DONE] v4 outputs written to:")
    print("  ", str(OUTDIR))
    print("  Tables:", str(TABLES))
    print("  Plots :", str(PLOTS))
    print("  Pred cache:", str(PRED_CACHE))


if __name__ == "__main__":
    main()