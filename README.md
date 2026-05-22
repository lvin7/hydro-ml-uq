# unc-2

Code for training, analysing, and plotting the uncertainty-analysis experiments for multi-step discharge forecasting.

## Repository Contents

- `run_experiments.py`: trains the model grid and writes trained models, scalers, metrics, and tuning summaries.
- `full_pipeline_analysis.py`: scans trained runs, computes deterministic and probabilistic ensemble summaries, selects the retained ensemble, and writes analysis tables.
- `figure2.py` to `figure5.py`: regenerate the paper figures from retained runs and analysis outputs.
- `data_utils.py`: data loading, cleaning, synchronization, scaling, and train/validation/test splitting.
- `analysis_utils.py`: shared model-loading, caching, metrics, ensemble, and attribution helpers.
- `hp_tuning.py`, `models.py`, `metrics.py`: training-time model, tuning, and diagnostic helpers.

Generated artifacts are intentionally not tracked in git. Store large data/model/output folders in the Zenodo record or another archive:

- `data/`
- `final-models/` or `models/`
- `analysis_out_v4/`
- `figures/`
- `pred_cache/`, `pred_cache_v4/`

## Environment

Python 3.11 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Expected Data Layout

By default, scripts expect cleaned input files in `data/`:

```text
data/
  Q.csv
  clean_<nwp>_<variable>_lead_<lead>h.csv
```

The default NWP sources are `ifs`, `ukmo`, `gfs`, and `gem`. The default variables are `tp_daily`, `t2m_raw`, and `sd_raw`; lead times are `24`, `48`, `72`, `96`, and `120` hours.

## Reproducing the Analysis

If trained models are not already available, run the experiment grid:

```bash
python run_experiments.py --outdir final-models
```

Then compute analysis tables and prediction caches:

```bash
python full_pipeline_analysis.py
```

Regenerate figures:

```bash
python figure2.py
python figure3.py
python figure4.py
python figure5.py
```

The default workflow writes tables under `analysis_out_v4/tables/`, plots under `analysis_out_v4/plots/`, figure files under `figures/`, and prediction caches under `pred_cache/` or `pred_cache_v4/`.

## Notes For Archival Use

- The scripts use relative paths by default and can be redirected with command-line arguments.
- Large generated artifacts should be included in the Zenodo archive if exact reproduction without retraining is required.
- `metrics.py` is still used by `run_experiments.py` for training diagnostics and should remain in the repository.

## License

This code is released under the MIT License. See `LICENSE`.
