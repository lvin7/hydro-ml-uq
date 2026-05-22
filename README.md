# HYDRO-ML-UQ: Machine Learning Workflow for Uncertainty Quantification in Streamflow Forecasting

Operational streamflow forecasts are essential for flood preparedness and reservoir management. However, predictive uncertainty is often poorly characterized, limiting the reliability of decisions based on these forecasts. This repository implements an **end-to-end multi-model uncertainty quantification (UQ) framework** for operational streamflow forecasting. Unlike standard approaches that treat uncertainty as a property of the predictive model alone, this framework decomposes and attributes uncertainty across the **entire forecasting workflow**: meteorological forcing choice, feature design, model architecture, hyperparameter optimization (HPO), and training variability.

The framework enables systematic analysis of how uncertainty emerges, propagates, and shifts across forecasting lead times, providing actionable guidance for improving forecast reliability and decision-making.

---

## Workflow Overview

![HYDRO-ML-UQ Workflow](figures/hydro_ml_uq_workflow.png)

The workflow evaluates multiple combinations of:

- **NWP forcings:** IFS, GFS, UKMO, GEM  
- **Feature sets:** Q, Q+p, Q+p+T, Q+p+T+s, Q+p+T+s+d  
- **Model architectures:** LSTM, TCN, TKAN, MLP  
- **HPO strategies:** Random Search, Bayesian Optimization, Hyperband, Evolutionary  
- **Training replicates:** multiple independent runs to capture variability

Ensembles explore all combinations, systematically quantifying uncertainty contributions from each stage.

---

## Repository Contents

- `run_experiments.py`: trains the full model grid, writes trained models, scalers, metrics, and tuning summaries  
- `data_utils.py`: data loading, cleaning, synchronization, scaling, train/validation/test splitting 
- `hp_tuning.py`, `models.py`, `metrics.py`: training, HPO, and diagnostic utilities 
- `full_pipeline_analysis.py`: scans trained runs, computes ensemble summaries, selects retained models, and generates analysis tables  
- `analysis_utils.py`: model-loading, caching, metrics, ensemble, and attribution helpers  
- `figure2.py` – `figure5.py`: regenerate paper figures from retained runs  
  

Store large data/model/output folders:

- `data/`  
- `models/`  
- `final-models/`  
- `analysis_out_v4/`  
- `figures/`  
- `pred_cache/`, `pred_cache_v4/`  

---

## Environment Setup

Recommended: Python 3.11

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Expected Data Layout

Before running the scripts, your `data/` folder should have cleaned input CSVs:

```text
data/
  Q.csv
  clean_<nwp>_<variable>_lead_<lead>h.csv
```

Where
- **`Q.csv`** → observed discharge  
- **`clean_<nwp>_<variable>_lead_<lead>h.csv`** → preprocessed NWP forcings  
- **Default NWP sources:** ifs, ukmo, gfs, gem  
- **Default variables:** tp_daily, t2m_raw, sd_raw  
- **Lead times:** 24, 48, 72, 96, 120 hours
