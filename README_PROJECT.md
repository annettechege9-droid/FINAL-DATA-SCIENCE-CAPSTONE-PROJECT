# PowerPulse Kenya - project folder (Annette Chege)

Forecasting monthly KPLC electricity sales (GWh) from KNBS data with time series analysis and machine learning.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook PowerPulse_Capstone.ipynb     # run all cells (~40 s); writes everything in outputs/
streamlit run app.py                           # interactive dashboard (needs the notebook outputs)
```

## Contents
| Path | What it is |
|---|---|
| `PowerPulse_Capstone.ipynb` | Fully explained notebook: data triage -> EDA -> statistical tests -> models -> evaluation -> explainability -> forecast -> conclusions |
| `powerpulse_core.py` | All shared code (data combining, SARIMA, ETS, ML models, rolling-origin evaluation, metrics, tests) |
| `app.py` | Streamlit dashboard (8 pages) |
| `PowerPulse_Capstone_Slides.pptx` | Presentation slides with speaker notes |
| `data/` | The three source CSVs that are combined, `powerpulse_final_dataset.csv` (result), and `reference/` with the four earlier files that were tested and NOT stacked |
| `outputs/` | Result tables (CSV), figures (`figs/`), forecasts |

## Key decisions
* **Combined:** 2019-2021 file + 2022-2023 file + 2024-2026 master = 87 months, no overlaps.
* **Not combined:** AUDIT_ONLY (identical copy), WORKING_NOT_FINAL (earlier concatenation), anomaly_review (derived outputs), verification_register (empty audit template, kept as metadata).
* **Target:** KPLC sales (GWh) - not consumption, not demand. **Horizon:** 12 months.
* **Split:** initial training Jan 2019-Dec 2021; validation targets Jan 2022-Dec 2024 (all choices); test targets Jan 2025-Mar 2026 (held out); expanding-window rolling origin.
* **Honest caveats:** 2019-2023 values are collected but not independently verified; May-June 2025 look anomalous and are analysed with and without adjustment.
