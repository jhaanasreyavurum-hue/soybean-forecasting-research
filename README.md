# Soybean Forecasting & Market Analytics

A combined research repository and interactive Streamlit dashboard for soybean market-price analysis. It brings together forecasting notebooks, cleaned market datasets, model outputs, graphs, and a seven-page dashboard.

## Project overview

The dashboard compares historical soybean price behavior and forecast performance for **Indore and Gondal**. The research materials also include a Rajkot-labelled cleaned workbook and files whose names refer to Gondal/Rajkot; confirm the market naming and location mapping before treating those labels as identical.

The supplied research includes statistical time-series models, volatility models, machine-learning experiments, deep-learning experiments, model summaries, and prediction outputs. The dashboard uses stored results; it does not retrain all models when launched.

## Dashboard pages

| Page | Purpose |
|---|---|
| Overview | KPI cards and model leaderboard |
| Price explorer | Price history, market spread, arrivals, seasonality, crude oil |
| Model comparison | Forecast errors, skill versus naive baseline, model diagnostics |
| Forecast lab | Actual vs predicted prices, errors, direction accuracy, next-days forecast |
| Volatility | Rolling volatility, ARCH/GARCH comparison, interval calibration |
| Market & policy | Price vs MSP, storage gain, arrival pressure, market relationships |
| Insights & data | Findings, dataset inventory, data quality, downloads, caveats |

## Models represented in the materials

- ARIMA and SARIMA
- ARIMAX and SARIMAX
- VAR and VECM
- ARCH and GARCH
- Machine-learning experiments
- LSTM and related deep-learning experiments

Not every model is implemented or retrained by the Streamlit app. Refer to the individual notebooks and result files for their particular setup and metrics.

## Repository layout

```text
.
├── app.py                      # Streamlit entry point
├── common.py                   # Shared dashboard loaders and helpers
├── requirements.txt            # Dashboard dependencies
├── data/                       # Dashboard input and stored model results
├── pages/                      # Seven Streamlit pages
├── scripts/                    # Dashboard data preparation/build scripts
└── research/
    ├── notebooks/              # Supplied executed Jupyter notebooks
    ├── datasets/               # Cleaned market CSV/XLSX files
    ├── results/                # Model summaries, predictions, audit files
    ├── figures/                # Graph outputs supplied in an archive
    ├── archx_lstm_project/      # Supporting model code and notes
    └── source_archives/         # Original nested ZIP files supplied with project
```

## Run the dashboard locally

Python 3.10 or newer is recommended. Open this repository folder in VS Code, then run these commands in the integrated terminal.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The terminal will show a local URL, usually `http://localhost:8501`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Research notebooks

Open the `.ipynb` files in `research/notebooks/` using VS Code (with the Jupyter extension) or Jupyter Notebook. Notebook dependencies can vary. `research/requirements.txt` lists common research libraries; it includes TensorFlow and may install large packages. The dashboard's root `requirements.txt` is separate and is sufficient for the dashboard itself. Some notebooks originated in Google Colab and may require local file-path/input adjustments before they run in VS Code.

## Data and interpretation notes

- The app's current market labels are Indore and Gondal. Some research files use Rajkot in their filenames; verify the intended market/location before combining series or describing results.
- The app README supplied with the project notes that the crude-oil units were not verified and some daily values were forward-filled.
- Forward-filled values can create repeated prices. Check the data-preparation steps and missing-value treatment in the relevant scripts/notebooks before drawing conclusions.
- Forecast metrics should be compared on the same test period and target frequency. Lower error is not, by itself, evidence of a statistically significant improvement.
- Check data provenance, usage rights, and any redistribution restrictions before making the repository public.

## Reproducibility

The repository includes notebook outputs and stored dashboard results for inspection. The dashboard displays those saved outputs. Re-running the original experiments may require the source datasets, notebook-specific libraries, and following each notebook's preprocessing and train/test split.

## License

No license is declared. Unless a license is added, others should not assume they have permission to reuse or redistribute the code or data. Confirm that you have permission to publish the included data before making this repository public.
