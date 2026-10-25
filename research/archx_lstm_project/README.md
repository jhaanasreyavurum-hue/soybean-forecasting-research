# Soybean weekly forecasting — ARCHX and LSTM only

This project adapts the supplied tomato project structure to the uploaded soybean modal-price files and crude-oil CSV. It forecasts Indore and Gondal (Rajkot district) separately and compares held-out forecasts against a naive last-week benchmark.

## Files
- `data/price_indore.csv` — Indore soybean modal price.
- `data/price_gondal_rajkot.xlsx` — Gondal soybean modal price.
- `data/crude_oil.csv` — Date/Price crude oil series.
- `run_models.py` — only ARCHX-style model and LSTM variants are attempted.
- `outputs/` — generated aligned weekly series, audit, predictions, metrics and run notes.

## Run
Use Python 3.10–3.12 (TensorFlow wheel support varies by Python version):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_models.py
```

In Colab, upload/extract this folder, then run `!pip install arch tensorflow openpyxl scikit-learn` and `!python run_models.py`.

## Data and method choices
- Price target: `p_modal`, interpreted as INR/quintal based on the market data schema.
- Weekly frequency: weeks ending Friday (`W-FRI`), with means of the daily observations that actually exist in each week. Soybean prices are never forward-filled.
- Date alignment: only rows with target price, oil price, target return and lagged oil return are retained. The oil feature is lagged one complete week.
- Split: chronological 80% train / 20% test; no random shuffle. Metrics are computed only on the held-out test period.
- Models: (1) naive last-week price benchmark; (2) ARCHX-style ARX mean with ARCH(1) conditional variance and lagged oil return; (3) LSTM using past soybean returns; (4) LSTM-X adding lagged oil return.
- LSTM forecasts are rolling one-step-ahead, using prior observed price to convert predicted log returns back to price. The neural network parameters are fit on the training window only.

## Important caveats
1. **Crude-oil unit/source is not confirmed by the CSV alone.** Its values look plausible for USD/barrel, but check the original source metadata. Do not convert to INR without a dated USD/INR series. For these models the lagged log-return is unit-invariant to a constant currency conversion, so the oil return feature can be used without FX conversion.
2. **ARCHX terminology:** the `arch` package places exogenous regressors in the conditional-mean (ARX) equation, not in a general ARCH variance equation. This script therefore means ARX + ARCH(1), with oil in the mean. Do not claim crude oil directly explains conditional variance. A true variance-equation ARCH-X requires a custom likelihood / other implementation.
3. **Oil data frequency/coverage:** weekly oil values are means of observed daily prices. Weeks without oil observations are not filled; rows requiring them are excluded after lagging.
4. **Missing market days:** weekly means use available records, not fabricated daily prices. This means weekly observations can have differing counts of daily prices; inspect `input_audit.csv` and aligned series.
5. **No results are claimed until the script finishes successfully.** TensorFlow and `arch` must be installed. The output metrics compare the exact same test dates within each market.

## Outputs
- `outputs/input_audit.csv`
- `outputs/indore_weekly_aligned.csv`, `outputs/gondal_weekly_aligned.csv`
- `outputs/out_of_sample_metrics.csv`
- per-model `*_predictions.csv`
- `outputs/run_notes.json`
