# Oulu airport temperature forecasting with TimesFM 3.0

Zero-shot hourly temperature forecasting for Oulu airport, Finland, using Google's [TimesFM 3.0](https://github.com/google-research/timesfm) time-series foundation model and open data from the Finnish Meteorological Institute (FMI).

Task: given the hourly temperature history up to now, predict the next 72 hours with a 10-90% uncertainty band. The model is pretrained and used as is, with no fine-tuning.

Everything is in one notebook: [`oulu-temp-forecast.ipynb`](oulu-temp-forecast.ipynb)

## What the notebook does

1. Loads hourly FMI observations (June 2019 to October 2026) and handles gaps and daylight saving
2. Predicts the next 72 hours from the latest data
3. Pulls live observations from the FMI API and compares TimesFM with FMI's own forecast
4. Backtests 287 forecasts over 2026 against three baselines and four context lengths
5. Checks whether the gains are real with a paired block bootstrap
6. Holds out October 2026 to compare a single long forecast with daily reruns

## Results

287 forecasts issued at midnight, 1 January to 14 October 2026. MAE in °C.

| Method | 1h | 6h | 24h | 72h | Mean 1-72h | Inside 10-90% band |
|---|---|---|---|---|---|---|
| Persistence | 0.65 | 2.29 | 2.66 | 3.61 | 3.33 | |
| Same hour yesterday | 2.72 | 3.11 | 2.66 | 3.61 | 3.22 | |
| Climatology (2019-2025) | 3.85 | 3.88 | 3.87 | 3.82 | 3.94 | |
| TimesFM 3.0, 1 year context | 0.37 | 1.39 | 2.54 | 3.37 | 2.72 | 81% |

- TimesFM is 18% better than persistence averaged over all 72 hours, and about 40% better at 1 to 6 hours ahead.
- The 10-90% band holds 81% of actual values, close to the ideal 80%.
- Winter is the hard part. In January TimesFM was slightly worse than persistence. From March to October it was better every month.
- Context length (3 weeks to 1.75 years) makes little difference.
- In October 2026, one forecast for the whole month had an MAE of 3.92 °C and missed a cold snap completely. Rerun every midnight it dropped to 1.66 °C, 14% better than persistence.

The biggest errors come from weather arriving from outside Oulu, which past temperatures at one station cannot see. A natural next step is to give FMI's own forecast to TimesFM as a covariate.

## Setup

Tested on Windows with Python 3.12, PyTorch 2.11 (CUDA 12.8) and an RTX 3080 Laptop GPU.

```
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/google-research/timesfm.git ..\timesfm-src
pip install -e "..\timesfm-src[torch]"
pip install -r requirements.txt
jupyter notebook
```

The model weights (about 1.3 GB) download from Hugging Face on first run.

## Data

Hourly mean air temperature at Oulu airport from FMI's [Download observations](https://en.ilmatieteenlaitos.fi/download-observations) service, in `data/`. The live section uses the [FMI open data API](https://en.ilmatieteenlaitos.fi/open-data).

## Credits and licences

- Weather data: Finnish Meteorological Institute, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Model: [TimesFM](https://github.com/google-research/timesfm) by Google Research. The 3.0 weights are under the TimesFM Non-Commercial License, so this project is for learning and research only.
