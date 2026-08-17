# Mean Reversion Research Engine

An educational quantitative-research workstation for testing whether carefully constructed equity residuals exhibit measurable convergence. It is not a live trading system, investment recommendation, or evidence of future profitability.

## Objective and thesis

Raw prices combine trends, common factors, and changing relationships. This project compares a transparent fixed-z benchmark with residual constructions such as detrended log price, factor residuals, static/dynamic pairs, PCA, and cross-sectional states. The research thesis is that large, stationary, stable residual dislocations may be more informative than raw-price deviations; it is a testable hypothesis, not a conclusion.

## Features

- Wide and guided Bloomberg-style CSV normalization without Bloomberg APIs or licensed data.
- Prior-window residual engines, pair/cointegration diagnostics, Kalman pairs, PCA, OU AR(1)/MLE, first-passage simulation, and cost-aware finite-horizon stopping research.
- Gate-first consensus, explainable final research states, frozen predictions, and later live evaluations.
- Strict walk-forward event studies and benchmark research where a close-*t* signal executes at close *t+1*.

See [architecture](docs/ARCHITECTURE.md), [math guide](docs/MATH_GUIDE.md), [Bloomberg CSV guide](docs/BLOOMBERG_CSV_GUIDE.md), and [walk-forward guide](docs/WALK_FORWARD_GUIDE.md).

## Install and run

Requires Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run mean_reversion_engine.py
```

Use the **Data workspace** to upload `Date,TICKER,...` wide CSVs or explicitly map repeated date/price pairs. Do not commit Bloomberg exports: the repository ignores `data/private/`, `data/bloomberg/`, and `*.bbg.csv`.

## Validation

```powershell
python -m pytest -q
python scripts/run_smoke_tests.py
```

## Limitations

Models are fitted to historical data and are vulnerable to parameter uncertainty, structural breaks, transaction-cost assumptions, selection bias, and data-quality issues. First-passage and stopping results are numerical research outputs. Backtests and calibration tables do not establish live performance, profitability, or suitability for capital.
