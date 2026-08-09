# Mean Reversion Research Engine

An extensible workstation for testing whether carefully constructed equity residuals show measurable mean reversion. This is an educational quantitative-research project, not a live trading system, broker integration, investment recommendation, or evidence of future profitability.

## Current status

**Step 5 — PCA and cross-sectional research.** The repository provides validated daily CSV ingestion, non-mutating transforms, immutable forward-prediction infrastructure, prior-window single-stock/pair residual research, OU diagnostics, PCA common-factor reconstruction, and cross-sectional residual states. It intentionally does not implement Kalman filtering, optimal stopping, first-passage analysis, live trading, or Bloomberg integration.

## Research question

After removing deterministic and systematic effects, do sufficiently large, stationary, stable residual dislocations converge more reliably than raw price deviations? The hypotheses are explicitly testable—not established claims—at [docs/RESEARCH_THESIS.md](docs/RESEARCH_THESIS.md).

## Architecture

CSV inputs become a source-independent `MarketDataBundle`. Later stages will add residual engines, diagnostics, OU and first-passage layers, and economic threshold research. A `PredictionSnapshot` freezes the model-time record; `PredictionEvaluation` records later outcomes separately. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Data and privacy

The included sample is **SYNTHETIC** and is not market history. Never commit licensed data, credentials, or Bloomberg exports. Private data paths and `*.bbg.csv` are ignored. Bloomberg is a documented future adapter only; the public repository runs without Bloomberg libraries.

## Installation and commands

Requires Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_smoke_tests.py
python -m streamlit run mean_reversion_engine.py
```

## Repository layout

- `src/data/`: provider contract, CSV loader, validation, transforms
- `src/prediction/`: immutable snapshot/evaluation schemas and JSON store
- `docs/`: thesis, math, architecture, data, literature, and roadmap notes
- `data/`: synthetic example only
- `tests/`: deterministic foundation tests

## Planned models

Future work includes simple z-score benchmarks; drift, factor, pairs, Kalman, PCA and cross-sectional residuals; AR(1) and exact-MLE OU estimation; first-passage simulation; cost- and stop-aware optimal stopping; walk-forward research; and a richer Streamlit workstation. The staged scope is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Limitations

No model or signal is implemented in Step 1, daily data are the initial design frequency, and no result should be interpreted as a recommendation or profitability claim.
