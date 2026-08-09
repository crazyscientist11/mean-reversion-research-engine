# Agent instructions

- Read `docs/RESEARCH_THESIS.md` before changing strategy logic and `docs/LITERATURE_NOTES.md` before changing OU or optimal-stopping logic.
- Do not silently alter formulas or sign conventions. Preserve the distinctions among price, log price, return, residual, spread, statistical equilibrium, and fundamental fair value.
- Add regression tests for every confirmed quantitative bug. Never weaken a valid test merely to make it pass.
- Never introduce lookahead or fit scalers, regressions, PCA, stationarity, or OU models using future observations.
- Preserve the distinction between historical backtests and true forward prediction snapshots. Never mutate frozen predictions with realized future data.
- Never commit proprietary data, credentials, or API keys. Do not add live brokers, order execution, or investment recommendations.
- Run the complete suite after meaningful changes, report exact commands run, and never claim a test passed unless it ran.
- Do not claim profitability. Describe outputs as research signals, never recommendations.

