# Validation report

The repository uses synthetic data and deterministic unit tests. The suite covers CSV validation/normalization and data quality, prior-window residual models, pair algebra and cointegration behavior, PCA/cross-sectional calculations, Kalman and OU-likelihood pair methods, OU AR(1)/MLE formulas and recovery, first-passage simulation, stopping-policy gates/costs, consensus/confidence, frozen prediction/evaluation immutability, live boundary mappings, and chronological backtest accounting/no-lookahead behavior.

Analytical checks include exact OU transition variance, AR(1)-to-OU parameter mapping, half-life, seeded Monte Carlo reproducibility, pair residual inversion, and basis-point/fixed-cost accounting. No-lookahead tests cover rolling model families and a future-data perturbation test for historical benchmark signals.

Limitations remain material: synthetic tests are not market validation; some workstation displays expose only a subset of model modules; probability estimates need out-of-sample calibration; stopping uses finite grids; no live market or Bloomberg API is included; and historical backtests remain susceptible to selection, survivorship, and execution assumptions.
