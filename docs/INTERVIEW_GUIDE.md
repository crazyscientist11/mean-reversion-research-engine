# Interview guide

**Why not mean-revert raw prices?** Prices mix trend, factor exposure, and changing business relationships. The project studies residuals after explicit statistical baselines; it does not call them fair value.

**Why multiple residual engines?** Detrended, factor, pair, Kalman, PCA, and cross-sectional constructions expose different assumptions. Agreement is recorded, but dependent models are not treated as independent votes.

**Why stationarity and OU gates?** A z-score alone does not establish a stable equilibrium. Stationarity diagnostics and a valid OU fit are hard eligibility gates; confidence never overrides them.

**OU intuition and estimators?** `dX=κ(θ-X)dt+σdW` describes conditional pull toward a statistical equilibrium. AR(1) is transparent; exact MLE uses the transition density. Half-life is expected-displacement decay, not a boundary-hitting time.

**First passage and stopping?** First passage simulates exit-before-stop frequencies under fixed OU assumptions. Finite-horizon stopping compares entry/hold/exit/stop values after stated costs, so entry regions can be bounded.

**Static versus Kalman beta?** Static beta is one prior-window relationship. Kalman beta is a latent, time-varying estimate with prior and filtered states; neither is a claim of market neutrality. OU-likelihood optimization chooses a transparent grid candidate by fit, not by profit.

**How is lookahead prevented?** Rolling models fit only data permitted at each date. In historical research a signal formed at close *t* executes at close *t+1*. Regression tests modify future observations and verify earlier outputs do not change.

**Where does Bloomberg fit?** The public project accepts locally exported CSVs, normalizes them explicitly, and never requires Bloomberg APIs or licensed data. Exports are ignored by Git.

**What does a frozen prediction mean?** A snapshot records model-time parameters, probabilities, boundaries, and action. A later evaluation compares new prices with that frozen record; it does not refit or rewrite history.

**What do backtests prove?** They can describe historical outcomes, costs, calibration, and failure modes. They do not prove profitability, live execution quality, or suitability for capital.
