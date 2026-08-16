# Step 2 output guide

These are **research outputs**, not recommendations. They describe a statistical model and do not establish that a price will move, that a residual is stationary, or that a result is economically tradeable.

## Step 10 final decision monitor

The final decision object is a frozen research record, not a personalized recommendation. It can report `ENTER_LONG`, `ENTER_SHORT`, `WAIT`, `HOLD_LONG`, `HOLD_SHORT`, `EXIT_LONG`, `EXIT_SHORT`, `STOP_LONG`, `STOP_SHORT`, `NO_SIGNAL`, `CONFLICTED`, or `INSUFFICIENT_DATA`.

Critical gates are evaluated first. Insufficient data, invalid residuals, failed stationarity, invalid OU fits, unacceptable half-life, parameter instability, and `SEVERE_BREAK` always prevent a timing signal; confidence and consensus cannot override them. Consensus combines valid model evidence but limits the effective vote of models in a shared dependency group. The 0--100 confidence score is calculated only after gates pass. Its documented components are deviation (capped at |z|=2.5), stationarity, OU agreement, half-life, stability, regime, consensus, first-passage quality, economic value, and boundary robustness.

`WhySignalResult` lists every material check with its measured value, threshold, PASS/CAUTION/FAIL status, and explanation. A saved `FinalResearchDecision` is serialized inside the immutable prediction snapshot; later evaluations are separate records and do not modify it.

- **Price:** the supplied adjusted close.
- **Log price:** `ln(price)`; used because relative changes become additive.
- **Return:** the close-to-close log return, `ln(P_t / P_{t-1})`.
- **Expected return / model-implied return:** a return estimated from contemporaneous market and optional sector returns using coefficients fitted only through the preceding observation. It is not a forecast of fundamental value.
- **Statistical baseline:** the prior rolling log-price mean or extrapolated trend used as a comparison point.
- **Trend-implied price:** `exp(trend-implied log price)`. It is a statistical trend baseline, not fair value.
- **Residual:** actual minus model-implied value. A factor residual is one-period target-specific log return after the factor model.
- **Z-score:** the current deviation divided by the *prior* sample standard deviation of the relevant history. It measures historical scale; it is not a buy/sell signal.
- **Beta:** fitted sensitivity of target log returns to a factor return. It is window-specific and may be unstable.
- **R-squared:** in-window fraction of return variation described by the fitted regression. It does not validate future fit or reversion.
- **Accumulated factor residual:** rolling sum of factor residual returns. It is a residual state candidate for later research, not proof of stationarity or an OU process.

Invalid outputs mean there were not enough prior complete observations or there was effectively no historical variation to standardize. They should not be replaced with zero.

## Step 3 pair outputs

- **Pair regression:** a prior-window relationship between target and peer log prices.
- **Hedge ratio (regression beta):** the fitted log-price coefficient. It is historical and does not by itself specify shares, dollars, or market-neutral exposure.
- **Pair spread:** target log price minus the fitted target log-price baseline implied by the peer.
- **Spread z-score:** current spread relative to the mean and sample standard deviation of prior-window fitted residuals. A positive value means the target is relatively high versus the fitted relationship; a negative value means relatively low. Neither is a recommendation.
- **Relative state:** `TARGET_RELATIVELY_HIGH`, `TARGET_RELATIVELY_LOW`, or `NEUTRAL`, based only on the configured z-score threshold; `INVALID` means a calculation was unavailable.
- **Cointegration statistic and p-value:** prior-window Engle–Granger outputs. They are not guarantees that the relationship remains stable.
- **Return correlation:** correlation of prior-window log returns. It is reported for context and is not a pair-selection rule.
- **Peer comparison table:** a transparent metric table. Its rank sorts available diagnostics for review; it does not automatically choose or approve a pair.

Structural breaks, changing business exposures, corporate actions, and model instability can invalidate a historical pair relationship.

## Step 4 OU outputs

- **Theta:** the OU model’s estimated long-run residual/spread level, not a fundamental value.
- **Kappa:** estimated conditional speed of expected displacement decay, per trading day in this project.
- **Sigma:** estimated OU diffusion volatility per square-root trading day.
- **Half-life / 75% / 90% times:** trading-day times for the OU *conditional expectation* to decay by those amounts. They are not guaranteed realized exit or hitting times.
- **Log likelihood, AIC, BIC:** in-sample fit diagnostics; they do not prove the OU model is correct or stable.
- **Conditional path:** the model-implied expected state at each future horizon from the current state.
- **Model-implied bands:** conditional OU distribution bands around that expected path. They are not price forecasts or guaranteed confidence intervals.
- **AR(1) versus exact MLE differences:** estimator disagreement made visible as a diagnostic, not an automatic rejection rule.

Invalid results are displayed as invalid. The application never coerces a constant, non-finite, explosive, or optimizer-failed series into a mean-reverting model.

## Step 5 PCA and cross-sectional outputs

- **PCA reconstructed return:** the part of a current return represented by components fitted only on prior standardized returns.
- **Idiosyncratic residual:** actual return minus PCA reconstructed return. It is a model-specific residual, not a valuation statement.
- **Explained variance:** fraction of prior standardized-universe variation represented by a component; it is in-sample and can change.
- **Loadings:** component coordinates for each stock. Component sign orientation is arbitrary.
- **Accumulated residual state:** rolling sum of PCA residual returns; it is not proof of stationarity.
- **Cross-sectional rank, percentile, and z-score:** how a residual compares to the available peer universe at that date.
- **RELATIVELY_HIGH / RELATIVELY_LOW / NEUTRAL:** descriptive cross-sectional research states, not buy/sell signals and not claims that reversal will occur.

Missing securities reduce the available peer universe. If the configured minimum universe is unavailable, the cross-sectional output stays invalid rather than silently filling values.

## Step 6 dynamic pair outputs

Kalman **prior beta** is the relationship before the current observation; **filtered beta** is after it. Innovation is actual minus prior model-implied log target; standardized innovation divides by its model variance. OU-likelihood optimized beta is an in-sample statistical construction chosen by likelihood, not a claim of future performance. A weak-identification flag means nearby betas have nearly equal training likelihood.

## Step 8 first-passage outputs

Exit-first and stop-first probabilities are seeded Monte Carlo estimates using fixed OU parameters. Monte Carlo SE describes sampling uncertainty. Median exit time is conditional on simulated exits. Favorable/adverse excursions are path distributions, not outcomes or recommendations.
