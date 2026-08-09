# Math guide (Step 1 concepts)

- **Price:** observed adjusted close, (P_t).
- **Simple return:** (P_t/P_{t-1}-1).
- **Log price:** (ln(P_t)), defined only for positive prices.
- **Log return:** (ln(P_t/P_{t-1})).
- **Residual/spread:** a later-model output after removing a specified relationship; neither is automatically a tradeable signal.
- **Statistical equilibrium:** a model-implied reference level, distinct from fundamental fair value.
- **Z-score:** standardized distance from a chosen reference distribution; it is a benchmark, not proof of stationarity.
- **Stationarity:** a statistical property assessed with explicit tests and assumptions in a future module.
- **OU concept:** a continuous-time mean-reverting stochastic process that can describe signed residuals; it is not implemented here.
- **First passage:** the time until a process reaches an exit or stop boundary.
- **Optimal stopping:** choosing entry, hold, exit, or stop timing using an explicit objective.
- **Cost-aware thresholds:** thresholds whose economics incorporate trading costs instead of subtracting costs only after the fact.

Detailed derivations, estimator choices, and stopping formulas are deliberately deferred until they can be implemented and validated.

## Step 2 statistical baselines

For the simple benchmark, let `p_t = ln(P_t)`. At date `t`, the baseline is the mean of `p_(t-w)` through `p_(t-1)` and the scale is their **sample** standard deviation (`ddof=1`). The current observation is excluded: `z_t = (p_t - mean_prior) / std_prior`.

The detrended model fits `p_i = alpha + beta i + epsilon_i` over the previous `w` log prices, then extrapolates one step to form the trend-implied log price at `t`. Its residual is actual log price minus that baseline. Residual z-scores also use only earlier residuals.

The factor model uses log returns: `r_target,t = alpha + beta_market*r_market,t + beta_sector*r_sector,t + epsilon_t`. Coefficients are fitted using complete prior returns only; observed factor returns at `t` produce the model-implied target return at that close. `epsilon_t` is a one-period residual return. A rolling sum of `epsilon` creates an accumulated residual state for future study; Step 2 neither tests stationarity nor fits OU dynamics.

## Step 3 pair relationship research

For target A and peer B, Step 3 estimates `ln(P_A) = alpha + beta*ln(P_B) + epsilon` on the `w` observations ending at `t-1`. The observed log prices at `t` then form `epsilon_t = ln(P_A,t) - alpha - beta*ln(P_B,t)`. The reported spread mean and sample standard deviation are calculated from the fitted historical residuals in that same prior window; the current price is excluded.

`beta` is a statistical regression hedge ratio. The illustrative regression-exposure representation is `+1` target unit and `-beta` peer units. This is neither a number of shares nor a dollar-neutral, beta-neutral, or market-neutral portfolio without further conversion and exposure calculations.

Engle–Granger cointegration tests whether the prior-window regression residual is inconsistent with a unit-root null. High correlation is not cointegration: correlated returns can coexist with an unstable long-run price relationship. A cointegration p-value is finite-sample statistical evidence, not proof that a relationship will persist or a guaranteed arbitrage.

## Step 4 Ornstein–Uhlenbeck dynamics

The reusable OU state model is `dX_t = kappa(theta - X_t)dt + sigma dW_t`. Step 4 uses `dt = 1 trading day`: `kappa` is per trading day, `sigma` is per square-root trading day, and time outputs are trading days. No calendar-year or 252-day annualization is applied silently.

- **theta:** model long-run equilibrium level for the supplied residual/spread state.
- **kappa:** conditional mean-reversion speed.
- **sigma:** diffusion volatility of the state.

The AR(1) estimator fits `X_(t+1) = a + b X_t + error`. A usable OU mapping requires `0 < b < 1`, with `kappa = -ln(b)/dt` and `theta = a/(1-b)`. The continuous-time sigma is derived from the exact transition variance: `Var(error) = sigma^2(1-exp(-2*kappa*dt))/(2*kappa)`; it is not simply the AR innovation standard deviation.

The exact-MLE estimator uses `X_(t+dt)|X_t` with conditional mean `theta + (X_t-theta)exp(-kappa*dt)` and conditional variance `sigma^2(1-exp(-2*kappa*dt))/(2*kappa)`. It optimizes the sum of conditional normal log likelihoods subject to positive `kappa` and `sigma`.

For a valid fit, half-life is `ln(2)/kappa`; 75% and 90% conditional expected displacement decay times are `-ln(0.25)/kappa` and `-ln(0.10)/kappa`. Conditional expectation at horizon `h` is `theta + (X_t-theta)exp(-kappa*h)`, with the same exact-transition variance formula using `h`.

These are conditional expectation and distribution statements. They do **not** give the first time a stochastic path hits an exit boundary, guarantee realized convergence, or supply optimal trading thresholds. First-passage and optimal-stopping research remain future work.

## Step 5 PCA and cross-sectional residuals

PCA begins with daily log returns. At date `t`, each security is standardized with mean and standard deviation calculated only on the prior training window ending at `t-1`. PCA loadings and component counts are fitted on that same standardized window. The current return is then projected through the frozen scaler and loadings, reconstructed in standardized units, and converted back to return units.

`residual_return = actual_return - PCA_reconstructed_common_return`. The accumulated PCA residual state is a rolling sum of residual returns; it is not automatically stationary and is not fitted to OU inside this module. Principal-component signs are not inherently meaningful because an eigenvector may be sign-flipped without changing reconstruction.

Cross-sectional deviation compares a stock with peers at the same date. For available residual returns it computes the peer mean, sample standard deviation, rank, percentile, and z-score. This differs from a time-series residual z-score, which asks whether a stock is unusual relative to its own prior residual behavior. The two diagnostics are not interchangeable.
