# Live prediction monitor

The monitor separates a **frozen model** from a later **live observation**. Creating a prediction saves the data cutoff, fitted parameters, boundaries, probabilities, confidence, action, and parameter hash in an immutable `PredictionSnapshot`. Entering a later price never refits OLS, factor or pair coefficients, PCA, Kalman state, OU parameters, diagnostics, regimes, boundaries, probabilities, or confidence. Creating a new model requires the explicit “Create new frozen prediction” action and produces a new ID.

For a frozen pair model, the live residual is `log(P_target) - alpha - beta*log(P_peer)`, using the original alpha and beta. A residual boundary maps to a live implied target price as `exp(X_boundary + alpha + beta*log(P_peer_live))`. The residual boundary stays fixed; the implied target price moves as the peer price moves. When a model has no clean exact price inversion, the monitor shows the residual/state boundary instead of inventing a dollar target.

An evaluation is a separate `PredictionEvaluation` record containing the timestamp, elapsed trading days, live prices, residual, region distances, fraction reverted, exit/stop status, excursions, and P&L when available. Fraction reverted is measured in residual space from the original residual toward the original exit boundary and is intentionally not clipped: negative values mean the residual moved farther from the target.

This workflow can sit beside a Bloomberg process: export historical prices to fit and freeze a model at cutoff T, then enter current prices later for comparison. Bloomberg connectivity is deliberately not implemented here.
