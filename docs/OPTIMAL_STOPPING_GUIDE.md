# Paper-inspired numerical optimal stopping

Xin Li (2015), Chapter 2, analytically studies OU double stopping with transaction costs and stop-loss constraints under its stated continuous-time assumptions. This repository does **not** reproduce that analytical solution.

Our implementation is a finite-horizon, discrete-trading-day numerical extension. It uses exact discrete OU transition probabilities over a configurable state grid, backward value iteration, explicit flat/long-residual/short-residual states, transaction costs, an annual opportunity rate converted as `(1+r_annual)^(-1/252)`, and fixed stop boundaries. Short-residual behavior is our numerical extension, not claimed as a direct paper result.

Costs enter entry and exit values before action selection. Stops are forced state transitions, so they affect entry value rather than being subtracted only afterward. This can yield bounded entry regions. Results are model research states, not recommendations. First-passage simulation can describe a selected policy but does not override the value iteration unless a future explicit gate does so.

Key limitations: grid truncation/tail approximation, finite horizon, approximate residual-to-P&L mapping, no parameter uncertainty, no minimum holding period, and no claim of analytical optimality.
