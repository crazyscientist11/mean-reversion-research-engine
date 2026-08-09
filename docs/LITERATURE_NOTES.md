# Literature notes

This architecture is informed by Xin Li, *Optimal Multiple Stopping Approach to Mean Reversion Trading*, Columbia University (2015). It is an original summary, not a reproduction of the dissertation.

1. Mean reversion alone does not settle trade timing.
2. OU dynamics may describe signed mean-reverting spreads.
3. Exact transition-density maximum likelihood can estimate equilibrium, speed, and volatility.
4. Pair weights may be selected to improve OU fit.
5. Entry and liquidation are distinct, nested decisions.
6. Costs should change thresholds, not merely realized P&L.
7. Stop constraints can alter entry and take-profit regions.
8. Entry can be bounded rather than “lower is always better.”
9. Minimum holding periods alter opportunity value and timing.
10. First-passage times naturally describe threshold strategies.
11. Repeated trading is an optimal-switching problem, not just one entry and exit.
12. Exponential OU can suit positive-valued mean-reverting processes.
13. CIR is generally more natural for positive state variables such as rates or volatility than signed equity residuals.

The repository is inspired by these concepts but does not claim to reproduce the dissertation's analytical results unless and until they are implemented and validated.

