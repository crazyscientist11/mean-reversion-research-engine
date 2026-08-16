# Walk-forward research

For daily close data, a signal is formed using information available through close **t** and is executed at close **t+1**. This prevents an action from being priced at the same close that supplied its signal. Every rolling statistic and optional full-model callback is supplied only history through its signal date; outcomes are measured separately afterward.

The fixed-z benchmark uses configurable research settings (defaults: entry `|z| >= 2.0`, exit `|z| <= 0.5`, stop `|z| >= 4.0`). It is a transparent benchmark, not an optimal strategy. Pair accounting uses both legs and explicit costs; a single-stock residual strategy is not called market neutral without actual hedge holdings.

Chronological train/validation/test splits never shuffle rows. Parameter sweeps do not select a winner automatically. Calibration compares original predicted probabilities with later realized exit frequency and should not be called calibrated without supporting out-of-sample evidence.
