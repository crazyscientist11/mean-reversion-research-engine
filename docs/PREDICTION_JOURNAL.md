# Prediction journal

A backtest reconstructs a historical decision. A forward `PredictionSnapshot` records what the model actually said using only data available at a stated cutoff. It contains identity, time, source, model version, deterministic parameter hash, and empty structured fields for future model outputs.

Future facts belong in separate `PredictionEvaluation` records. The store rejects duplicate snapshot IDs unless both configuration and the individual call explicitly opt into overwrite. Evaluation JSON is stored in a different directory, so recording outcomes cannot rewrite the original snapshot. This supports later calibration work: predicted convergence probability versus realized frequency.

