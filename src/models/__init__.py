"""Step 4 research models; no trading, first-passage, or optimal-stopping logic."""
from .benchmark import rolling_zscore_benchmark
from .drift import detrended_log_price_residual
from .factors import factor_residual_model
from .pairs import peer_comparison, rolling_pair_model
from .ou import fit_ou_ar1, fit_ou_both, fit_ou_mle, rolling_ou_fits

__all__ = ["detrended_log_price_residual", "factor_residual_model", "fit_ou_ar1", "fit_ou_both", "fit_ou_mle", "peer_comparison", "rolling_ou_fits", "rolling_pair_model", "rolling_zscore_benchmark"]
