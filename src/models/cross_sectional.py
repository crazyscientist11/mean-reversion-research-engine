"""Cross-sectional residual ranking; no trade direction or reversal claim."""
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class CrossSectionalState(str, Enum):
    RELATIVELY_HIGH = "RELATIVELY_HIGH"
    RELATIVELY_LOW = "RELATIVELY_LOW"
    NEUTRAL = "NEUTRAL"
    INVALID = "INVALID"


@dataclass
class CrossSectionalResult:
    dates: pd.DatetimeIndex
    residuals: pd.DataFrame
    means: pd.Series
    standard_deviations: pd.Series
    ranks: pd.DataFrame
    percentiles: pd.DataFrame
    zscores: pd.DataFrame
    states: pd.DataFrame
    valid_observations: pd.Series
    minimum_universe_size: int

    def current_table(self) -> pd.DataFrame:
        date = self.dates[-1]
        return pd.DataFrame({"ticker": self.residuals.columns, "residual": self.residuals.loc[date], "zscore": self.zscores.loc[date], "percentile": self.percentiles.loc[date], "relative_state": self.states.loc[date]}).reset_index(drop=True)


def cross_sectional_residuals(residual_returns: pd.DataFrame, *, minimum_universe_size: int = 3, zscore_threshold: float = 2.0, epsilon: float = 1e-12) -> CrossSectionalResult:
    """Rank each date's available residual returns within its peer universe."""
    if minimum_universe_size < 2 or zscore_threshold <= 0:
        raise ValueError("minimum_universe_size must be at least 2 and zscore_threshold positive")
    if not isinstance(residual_returns.index, pd.DatetimeIndex) or not residual_returns.index.is_unique or not residual_returns.index.is_monotonic_increasing:
        raise ValueError("residual returns must use a unique, ascending DatetimeIndex")
    values = residual_returns.apply(pd.to_numeric, errors="raise").copy(deep=True)
    columns = values.columns
    ranks = pd.DataFrame(np.nan, index=values.index, columns=columns)
    percentiles = ranks.copy()
    zscores = ranks.copy()
    states = pd.DataFrame(CrossSectionalState.INVALID.value, index=values.index, columns=columns)
    means, standard_deviations = pd.Series(np.nan, index=values.index), pd.Series(np.nan, index=values.index)
    valid = pd.Series(False, index=values.index)
    for date, row in values.iterrows():
        available = row.dropna()
        if len(available) < minimum_universe_size:
            continue
        mean, std = float(available.mean()), float(available.std(ddof=1))
        if not np.isfinite(std) or std <= epsilon:
            continue
        z = (available - mean) / std
        means.loc[date], standard_deviations.loc[date], valid.loc[date] = mean, std, True
        zscores.loc[date, available.index] = z
        ranks.loc[date, available.index] = available.rank(method="average", ascending=True)
        percentiles.loc[date, available.index] = available.rank(method="average", pct=True, ascending=True)
        states.loc[date, available.index] = np.where(z >= zscore_threshold, CrossSectionalState.RELATIVELY_HIGH.value, np.where(z <= -zscore_threshold, CrossSectionalState.RELATIVELY_LOW.value, CrossSectionalState.NEUTRAL.value))
    return CrossSectionalResult(values.index, values, means, standard_deviations, ranks, percentiles, zscores, states, valid, minimum_universe_size)
