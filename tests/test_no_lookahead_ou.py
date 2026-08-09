import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.models.ou import rolling_ou_fits, simulate_ou


def test_future_values_do_not_change_rolling_ou_outputs_through_cutoff() -> None:
    values = pd.Series(simulate_ou(theta=0.2, kappa=0.12, sigma=0.1, n_observations=180, seed=22), index=pd.date_range("2020-01-01", periods=180))
    cutoff = values.index[130]
    changed = values.copy()
    changed.loc[changed.index > cutoff] += 5.0
    original = rolling_ou_fits(values, method="AR1", window=60)
    altered = rolling_ou_fits(changed, method="AR1", window=60)
    assert_frame_equal(original.loc[:cutoff], altered.loc[:cutoff])
