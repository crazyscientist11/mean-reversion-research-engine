import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from src.models.kalman_pair import kalman_pair_filter
from src.models.ou_pair_optimizer import optimize_ou_hedge_ratio

def test_future_data_cannot_change_kalman_or_cutoff_optimizer() -> None:
    rng=np.random.default_rng(16); b=4+np.cumsum(rng.normal(0,.01,180)); a=.2+1.3*b+rng.normal(0,.01,180); prices=pd.DataFrame({"A":np.exp(a),"B":np.exp(b)},index=pd.date_range("2020-01-01",periods=180)); cutoff=prices.index[130]; changed=prices.copy(); changed.loc[changed.index>cutoff,"A"]*=5
    first=kalman_pair_filter(prices,target_ticker="A",peer_ticker="B"); second=kalman_pair_filter(changed,target_ticker="A",peer_ticker="B")
    assert_series_equal(first.beta.loc[:cutoff],second.beta.loc[:cutoff])
    one=optimize_ou_hedge_ratio(prices,target_ticker="A",peer_ticker="B",beta_min=.5,beta_max=2,grid_size=21,training_end=cutoff); two=optimize_ou_hedge_ratio(changed,target_ticker="A",peer_ticker="B",beta_min=.5,beta_max=2,grid_size=21,training_end=cutoff)
    assert one.optimal_beta==two.optimal_beta
