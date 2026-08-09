import numpy as np
import pandas as pd
from src.models.ou import simulate_ou
from src.models.ou_pair_optimizer import optimize_ou_hedge_ratio

def test_ou_likelihood_optimizer_finds_known_beta_neighborhood() -> None:
    rng=np.random.default_rng(7); n=400; peer=4+np.cumsum(rng.normal(0,.01,n)); spread=simulate_ou(theta=0,kappa=.15,sigma=.04,n_observations=n,seed=8); target=.3+1.4*peer+spread
    prices=pd.DataFrame({"A":np.exp(target),"B":np.exp(peer)},index=pd.date_range("2020-01-01",periods=n))
    result=optimize_ou_hedge_ratio(prices,target_ticker="A",peer_ticker="B",beta_min=.8,beta_max=2,grid_size=61)
    assert result.optimal_beta is not None and abs(result.optimal_beta-1.4)<=.08
    assert result.ou_fit is not None and result.ou_fit.valid
