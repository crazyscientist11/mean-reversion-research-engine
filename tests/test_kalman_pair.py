import numpy as np
import pandas as pd
from src.models.kalman_pair import kalman_pair_filter

def test_kalman_beta_tracks_slowly_changing_relationship() -> None:
    rng=np.random.default_rng(6); n=240; peer=4+np.cumsum(rng.normal(0,.015,n)); true_beta=1+np.linspace(0,.5,n); target=.2+true_beta*peer+rng.normal(0,.01,n)
    prices=pd.DataFrame({"A":np.exp(target),"B":np.exp(peer)},index=pd.date_range("2020-01-01",periods=n))
    result=kalman_pair_filter(prices,target_ticker="A",peer_ticker="B",process_noise_beta=1e-4,observation_noise=1e-4)
    assert np.mean((result.beta.to_numpy()-true_beta)**2) < np.mean((1.0-true_beta)**2)
    assert result.valid.all()
