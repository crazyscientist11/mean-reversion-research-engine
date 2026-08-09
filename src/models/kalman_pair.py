"""Two-state linear Kalman filter for dynamic log-price pair relationships."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class KalmanPairResult:
    dates: pd.DatetimeIndex; prior_alpha: pd.Series; prior_beta: pd.Series; alpha: pd.Series; beta: pd.Series
    predicted_log_target: pd.Series; innovation: pd.Series; innovation_variance: pd.Series; standardized_innovation: pd.Series; spread_state: pd.Series; valid: pd.Series

def kalman_pair_filter(prices: pd.DataFrame, *, target_ticker: str, peer_ticker: str, process_noise_alpha: float=1e-5, process_noise_beta: float=1e-5, observation_noise: float=1e-4, initial_alpha: float=0.0, initial_beta: float=1.0, initial_covariance: float=1.0) -> KalmanPairResult:
    if target_ticker == peer_ticker or target_ticker not in prices or peer_ticker not in prices: raise ValueError("distinct target and peer tickers are required")
    if min(process_noise_alpha, process_noise_beta, observation_noise, initial_covariance) <= 0: raise ValueError("noise and covariance settings must be positive")
    frame=prices[[target_ticker,peer_ticker]].copy();
    if not isinstance(frame.index,pd.DatetimeIndex) or not frame.index.is_monotonic_increasing or not frame.index.is_unique: raise ValueError("prices need unique ascending dates")
    if (frame.dropna()<=0).any().any(): raise ValueError("prices must be positive")
    logs=np.log(frame); idx=frame.index; cols={n:pd.Series(np.nan,index=idx) for n in ["prior_alpha","prior_beta","alpha","beta","predicted","innovation","innovation_variance","standardized","spread"]}; valid=pd.Series(False,index=idx)
    state=np.array([initial_alpha,initial_beta],float); cov=np.eye(2)*initial_covariance; q=np.diag([process_noise_alpha,process_noise_beta])
    for date,row in logs.iterrows():
        if row.isna().any(): continue
        prior=state.copy(); prior_cov=cov+q; h=np.array([1.0,float(row[peer_ticker])]); prediction=float(h@prior); innovation=float(row[target_ticker]-prediction); variance=float(h@prior_cov@h+observation_noise)
        if variance<=0 or not np.isfinite(variance): continue
        gain=(prior_cov@h)/variance; state=prior+gain*innovation; cov=(np.eye(2)-np.outer(gain,h))@prior_cov
        for n,v in zip(["prior_alpha","prior_beta"],prior): cols[n].loc[date]=v
        for n,v in zip(["alpha","beta"],state): cols[n].loc[date]=v
        cols["predicted"].loc[date]=prediction; cols["innovation"].loc[date]=innovation; cols["innovation_variance"].loc[date]=variance; cols["standardized"].loc[date]=innovation/np.sqrt(variance); cols["spread"].loc[date]=float(row[target_ticker]-state[0]-state[1]*row[peer_ticker]); valid.loc[date]=True
    return KalmanPairResult(idx,cols["prior_alpha"],cols["prior_beta"],cols["alpha"],cols["beta"],cols["predicted"],cols["innovation"],cols["innovation_variance"],cols["standardized"],cols["spread"],valid)
