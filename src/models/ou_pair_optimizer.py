"""Bounded in-sample static hedge-ratio search using exact OU likelihood."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .ou import fit_ou_mle, OUFitResult

@dataclass(frozen=True)
class OUPairOptimizationResult:
    optimal_beta: float|None; objective: float|None; ou_fit: OUFitResult|None; candidates: pd.DataFrame; weakly_identified: bool; invalid_reason: str|None

def optimize_ou_hedge_ratio(prices: pd.DataFrame, *, target_ticker: str, peer_ticker: str, beta_min: float=-3.0, beta_max: float=3.0, grid_size: int=61, minimum_observations: int=40, flat_objective_tolerance: float=1e-3, training_end: pd.Timestamp|None=None) -> OUPairOptimizationResult:
    if target_ticker==peer_ticker or target_ticker not in prices or peer_ticker not in prices: raise ValueError("distinct target and peer tickers are required")
    if not beta_min<beta_max or not 3<=grid_size<=501: raise ValueError("invalid beta bounds or grid size")
    logs=np.log(prices.loc[:training_end,[target_ticker,peer_ticker]].dropna());
    if len(logs)<minimum_observations: return OUPairOptimizationResult(None,None,None,pd.DataFrame(),False,"Insufficient historical observations.")
    rows=[]
    for beta in np.linspace(beta_min,beta_max,grid_size):
        fit=fit_ou_mle(logs[target_ticker]-beta*logs[peer_ticker])
        objective=fit.log_likelihood/(fit.n_observations-1) if fit.valid and fit.log_likelihood is not None else np.nan
        rows.append({"beta":beta,"average_conditional_log_likelihood":objective,"theta":fit.theta,"kappa":fit.kappa,"sigma":fit.sigma,"half_life":fit.half_life,"valid":fit.valid,"fit":fit})
    table=pd.DataFrame(rows); usable=table[table.valid & table.average_conditional_log_likelihood.notna()]
    if usable.empty: return OUPairOptimizationResult(None,None,None,table,False,"No valid OU fit in hedge-ratio grid.")
    best=usable.loc[usable.average_conditional_log_likelihood.idxmax()]; ordered=usable.average_conditional_log_likelihood.nlargest(2).to_numpy(); weak=len(ordered)>1 and abs(ordered[0]-ordered[1])<=flat_objective_tolerance
    return OUPairOptimizationResult(float(best.beta),float(best.average_conditional_log_likelihood),best.fit,table.drop(columns="fit"),weak,None)
