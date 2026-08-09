"""Seeded exact-transition OU first-passage simulation; no parameter fitting occurs here."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .first_passage import BoundaryDefinition, ResidualDirection

@dataclass
class FirstPassageResult:
    exit_before_stop: float; stop_before_exit: float; neither_before_horizon: float
    exit_within_5: float; exit_within_10: float; exit_within_20: float
    monte_carlo_se_exit: float; median_exit_time: float|None; mean_exit_time: float|None
    maximum_favorable_excursion: np.ndarray; maximum_adverse_excursion: np.ndarray; sample_paths: np.ndarray
    exit_times: np.ndarray; stop_times: np.ndarray; metadata: dict

def simulate_ou_first_passage(*, theta: float, kappa: float, sigma: float, boundaries: BoundaryDefinition, dt: float=1.0, number_paths: int=10000, maximum_horizon: int=20, seed: int=0, substeps: int=1, sample_path_count: int=100) -> FirstPassageResult:
    boundaries.validate()
    if kappa<=0 or sigma<0 or dt<=0 or number_paths<1 or maximum_horizon<1 or substeps<1: raise ValueError("invalid OU simulation settings")
    rng=np.random.default_rng(seed); step_dt=dt/substeps; decay=np.exp(-kappa*step_dt); step_std=np.sqrt(sigma**2*(-np.expm1(-2*kappa*step_dt))/(2*kappa))
    x=np.full(number_paths,boundaries.current_state,float); start=x.copy(); hit=np.zeros(number_paths,int); hit_time=np.full(number_paths,np.nan); exit_times=np.full(number_paths,np.nan); stop_times=np.full(number_paths,np.nan); max_x=x.copy(); min_x=x.copy(); samples=[x[:min(sample_path_count,number_paths)].copy()]
    for step in range(1,maximum_horizon*substeps+1):
        prior=x.copy(); active=hit==0; x[active]=theta+(x[active]-theta)*decay+rng.normal(0,step_std,active.sum()); max_x=np.maximum(max_x,x); min_x=np.minimum(min_x,x)
        if boundaries.direction is ResidualDirection.LONG_RESIDUAL:
            exit_cross=(prior<boundaries.exit_boundary)&(x>=boundaries.exit_boundary); stop_cross=(prior>boundaries.stop_boundary)&(x<=boundaries.stop_boundary)
        else:
            exit_cross=(prior>boundaries.exit_boundary)&(x<=boundaries.exit_boundary); stop_cross=(prior<boundaries.stop_boundary)&(x>=boundaries.stop_boundary)
        both=active&exit_cross&stop_cross; exit_only=active&exit_cross&~stop_cross; stop_only=active&stop_cross&~exit_cross
        # If a coarse step spans both boundaries, linear crossing fractions decide first; exact ties choose stop conservatively.
        if both.any():
            exit_fraction=np.abs((boundaries.exit_boundary-prior[both])/(x[both]-prior[both])); stop_fraction=np.abs((boundaries.stop_boundary-prior[both])/(x[both]-prior[both])); choose_exit=exit_fraction<stop_fraction; indices=np.where(both)[0]; exit_only[indices[choose_exit]]=True; stop_only[indices[~choose_exit]]=True
        now=step*step_dt; hit[exit_only]=1; hit[stop_only]=2; hit_time[exit_only|stop_only]=now; exit_times[exit_only]=now; stop_times[stop_only]=now
        if step%substeps==0: samples.append(x[:min(sample_path_count,number_paths)].copy())
    exit_probability=float(np.mean(hit==1)); stop_probability=float(np.mean(hit==2)); exits=exit_times[~np.isnan(exit_times)]
    favorable=max_x-start if boundaries.direction is ResidualDirection.LONG_RESIDUAL else start-min_x; adverse=start-min_x if boundaries.direction is ResidualDirection.LONG_RESIDUAL else max_x-start
    within=lambda h: float(np.mean((hit==1)&(exit_times<=h)))
    return FirstPassageResult(exit_probability,stop_probability,float(np.mean(hit==0)),within(5),within(10),within(20),float(np.sqrt(exit_probability*(1-exit_probability)/number_paths)),float(np.median(exits)) if len(exits) else None,float(np.mean(exits)) if len(exits) else None,favorable,adverse,np.asarray(samples).T,exit_times,stop_times,{"seed":seed,"substeps":substeps,"dt":dt,"first_crossing":"linear interpolation; ties assigned to stop"})

def evaluate_boundary_candidates(*, theta:float,kappa:float,sigma:float,candidates:list[BoundaryDefinition], **kwargs) -> pd.DataFrame:
    rows=[]
    for index,candidate in enumerate(candidates):
        result=simulate_ou_first_passage(theta=theta,kappa=kappa,sigma=sigma,boundaries=candidate,seed=int(kwargs.pop("seed",0))+index,**kwargs)
        rows.append({"exit":candidate.exit_boundary,"stop":candidate.stop_boundary,"exit_first_probability":result.exit_before_stop,"stop_first_probability":result.stop_before_exit,"median_holding_time":result.median_exit_time,"mean_favorable_excursion":float(np.mean(result.maximum_favorable_excursion)),"mean_adverse_excursion":float(np.mean(result.maximum_adverse_excursion)),"monte_carlo_se_exit":result.monte_carlo_se_exit})
    return pd.DataFrame(rows)
