import numpy as np
from src.first_passage import BoundaryDefinition, ResidualDirection
from src.monte_carlo import simulate_ou_first_passage

def test_seed_reproducibility_and_deterministic_exit() -> None:
    b=BoundaryDefinition(0.,1.,-1.,ResidualDirection.LONG_RESIDUAL)
    one=simulate_ou_first_passage(theta=2,kappa=1,sigma=0,boundaries=b,number_paths=30,maximum_horizon=5,seed=1)
    two=simulate_ou_first_passage(theta=2,kappa=1,sigma=0,boundaries=b,number_paths=30,maximum_horizon=5,seed=1)
    assert one.exit_before_stop==1 and np.array_equal(one.sample_paths,two.sample_paths)

def test_closer_exit_has_higher_exit_probability() -> None:
    common=dict(theta=1.0,kappa=.2,sigma=.3,number_paths=4000,maximum_horizon=20,seed=4)
    close=simulate_ou_first_passage(boundaries=BoundaryDefinition(0,.3,-1,ResidualDirection.LONG_RESIDUAL),**common)
    far=simulate_ou_first_passage(boundaries=BoundaryDefinition(0,.9,-1,ResidualDirection.LONG_RESIDUAL),**common)
    assert close.exit_before_stop>far.exit_before_stop
