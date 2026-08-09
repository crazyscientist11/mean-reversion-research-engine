from dataclasses import dataclass
from .states import PositionState,ResearchAction
from .optimal_stopping import StoppingPolicy
@dataclass(frozen=True)
class DecisionGates:
    ou_valid:bool; stationarity_valid:bool=True; regime_valid:bool=True; sufficient_data:bool=True
def evaluate_policy(policy:StoppingPolicy,current_state:float,position:PositionState,gates:DecisionGates)->ResearchAction:
    if not gates.sufficient_data:return ResearchAction.INSUFFICIENT_DATA
    if not(gates.ou_valid and gates.stationarity_valid and gates.regime_valid):return ResearchAction.NO_SIGNAL
    return policy.action_at(current_state,position)[0]
