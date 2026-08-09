from src.costs import CostModel
from src.decision import StoppingConfig,build_policy,PositionState,DecisionGates,evaluate_policy,ResearchAction
from src.decision.trade_expression import PairTradeExpression
from src.decision import benchmark_action
def test_gates_and_costs_prevent_entry():
    p=build_policy(StoppingConfig(theta=0,kappa=.2,sigma=.2,grid_min=-2,grid_max=2,long_stop=-1.8,short_stop=1.8),CostModel(fixed_entry_cost=100))
    assert evaluate_policy(p,-1,PositionState.FLAT,DecisionGates(ou_valid=False)) is ResearchAction.NO_SIGNAL
    assert evaluate_policy(p,-1,PositionState.FLAT,DecisionGates(ou_valid=True,sufficient_data=False)) is ResearchAction.INSUFFICIENT_DATA
def test_pair_expression_marks_two_legs():
    e=PairTradeExpression("A","B",1.5,100,50,10000); assert len(e.shares())==2 and e.marked_pnl(101,49)!=0
def test_benchmark_state_machine_actions():
    assert benchmark_action(-2.1,PositionState.FLAT) is ResearchAction.ENTER_LONG
    assert benchmark_action(0,PositionState.LONG_RESIDUAL) is ResearchAction.EXIT_LONG
    assert benchmark_action(-4.1,PositionState.LONG_RESIDUAL) is ResearchAction.STOP_LONG

def test_policy_reports_runner_up_and_never_exits_while_flat():
    policy=build_policy(StoppingConfig(theta=0,kappa=.2,sigma=.2,grid_min=-2,grid_max=2,long_stop=-1.8,short_stop=1.8),CostModel())
    action,value,next_action,next_value=policy.action_at(-1,PositionState.FLAT)
    assert action in {ResearchAction.WAIT,ResearchAction.ENTER_LONG,ResearchAction.ENTER_SHORT}
    assert next_action in {ResearchAction.WAIT,ResearchAction.ENTER_LONG,ResearchAction.ENTER_SHORT}
    assert value>=next_value

def test_stopped_long_state_has_stop_action():
    policy=build_policy(StoppingConfig(theta=0,kappa=.2,sigma=.2,grid_min=-2,grid_max=2,long_stop=-1.5,short_stop=1.5),CostModel())
    assert policy.action_at(-1.8,PositionState.LONG_RESIDUAL)[0] is ResearchAction.STOP_LONG
