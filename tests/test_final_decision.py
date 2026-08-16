from datetime import datetime, timezone
import pytest
from src.consensus import EvidenceDirection, ModelEvidence, build_consensus
from src.decision import CriticalGates, PositionState, ResearchAction, build_final_decision
from src.prediction.schemas import PredictionSnapshot, parameter_hash
from src.config import Frequency

def decision(gates=CriticalGates(), evidence=None):
    consensus = build_consensus(evidence or [ModelEvidence("pair", EvidenceDirection.LONG_RESIDUAL, True)])
    return build_final_decision(decision_id="d1", as_of=datetime(2025,1,1,tzinfo=timezone.utc), model_cutoff=datetime(2025,1,1,tzinfo=timezone.utc), target_expression="AAA/BBB", position_state=PositionState.FLAT, current_prices={"AAA":100}, current_residual=-.2, current_zscore=-2.5, consensus=consensus, gates=gates, policy_action=ResearchAction.ENTER_LONG)

def test_failed_hard_gate_and_severe_break_are_no_signal():
    assert decision(CriticalGates(stationarity_valid=False)).research_action is ResearchAction.NO_SIGNAL
    assert decision(CriticalGates(regime_status="SEVERE_BREAK")).research_action is ResearchAction.NO_SIGNAL

def test_conflict_and_zero_models_have_required_actions():
    assert decision(evidence=[ModelEvidence("a", EvidenceDirection.LONG_RESIDUAL, True), ModelEvidence("b", EvidenceDirection.SHORT_RESIDUAL, True)]).research_action is ResearchAction.CONFLICTED
    assert decision(evidence=[ModelEvidence("bad", EvidenceDirection.INVALID, False)]).research_action is ResearchAction.INSUFFICIENT_DATA

def test_final_decision_serialization_and_snapshot_are_immutable():
    original = decision(); assert type(original).from_dict(original.to_dict()) == original
    now = original.as_of
    snapshot = PredictionSnapshot("p", now, now, "AAA", Frequency.DAILY, 100, "1", "test", parameter_hash({"x": 1}), **original.snapshot_outputs())
    with pytest.raises(TypeError): snapshot.model_outputs["final_decision"]["research_action"] = "WAIT"
    assert snapshot.model_outputs["final_decision"]["research_action"] == "ENTER_LONG"
