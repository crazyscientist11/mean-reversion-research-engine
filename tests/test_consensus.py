from src.consensus import ConsensusState, EvidenceDirection, ModelEvidence, build_consensus

def evidence(name, direction, **kwargs): return ModelEvidence(name, direction, True, **kwargs)

def test_zero_valid_models_is_insufficient_data():
    result = build_consensus([ModelEvidence("bad", EvidenceDirection.INVALID, False)])
    assert result.state is ConsensusState.INSUFFICIENT_DATA

def test_strong_long_short_disagreement_is_conflicted():
    result = build_consensus([evidence("a", EvidenceDirection.LONG_RESIDUAL), evidence("b", EvidenceDirection.SHORT_RESIDUAL)])
    assert result.state is ConsensusState.CONFLICTED

def test_related_models_do_not_receive_full_independent_weight():
    result = build_consensus([evidence("a", EvidenceDirection.LONG_RESIDUAL, dependency_group="price"), evidence("b", EvidenceDirection.LONG_RESIDUAL, dependency_group="price"), evidence("pair", EvidenceDirection.SHORT_RESIDUAL)])
    assert result.weighted_support["LONG_RESIDUAL"] == result.weighted_support["SHORT_RESIDUAL"]
