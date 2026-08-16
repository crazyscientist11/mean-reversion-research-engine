from src.confidence import calculate_confidence

def test_confidence_is_deterministic_and_capped():
    first = calculate_confidence(critical_gates_pass=True, zscore=2.5, stationarity_quality=.5)
    second = calculate_confidence(critical_gates_pass=True, zscore=100, stationarity_quality=.5)
    assert first.score == calculate_confidence(critical_gates_pass=True, zscore=2.5, stationarity_quality=.5).score
    assert first.components[0].earned == second.components[0].earned == 12

def test_confidence_cannot_override_a_failed_gate():
    result = calculate_confidence(critical_gates_pass=False, zscore=10, stationarity_quality=1)
    assert result.score is None
