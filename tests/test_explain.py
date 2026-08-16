from src.decision.explain import ExplainStatus, WhySignalItem, make_explanation

def test_explanation_identifies_failing_gate():
    why = make_explanation([WhySignalItem("OU validity", "invalid", "valid fit", ExplainStatus.FAIL, "Fit failed.")], "NO_SIGNAL")
    assert why.blocking_reasons == ("OU validity",)
