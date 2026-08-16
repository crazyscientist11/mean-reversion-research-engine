"""The frozen, explainable top-level research decision contract."""
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from src.confidence import ConfidenceResult, calculate_confidence
from src.consensus import ConsensusResult, ConsensusState
from .explain import ExplainStatus, WhySignalItem, WhySignalResult, make_explanation
from .states import PositionState, ResearchAction


@dataclass(frozen=True)
class CriticalGates:
    sufficient_data: bool = True
    residual_valid: bool = True
    stationarity_valid: bool = True
    ou_valid: bool = True
    half_life_acceptable: bool = True
    parameter_stable: bool = True
    regime_status: str = "NORMAL"
    def failures(self) -> tuple[str, ...]:
        result = []
        for name, passed in (("Insufficient data", self.sufficient_data), ("Invalid residual", self.residual_valid), ("Stationarity gate", self.stationarity_valid), ("OU fit", self.ou_valid), ("Half-life", self.half_life_acceptable), ("Parameter stability", self.parameter_stable)):
            if not passed: result.append(name)
        if self.regime_status == "SEVERE_BREAK": result.append("SEVERE_BREAK regime")
        return tuple(result)


@dataclass(frozen=True)
class FinalResearchDecision:
    decision_id: str
    as_of: datetime
    model_cutoff: datetime
    target_expression: str
    position_state: PositionState
    current_prices: dict[str, float]
    current_residual: float | None
    current_zscore: float | None
    ou_theta: float | None
    ou_kappa: float | None
    ou_sigma: float | None
    ou_half_life: float | None
    long_entry_region: tuple[float, float] | None
    short_entry_region: tuple[float, float] | None
    exit_region: tuple[float, float] | None
    stop_region: tuple[float, float] | None
    research_action: ResearchAction
    p_exit_before_stop: float | None = None
    p_stop_before_exit: float | None = None
    p_exit_5d: float | None = None
    p_exit_10d: float | None = None
    p_exit_20d: float | None = None
    median_holding_time: float | None = None
    expected_economic_value: float | None = None
    transaction_cost_assumptions: dict[str, float] = field(default_factory=dict)
    opportunity_cost_assumption: float | None = None
    stationarity_classification: str = "NOT_ASSESSED"
    parameter_stability: str = "NOT_ASSESSED"
    regime_status: str = "NORMAL"
    consensus: dict[str, Any] = field(default_factory=dict)
    agreement_percentage: float = 0.0
    confidence: dict[str, Any] = field(default_factory=dict)
    why_signal: dict[str, Any] = field(default_factory=dict)
    valid: bool = False
    invalid_reason: str | None = None
    parameter_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["as_of"] = self.as_of.isoformat(); result["model_cutoff"] = self.model_cutoff.isoformat()
        result["position_state"] = self.position_state.value; result["research_action"] = self.research_action.value
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FinalResearchDecision":
        record = value.copy()
        record["as_of"] = datetime.fromisoformat(record["as_of"]); record["model_cutoff"] = datetime.fromisoformat(record["model_cutoff"])
        record["position_state"] = PositionState(record["position_state"]); record["research_action"] = ResearchAction(record["research_action"])
        for name in ("long_entry_region", "short_entry_region", "exit_region", "stop_region"):
            if record.get(name) is not None: record[name] = tuple(record[name])
        return cls(**record)

    def snapshot_outputs(self) -> dict[str, dict[str, Any]]:
        """Frozen values ready for the existing PredictionSnapshot output fields."""
        return {"model_outputs": {"final_decision": self.to_dict()}, "ou_outputs": {"theta": self.ou_theta, "kappa": self.ou_kappa, "sigma": self.ou_sigma, "half_life": self.ou_half_life}, "probability_outputs": {"exit_before_stop": self.p_exit_before_stop, "stop_before_exit": self.p_stop_before_exit, "exit_5d": self.p_exit_5d, "exit_10d": self.p_exit_10d, "exit_20d": self.p_exit_20d}, "diagnostic_outputs": {"why_signal": self.why_signal, "parameter_hashes": self.parameter_hashes}, "consensus_outputs": {"consensus": self.consensus, "agreement_percentage": self.agreement_percentage, "confidence": self.confidence}}


def build_final_decision(*, decision_id: str, as_of: datetime, model_cutoff: datetime, target_expression: str, position_state: PositionState, current_prices: dict[str, float], current_residual: float | None, current_zscore: float | None, consensus: ConsensusResult, gates: CriticalGates, policy_action: ResearchAction = ResearchAction.WAIT, ou_theta: float | None = None, ou_kappa: float | None = None, ou_sigma: float | None = None, ou_half_life: float | None = None, long_entry_region: tuple[float, float] | None = None, short_entry_region: tuple[float, float] | None = None, exit_region: tuple[float, float] | None = None, stop_region: tuple[float, float] | None = None, probability_outputs: dict[str, float | None] | None = None, expected_economic_value: float | None = None, transaction_cost_assumptions: dict[str, float] | None = None, opportunity_cost_assumption: float | None = None, stationarity_classification: str = "NOT_ASSESSED", parameter_stability: str = "NOT_ASSESSED", confidence_qualities: dict[str, float] | None = None, parameter_hashes: dict[str, str] | None = None) -> FinalResearchDecision:
    """Apply non-negotiable gates before consensus, confidence, or policy actions."""
    failures = gates.failures()
    if not gates.sufficient_data:
        action, valid, reason = ResearchAction.INSUFFICIENT_DATA, False, "Insufficient data"
    elif failures:
        action, valid, reason = ResearchAction.NO_SIGNAL, False, "; ".join(failures)
    elif consensus.state is ConsensusState.INSUFFICIENT_DATA:
        action, valid, reason = ResearchAction.INSUFFICIENT_DATA, False, "No valid residual models"
    elif consensus.state is ConsensusState.CONFLICTED:
        action, valid, reason = ResearchAction.CONFLICTED, False, "Valid models have strong long/short disagreement"
    else:
        action, valid, reason = policy_action, True, None
    qualities = confidence_qualities or {}
    confidence = calculate_confidence(critical_gates_pass=not failures, zscore=current_zscore, model_agreement=consensus.agreement_percentage / 100.0, **qualities)
    status = ExplainStatus.PASS if valid else ExplainStatus.FAIL
    explanations = [WhySignalItem("Critical gates", "passed" if not failures else "; ".join(failures), "all critical gates must pass", ExplainStatus.PASS if not failures else ExplainStatus.FAIL, "Hard gates are evaluated before confidence and policy."), WhySignalItem("Model consensus", consensus.state.value, f"{consensus.valid_model_count} valid models", status if consensus.state not in (ConsensusState.CONFLICTED, ConsensusState.INSUFFICIENT_DATA) else ExplainStatus.FAIL, "Consensus is dependency-aware; related models do not receive full independent vote weight.")]
    why = make_explanation(explanations, action.value)
    probability = probability_outputs or {}
    return FinalResearchDecision(decision_id, as_of, model_cutoff, target_expression, position_state, dict(current_prices), current_residual, current_zscore, ou_theta, ou_kappa, ou_sigma, ou_half_life, long_entry_region, short_entry_region, exit_region, stop_region, action, probability.get("exit_before_stop"), probability.get("stop_before_exit"), probability.get("exit_5d"), probability.get("exit_10d"), probability.get("exit_20d"), probability.get("median_holding_time"), expected_economic_value, transaction_cost_assumptions or {}, opportunity_cost_assumption, stationarity_classification, parameter_stability, gates.regime_status, consensus.to_dict(), consensus.agreement_percentage, confidence.to_dict(), why.to_dict(), valid, reason, parameter_hashes or {})
