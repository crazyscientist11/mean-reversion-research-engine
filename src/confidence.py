"""Transparent, capped research-confidence calculation; never a gate override."""
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ConfidenceComponent:
    name: str
    earned: float
    maximum: float
    detail: str

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True)
class ConfidenceResult:
    score: float | None
    components: tuple[ConfidenceComponent, ...]
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "unavailable_reason": self.unavailable_reason, "components": [c.to_dict() for c in self.components]}


def calculate_confidence(*, critical_gates_pass: bool, zscore: float | None, stationarity_quality: float = 0.0, ou_agreement: float = 0.0, half_life_quality: float = 0.0, parameter_stability: float = 0.0, regime_quality: float = 0.0, model_agreement: float = 0.0, first_passage_quality: float = 0.0, economic_value_margin: float = 0.0, boundary_robustness: float = 0.0) -> ConfidenceResult:
    """Return a 0--100 score after gates pass. Quality inputs are normalized 0--1."""
    if not critical_gates_pass:
        return ConfidenceResult(None, (), "Critical gates failed; confidence is intentionally not calculated.")
    def quality(value: float) -> float: return min(1.0, max(0.0, float(value)))
    deviation = 0.0 if zscore is None else min(abs(float(zscore)) / 2.5, 1.0)
    specifications = (
        ("Deviation magnitude", deviation, 12.0, "Capped at |z| = 2.5 so extreme deviations cannot dominate."),
        ("Stationarity", stationarity_quality, 14.0, "Quality of the stationarity diagnostics."),
        ("OU estimator agreement", ou_agreement, 15.0, "Agreement between usable OU estimators."),
        ("Half-life", half_life_quality, 12.0, "Fit against the configured acceptable holding-time range."),
        ("Parameter stability", parameter_stability, 14.0, "Rolling-parameter stability diagnostic."),
        ("Regime", regime_quality, 10.0, "Current regime diagnostic."),
        ("Model agreement", model_agreement, 13.0, "Dependency-aware consensus agreement."),
        ("First-passage", first_passage_quality, 5.0, "Boundary-hit probability quality."),
        ("Economic value", economic_value_margin, 3.0, "Expected value after stated costs."),
        ("Boundary robustness", boundary_robustness, 2.0, "Sensitivity of the stopping boundary decision."),
    )
    components = tuple(ConfidenceComponent(name, quality(value) * maximum, maximum, detail) for name, value, maximum, detail in specifications)
    return ConfidenceResult(round(sum(component.earned for component in components), 6), components)
