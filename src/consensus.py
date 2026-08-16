"""Dependency-aware aggregation of residual-model research evidence."""
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


class EvidenceDirection(str, Enum):
    LONG_RESIDUAL = "LONG_RESIDUAL"
    SHORT_RESIDUAL = "SHORT_RESIDUAL"
    NEUTRAL = "NEUTRAL"
    INVALID = "INVALID"


class ConsensusState(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    CONFLICTED = "CONFLICTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class ModelEvidence:
    model_name: str
    direction: EvidenceDirection
    valid: bool
    residual_zscore: float | None = None
    target_state: str | None = None
    ou_valid: bool | None = None
    gate_passed: bool = True
    weight: float = 1.0
    reliability: float = 1.0
    dependency_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name.strip() or self.weight < 0 or self.reliability < 0:
            raise ValueError("model name and non-negative weight/reliability are required")

    @property
    def contributes(self) -> bool:
        return self.valid and self.gate_passed and self.direction is not EvidenceDirection.INVALID


@dataclass(frozen=True)
class ConsensusResult:
    state: ConsensusState
    valid_model_count: int
    long_count: int
    short_count: int
    neutral_count: int
    agreement_percentage: float
    disagreeing_models: tuple[str, ...]
    evidences: tuple[ModelEvidence, ...]
    weighted_support: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        result["evidences"] = [{**asdict(e), "direction": e.direction.value} for e in self.evidences]
        return result


def build_consensus(evidences: Iterable[ModelEvidence], *, conflict_share: float = 0.35) -> ConsensusResult:
    """Aggregate evidence without counting models in one dependency group as independent votes.

    Each dependency group receives at most one unit of total effective weight, split
    among its members by their weight times reliability.
    """
    if not 0 < conflict_share < 0.5:
        raise ValueError("conflict_share must be between 0 and 0.5")
    items = tuple(evidences)
    valid = [item for item in items if item.contributes]
    counts = {direction: sum(item.direction is direction for item in valid) for direction in EvidenceDirection}
    if not valid:
        return ConsensusResult(ConsensusState.INSUFFICIENT_DATA, 0, 0, 0, 0, 0.0, (), items, {})

    groups: dict[str, list[ModelEvidence]] = {}
    for index, item in enumerate(valid):
        groups.setdefault(item.dependency_group or f"__independent_{index}", []).append(item)
    support = {direction.value: 0.0 for direction in (EvidenceDirection.LONG_RESIDUAL, EvidenceDirection.SHORT_RESIDUAL, EvidenceDirection.NEUTRAL)}
    for group in groups.values():
        raw = [item.weight * item.reliability for item in group]
        total = sum(raw)
        if total <= 0:
            continue
        # A group has the average base weight of its members, not their sum.
        group_weight = sum(item.weight for item in group) / len(group)
        for item, value in zip(group, raw):
            support[item.direction.value] += group_weight * value / total
    total_support = sum(support.values())
    if total_support <= 0:
        return ConsensusResult(ConsensusState.INSUFFICIENT_DATA, len(valid), counts[EvidenceDirection.LONG_RESIDUAL], counts[EvidenceDirection.SHORT_RESIDUAL], counts[EvidenceDirection.NEUTRAL], 0.0, (), items, support)
    long_share = support[EvidenceDirection.LONG_RESIDUAL.value] / total_support
    short_share = support[EvidenceDirection.SHORT_RESIDUAL.value] / total_support
    winner = max(support, key=support.get)
    if long_share >= conflict_share and short_share >= conflict_share:
        state = ConsensusState.CONFLICTED
        disagreeing = tuple(item.model_name for item in valid if item.direction is not EvidenceDirection.NEUTRAL)
        agreement = max(long_share, short_share) * 100.0
    else:
        state = {"LONG_RESIDUAL": ConsensusState.LONG, "SHORT_RESIDUAL": ConsensusState.SHORT, "NEUTRAL": ConsensusState.NEUTRAL}[winner]
        winner_direction = EvidenceDirection(winner)
        disagreeing = tuple(item.model_name for item in valid if item.direction is not winner_direction and item.direction is not EvidenceDirection.NEUTRAL)
        agreement = support[winner] / total_support * 100.0
    return ConsensusResult(state, len(valid), counts[EvidenceDirection.LONG_RESIDUAL], counts[EvidenceDirection.SHORT_RESIDUAL], counts[EvidenceDirection.NEUTRAL], agreement, disagreeing, items, support)
