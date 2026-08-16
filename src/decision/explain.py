"""Structured, plain-English explanations for final research actions."""
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable


class ExplainStatus(str, Enum): PASS = "PASS"; CAUTION = "CAUTION"; FAIL = "FAIL"; NOT_APPLICABLE = "NOT_APPLICABLE"

@dataclass(frozen=True)
class WhySignalItem:
    name: str
    measured_value: str
    reference: str
    status: ExplainStatus
    explanation: str

@dataclass(frozen=True)
class WhySignalResult:
    items: tuple[WhySignalItem, ...]
    final_action: str
    blocking_reasons: tuple[str, ...] = ()
    def to_dict(self) -> dict[str, Any]:
        return {"items": [{**asdict(item), "status": item.status.value} for item in self.items], "final_action": self.final_action, "blocking_reasons": list(self.blocking_reasons)}

def make_explanation(items: Iterable[WhySignalItem], final_action: str) -> WhySignalResult:
    values = tuple(items)
    blockers = tuple(item.name for item in values if item.status is ExplainStatus.FAIL)
    return WhySignalResult(values, final_action, blockers)
