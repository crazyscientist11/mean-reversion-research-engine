"""Schemas that keep model-time beliefs separate from future outcomes."""
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
from typing import Any

from src.config import Frequency


class FrozenDict(dict):
    """A JSON-serializable mapping that rejects in-place mutation."""
    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("prediction snapshots are immutable")
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def parameter_hash(parameters: dict[str, Any]) -> str:
    """SHA-256 of canonical JSON; stable across Python processes and dict order."""
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _timestamp(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(parsed, datetime):
        raise ValueError("timestamp must be an ISO datetime")
    return parsed


@dataclass(frozen=True)
class PredictionSnapshot:
    prediction_id: str
    created_at: datetime
    data_cutoff: datetime
    target_ticker: str
    frequency: Frequency
    current_price: float
    model_version: str
    data_source: str
    parameter_hash: str
    notes: str = ""
    model_outputs: dict[str, Any] = field(default_factory=dict)
    ou_outputs: dict[str, Any] = field(default_factory=dict)
    stopping_outputs: dict[str, Any] = field(default_factory=dict)
    probability_outputs: dict[str, Any] = field(default_factory=dict)
    diagnostic_outputs: dict[str, Any] = field(default_factory=dict)
    consensus_outputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prediction_id.strip() or not self.target_ticker.strip():
            raise ValueError("prediction_id and target_ticker are required")
        if not isinstance(self.frequency, Frequency):
            raise ValueError("frequency must be a Frequency value")
        if self.current_price <= 0:
            raise ValueError("current_price must be positive")
        if not self.model_version.strip() or len(self.parameter_hash) != 64:
            raise ValueError("model_version and a SHA-256 parameter_hash are required")
        for name in ("model_outputs", "ou_outputs", "stopping_outputs", "probability_outputs", "diagnostic_outputs", "consensus_outputs"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        result["data_cutoff"] = self.data_cutoff.isoformat()
        result["frequency"] = self.frequency.value
        return result

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "PredictionSnapshot":
        record = record.copy()
        record["created_at"] = _timestamp(record["created_at"])
        record["data_cutoff"] = _timestamp(record["data_cutoff"])
        record["frequency"] = Frequency(record["frequency"])
        return cls(**record)


@dataclass(frozen=True)
class PredictionEvaluation:
    prediction_id: str
    evaluation_timestamp: datetime
    evaluation_horizon: str
    realized_price: float | None = None
    exit_reached: bool | None = None
    stop_reached: bool | None = None
    fraction_reverted: float | None = None
    maximum_favorable_excursion: float | None = None
    maximum_adverse_excursion: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.prediction_id.strip() or not self.evaluation_horizon.strip():
            raise ValueError("prediction_id and evaluation_horizon are required")
        if self.realized_price is not None and self.realized_price <= 0:
            raise ValueError("realized_price must be positive when supplied")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evaluation_timestamp"] = self.evaluation_timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "PredictionEvaluation":
        record = record.copy()
        record["evaluation_timestamp"] = _timestamp(record["evaluation_timestamp"])
        return cls(**record)
