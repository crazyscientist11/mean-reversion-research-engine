from datetime import datetime, timezone

from src.config import Frequency
from src.prediction.schemas import PredictionSnapshot, parameter_hash


def make_snapshot() -> PredictionSnapshot:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return PredictionSnapshot("p-1", now, now, "AAA", Frequency.DAILY, 100.0, "0.1", "CSV", parameter_hash({"b": 2, "a": 1}))


def test_snapshot_round_trip() -> None:
    snapshot = make_snapshot()
    assert PredictionSnapshot.from_dict(snapshot.to_dict()) == snapshot


def test_parameter_hash_is_stable_and_sensitive() -> None:
    assert parameter_hash({"a": 1, "b": 2}) == parameter_hash({"b": 2, "a": 1})
    assert parameter_hash({"a": 1}) != parameter_hash({"a": 2})

