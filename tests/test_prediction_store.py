from datetime import datetime, timezone

import pytest

from src.config import Frequency, PredictionStorageParameters
from src.prediction.schemas import PredictionEvaluation, PredictionSnapshot, parameter_hash
from src.prediction.store import PredictionStore


def snapshot() -> PredictionSnapshot:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return PredictionSnapshot("p-1", now, now, "AAA", Frequency.DAILY, 100.0, "0.1", "CSV", parameter_hash({}))


def test_duplicate_snapshot_is_not_silently_overwritten(tmp_path) -> None:
    store = PredictionStore(PredictionStorageParameters(tmp_path))
    store.save_snapshot(snapshot())
    with pytest.raises(FileExistsError):
        store.save_snapshot(snapshot())


def test_evaluation_is_separate_from_snapshot(tmp_path) -> None:
    store = PredictionStore(PredictionStorageParameters(tmp_path))
    path = store.save_snapshot(snapshot())
    original = path.read_text(encoding="utf-8")
    store.save_evaluation(PredictionEvaluation("p-1", datetime(2025, 1, 2, tzinfo=timezone.utc), "1d", realized_price=101.0))
    assert path.read_text(encoding="utf-8") == original
    assert len(store.list_evaluations("p-1")) == 1


def test_malformed_snapshot_is_rejected(tmp_path) -> None:
    store = PredictionStore(PredictionStorageParameters(tmp_path))
    bad = tmp_path / "snapshots" / "bad.json"
    bad.parent.mkdir()
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        store.load_snapshot("bad")

