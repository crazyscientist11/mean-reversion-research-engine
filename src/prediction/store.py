"""Transparent JSON store with immutable snapshot files and separate evaluations."""
from datetime import datetime
import json
from pathlib import Path

from src.config import PredictionStorageParameters
from .schemas import PredictionEvaluation, PredictionSnapshot


class PredictionStore:
    def __init__(self, parameters: PredictionStorageParameters | None = None) -> None:
        self.parameters = parameters or PredictionStorageParameters()
        self.root = Path(self.parameters.storage_directory)
        self.snapshots_dir = self.root / "snapshots"
        self.evaluations_dir = self.root / "evaluations"

    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def save_snapshot(self, snapshot: PredictionSnapshot, *, overwrite: bool = False) -> Path:
        if not isinstance(snapshot, PredictionSnapshot):
            raise TypeError("snapshot must be a PredictionSnapshot")
        path = self.snapshots_dir / f"{snapshot.prediction_id}.json"
        if path.exists() and not (overwrite and self.parameters.allow_overwrite):
            raise FileExistsError("snapshot already exists; explicit configuration and overwrite are required")
        self._write(path, snapshot.to_dict())
        return path

    def load_snapshot(self, prediction_id: str) -> PredictionSnapshot:
        path = self.snapshots_dir / f"{prediction_id}.json"
        try:
            return PredictionSnapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed or missing snapshot {prediction_id!r}") from exc

    def list_snapshots(self) -> list[str]:
        if not self.snapshots_dir.exists():
            return []
        return sorted(path.stem for path in self.snapshots_dir.glob("*.json"))

    def save_evaluation(self, evaluation: PredictionEvaluation) -> Path:
        if not isinstance(evaluation, PredictionEvaluation):
            raise TypeError("evaluation must be a PredictionEvaluation")
        # A timestamped record adds future facts without touching the snapshot file.
        stamp = evaluation.evaluation_timestamp.strftime("%Y%m%dT%H%M%S%f")
        path = self.evaluations_dir / evaluation.prediction_id / f"{stamp}.json"
        if path.exists():
            raise FileExistsError("an evaluation already exists at this timestamp")
        self._write(path, evaluation.to_dict())
        return path

    def list_evaluations(self, prediction_id: str) -> list[PredictionEvaluation]:
        directory = self.evaluations_dir / prediction_id
        if not directory.exists():
            return []
        return [PredictionEvaluation.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(directory.glob("*.json"))]

