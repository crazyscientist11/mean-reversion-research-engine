"""Small end-to-end foundation check; creates no persistent predictions."""
from datetime import datetime, timezone
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.csv_provider import CSVDataProvider
from src.data.transforms import log_returns
from src.config import PredictionStorageParameters
from src.prediction.schemas import PredictionEvaluation, PredictionSnapshot, parameter_hash
from src.prediction.store import PredictionStore


def check(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"FAIL: {label}")
    print(f"PASS: {label}")


def main() -> None:
    bundle = CSVDataProvider().load(PROJECT_ROOT / "data" / "example_synthetic_prices.csv")
    returns = log_returns(bundle.prices)
    check(not returns.empty, "log returns calculated")
    print("Data summary:", bundle.summary())
    now = datetime.now(timezone.utc)
    snapshot = PredictionSnapshot("smoke-001", now, now, "AAPL", bundle.frequency, float(bundle.prices.iloc[-1, 0]), "0.1.0", "synthetic CSV", parameter_hash({"window": 60}))
    with TemporaryDirectory() as directory:
        store = PredictionStore(PredictionStorageParameters(Path(directory)))
        snapshot_path = store.save_snapshot(snapshot)
        original = snapshot_path.read_text(encoding="utf-8")
        check(store.load_snapshot("smoke-001") == snapshot, "snapshot round-trip")
        store.save_evaluation(PredictionEvaluation("smoke-001", now, "5d", realized_price=200.0))
        check(snapshot_path.read_text(encoding="utf-8") == original, "evaluation leaves snapshot unchanged")
    print("PASS: smoke test complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise

