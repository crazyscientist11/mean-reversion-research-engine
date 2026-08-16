from datetime import datetime, timezone
import pytest

from src.config import Frequency, PredictionStorageParameters
from src.prediction import FrozenPairModel, PredictionStore, classify_live_pair, make_live_evaluation, new_prediction_snapshot, pair_implied_target_price, pair_live_residual


def model(): return FrozenPairModel("AAA", "BBB", .2, 1.5, -.4, long_entry_region=(-.7, -.3), exit_boundary=0, long_stop_boundary=-1)
def snapshot(): return new_prediction_snapshot(target_ticker="AAA", current_price=100, data_cutoff=datetime(2025,1,1,tzinfo=timezone.utc), frequency=Frequency.DAILY, data_source="test", model_version="11", frozen_outputs={"model_outputs": {"pair": {"alpha": .2, "beta": 1.5}}, "probability_outputs": {"exit_5d": .4}})

def test_pair_residual_and_boundary_inversion_algebra():
    boundary = -.4; peer = 50; target = pair_implied_target_price(boundary, peer, .2, 1.5)
    assert pair_live_residual(target, peer, .2, 1.5) == pytest.approx(boundary)

def test_peer_moves_implied_price_but_not_frozen_residual_boundary():
    first = classify_live_pair(model(), target_price=100, peer_price=50)
    second = classify_live_pair(model(), target_price=100, peer_price=60)
    assert first.implied_target_prices["exit"] != second.implied_target_prices["exit"]
    assert model().exit_boundary == 0

def test_live_update_and_evaluation_cannot_mutate_snapshot(tmp_path):
    original = snapshot(); before = original.to_dict()
    state = classify_live_pair(model(), target_price=90, peer_price=50)
    evaluation = make_live_evaluation(original, timestamp=datetime(2025,1,2,tzinfo=timezone.utc), live_prices={"AAA": 90, "BBB": 50}, live_state=state, elapsed_trading_days=1)
    store = PredictionStore(PredictionStorageParameters(tmp_path)); store.save_snapshot(original); store.save_evaluation(evaluation)
    assert original.to_dict() == before
    assert original.probability_outputs["exit_5d"] == .4

def test_new_predictions_have_unique_ids():
    assert snapshot().prediction_id != snapshot().prediction_id

def test_fraction_reverted_is_not_clipped_when_residual_worsens():
    state = classify_live_pair(model(), target_price=pair_implied_target_price(-.8, 50, .2, 1.5), peer_price=50)
    assert state.fraction_reverted < 0
