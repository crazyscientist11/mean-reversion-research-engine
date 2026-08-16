"""Live observations against frozen prediction models; never refits a model."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import exp, log
from typing import Iterable
from uuid import uuid4

from src.config import Frequency
from .schemas import PredictionEvaluation, PredictionSnapshot, parameter_hash


class LiveResearchState(str, Enum):
    WAIT = "WAIT"
    ENTRY_REGION = "ENTRY_REGION"
    HOLD = "HOLD"
    EXIT_REGION = "EXIT_REGION"
    STOP_REGION = "STOP_REGION"


@dataclass(frozen=True)
class FrozenPairModel:
    target_ticker: str
    peer_ticker: str
    alpha: float
    beta: float
    initial_residual: float
    long_entry_region: tuple[float, float] | None = None
    short_entry_region: tuple[float, float] | None = None
    exit_boundary: float | None = None
    long_stop_boundary: float | None = None
    short_stop_boundary: float | None = None

    def residual(self, target_price: float, peer_price: float) -> float:
        return pair_live_residual(target_price, peer_price, self.alpha, self.beta)


@dataclass(frozen=True)
class LivePredictionState:
    current_residual: float
    state: LiveResearchState
    distance_to_entry: float | None
    distance_to_exit: float | None
    distance_to_stop: float | None
    implied_target_prices: dict[str, float | tuple[float, float]]
    fraction_reverted: float | None


def pair_live_residual(target_price: float, peer_price: float, alpha: float, beta: float) -> float:
    """X_live = log(P_target) - alpha - beta*log(P_peer), using frozen alpha/beta."""
    if target_price <= 0 or peer_price <= 0:
        raise ValueError("pair prices must be positive")
    return log(float(target_price)) - float(alpha) - float(beta) * log(float(peer_price))


def pair_implied_target_price(residual_boundary: float, peer_price: float, alpha: float, beta: float) -> float:
    """Invert a frozen residual boundary into a target price at the live peer price."""
    if peer_price <= 0:
        raise ValueError("peer price must be positive")
    return exp(float(residual_boundary) + float(alpha) + float(beta) * log(float(peer_price)))


def pair_implied_boundaries(model: FrozenPairModel, peer_price: float) -> dict[str, float | tuple[float, float]]:
    result: dict[str, float | tuple[float, float]] = {}
    for name, boundary in (("long_entry", model.long_entry_region), ("short_entry", model.short_entry_region)):
        if boundary is not None:
            result[name] = tuple(pair_implied_target_price(value, peer_price, model.alpha, model.beta) for value in boundary)
    for name, boundary in (("exit", model.exit_boundary), ("long_stop", model.long_stop_boundary), ("short_stop", model.short_stop_boundary)):
        if boundary is not None:
            result[name] = pair_implied_target_price(boundary, peer_price, model.alpha, model.beta)
    return result


def _distance(value: float, region: tuple[float, float] | None) -> float | None:
    if region is None: return None
    low, high = sorted(region)
    return 0.0 if low <= value <= high else min(abs(value - low), abs(value - high))


def classify_live_pair(model: FrozenPairModel, *, target_price: float, peer_price: float, position: str = "FLAT") -> LivePredictionState:
    """Classify the live observation against frozen boundaries without changing them."""
    value = model.residual(target_price, peer_price)
    long_entry = _distance(value, model.long_entry_region)
    short_entry = _distance(value, model.short_entry_region)
    entry_distance = min((item for item in (long_entry, short_entry) if item is not None), default=None)
    exit_distance = None if model.exit_boundary is None else abs(value - model.exit_boundary)
    stop = model.long_stop_boundary if position == "LONG_RESIDUAL" else model.short_stop_boundary if position == "SHORT_RESIDUAL" else None
    stop_distance = None if stop is None else abs(value - stop)
    if position == "LONG_RESIDUAL":
        state = LiveResearchState.STOP_REGION if stop is not None and value <= stop else LiveResearchState.EXIT_REGION if model.exit_boundary is not None and value >= model.exit_boundary else LiveResearchState.HOLD
    elif position == "SHORT_RESIDUAL":
        state = LiveResearchState.STOP_REGION if stop is not None and value >= stop else LiveResearchState.EXIT_REGION if model.exit_boundary is not None and value <= model.exit_boundary else LiveResearchState.HOLD
    elif long_entry == 0 or short_entry == 0:
        state = LiveResearchState.ENTRY_REGION
    else:
        state = LiveResearchState.WAIT
    denominator = model.initial_residual - model.exit_boundary if model.exit_boundary is not None else None
    fraction = None if denominator is None or abs(denominator) < 1e-12 else (model.initial_residual - value) / denominator
    return LivePredictionState(value, state, entry_distance, exit_distance, stop_distance, pair_implied_boundaries(model, peer_price), fraction)


def make_live_evaluation(snapshot: PredictionSnapshot, *, timestamp: datetime, live_prices: dict[str, float], live_state: LivePredictionState, elapsed_trading_days: int, realized_pnl: float | None = None, mfe: float | None = None, mae: float | None = None) -> PredictionEvaluation:
    """Create a separate evaluation record. The supplied snapshot is read only."""
    return PredictionEvaluation(snapshot.prediction_id, timestamp, f"{elapsed_trading_days} trading days", realized_price=live_prices.get(snapshot.target_ticker), exit_reached=live_state.state is LiveResearchState.EXIT_REGION, stop_reached=live_state.state is LiveResearchState.STOP_REGION, fraction_reverted=live_state.fraction_reverted, maximum_favorable_excursion=mfe, maximum_adverse_excursion=mae, elapsed_trading_days=elapsed_trading_days, live_prices=dict(live_prices), current_residual=live_state.current_residual, distance_to_entry=live_state.distance_to_entry, distance_to_exit=live_state.distance_to_exit, distance_to_stop=live_state.distance_to_stop, realized_pnl=realized_pnl)


def new_prediction_snapshot(*, target_ticker: str, current_price: float, data_cutoff: datetime, frequency: Frequency, data_source: str, model_version: str, frozen_outputs: dict, notes: str = "") -> PredictionSnapshot:
    """Create a deliberately new immutable snapshot with a unique ID and parameter hash."""
    identifier = f"prediction-{uuid4()}"
    return PredictionSnapshot(identifier, datetime.now(data_cutoff.tzinfo), data_cutoff, target_ticker, frequency, current_price, model_version, data_source, parameter_hash(frozen_outputs), notes, **frozen_outputs)
