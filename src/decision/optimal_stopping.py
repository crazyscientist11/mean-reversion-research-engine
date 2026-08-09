"""Finite-horizon discrete-OU policy iteration; a numerical extension, not Li (2015)'s closed-form solution."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from src.costs import CostModel
from .states import PositionState, ResearchAction


def regions(grid: np.ndarray, actions: np.ndarray, action: ResearchAction) -> list[tuple[float, float]]:
    """Return contiguous state intervals labelled with one policy action."""
    result: list[tuple[float, float]] = []
    start: float | None = None
    for index, (state, enabled) in enumerate(zip(grid, actions == action.value)):
        if enabled and start is None:
            start = float(state)
        if start is not None and (not enabled or index == len(grid) - 1):
            end = float(grid[index] if enabled else grid[index - 1])
            result.append((start, end))
            start = None
    return result


def benchmark_action(z: float, position: PositionState, entry: float = 2.0, exit: float = 0.5, stop: float = 4.0) -> ResearchAction:
    """State-aware z-score benchmark, deliberately separate from the economic policy."""
    if position is PositionState.FLAT:
        return ResearchAction.ENTER_LONG if z <= -entry else ResearchAction.ENTER_SHORT if z >= entry else ResearchAction.WAIT
    if position is PositionState.LONG_RESIDUAL:
        return ResearchAction.STOP_LONG if z <= -stop else ResearchAction.EXIT_LONG if abs(z) <= exit else ResearchAction.HOLD_LONG
    return ResearchAction.STOP_SHORT if z >= stop else ResearchAction.EXIT_SHORT if abs(z) <= exit else ResearchAction.HOLD_SHORT


@dataclass(frozen=True)
class StoppingConfig:
    theta: float
    kappa: float
    sigma: float
    dt: float = 1.0
    grid_min: float = -3.0
    grid_max: float = 3.0
    grid_points: int = 121
    horizon: int = 30
    long_stop: float = -2.5
    short_stop: float = 2.5
    gross_notional: float = 1.0


@dataclass
class StoppingPolicy:
    grid: np.ndarray
    flat_actions: np.ndarray
    long_actions: np.ndarray
    short_actions: np.ndarray
    flat_values: np.ndarray
    long_values: np.ndarray
    short_values: np.ndarray
    action_scores: dict[PositionState, dict[ResearchAction, np.ndarray]]
    config: StoppingConfig

    def action_at(self, x: float, position: PositionState) -> tuple[ResearchAction, float, ResearchAction, float]:
        """Return chosen and runner-up feasible actions at the nearest state-grid point."""
        index = int(np.argmin(np.abs(self.grid - x)))
        candidates = [(action, float(scores[index])) for action, scores in self.action_scores[position].items()]
        candidates.sort(key=lambda item: item[1], reverse=True)
        chosen, chosen_value = candidates[0]
        next_action, next_value = candidates[1] if len(candidates) > 1 else (ResearchAction.WAIT, float("nan"))
        return chosen, chosen_value, next_action, next_value


def _transition_matrix(config: StoppingConfig, grid: np.ndarray) -> np.ndarray:
    spacing = grid[1] - grid[0]
    mean = config.theta + (grid[:, None] - config.theta) * np.exp(-config.kappa * config.dt)
    variance = config.sigma**2 * (-np.expm1(-2.0 * config.kappa * config.dt)) / (2.0 * config.kappa)
    if variance <= 1e-14:
        indices = np.argmin(np.abs(grid[None, :] - mean), axis=1)
        matrix = np.zeros((len(grid), len(grid)))
        matrix[np.arange(len(grid)), indices] = 1.0
        return matrix
    standard_deviation = np.sqrt(variance)
    upper = norm.cdf((grid[None, :] + spacing / 2.0 - mean) / standard_deviation)
    lower = norm.cdf((grid[None, :] - spacing / 2.0 - mean) / standard_deviation)
    matrix = upper - lower
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def build_policy(config: StoppingConfig, costs: CostModel, allow_short: bool = True) -> StoppingPolicy:
    """Solve a bounded finite-horizon daily decision problem using exact OU transition probabilities."""
    if config.kappa <= 0 or config.sigma < 0 or config.grid_points < 5 or config.horizon < 1:
        raise ValueError("invalid numerical OU policy settings")
    if not config.grid_min < config.grid_max or not config.long_stop < config.short_stop:
        raise ValueError("state grid and stop boundaries must be ordered")

    grid = np.linspace(config.grid_min, config.grid_max, config.grid_points)
    transition = _transition_matrix(config, grid)
    discount = costs.daily_discount()
    entry_cost = costs.trade_cost(config.gross_notional, entering=True)
    exit_cost = costs.trade_cost(config.gross_notional, entering=False)
    flat = np.zeros(len(grid))
    long = np.full(len(grid), -exit_cost)
    short = np.full(len(grid), -exit_cost)
    score_book: dict[PositionState, dict[ResearchAction, np.ndarray]] = {}

    for _ in range(config.horizon):
        long_hold = discount * np.sum(transition * (long[None, :] + grid[None, :] - grid[:, None]), axis=1)
        short_hold = discount * np.sum(transition * (short[None, :] + grid[:, None] - grid[None, :]), axis=1)
        long_exit = np.where(grid <= config.long_stop, -np.inf, -exit_cost)
        short_exit = np.where(grid >= config.short_stop, -np.inf, -exit_cost)
        long_stop = np.where(grid <= config.long_stop, -exit_cost, -np.inf)
        short_stop = np.where(grid >= config.short_stop, -exit_cost, -np.inf)
        long_hold = np.where(grid <= config.long_stop, -np.inf, long_hold)
        short_hold = np.where(grid >= config.short_stop, -np.inf, short_hold)
        long = np.maximum.reduce([long_hold, long_exit, long_stop])
        short = np.maximum.reduce([short_hold, short_exit, short_stop])

        flat_wait = np.zeros(len(grid))
        enter_long = -entry_cost + discount * (transition @ long)
        enter_short = -entry_cost + discount * (transition @ short) if allow_short else np.full(len(grid), -np.inf)
        flat = np.maximum.reduce([flat_wait, enter_long, enter_short])
        score_book = {
            PositionState.FLAT: {ResearchAction.WAIT: flat_wait, ResearchAction.ENTER_LONG: enter_long, ResearchAction.ENTER_SHORT: enter_short},
            PositionState.LONG_RESIDUAL: {ResearchAction.HOLD_LONG: long_hold, ResearchAction.EXIT_LONG: long_exit, ResearchAction.STOP_LONG: long_stop},
            PositionState.SHORT_RESIDUAL: {ResearchAction.HOLD_SHORT: short_hold, ResearchAction.EXIT_SHORT: short_exit, ResearchAction.STOP_SHORT: short_stop},
        }

    def chosen_actions(position: PositionState) -> np.ndarray:
        scores = score_book[position]
        actions = list(scores)
        score_matrix = np.vstack([scores[action] for action in actions])
        return np.asarray([actions[index].value for index in np.argmax(score_matrix, axis=0)])

    return StoppingPolicy(grid, chosen_actions(PositionState.FLAT), chosen_actions(PositionState.LONG_RESIDUAL), chosen_actions(PositionState.SHORT_RESIDUAL), flat, long, short, score_book, config)


def sensitivity(config: StoppingConfig, costs: CostModel, costs_to_test: list[float]) -> list[dict]:
    """Re-solve the policy at several entry costs; this is a diagnostic, not an optimiser."""
    rows: list[dict] = []
    for entry_cost in costs_to_test:
        adjusted = CostModel(fixed_entry_cost=entry_cost, fixed_exit_cost=costs.fixed_exit_cost, bid_ask_bps=costs.bid_ask_bps, slippage_bps=costs.slippage_bps, commission=costs.commission, annual_opportunity_rate=costs.annual_opportunity_rate)
        policy = build_policy(config, adjusted)
        rows.append({"fixed_entry_cost": entry_cost, "long_entry_regions": regions(policy.grid, policy.flat_actions, ResearchAction.ENTER_LONG), "short_entry_regions": regions(policy.grid, policy.flat_actions, ResearchAction.ENTER_SHORT)})
    return rows
