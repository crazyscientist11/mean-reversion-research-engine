"""Step 1 landing page plus Step 2 single-stock research outputs."""
from io import BytesIO
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from src.data.validation import validate_price_frame
from src.models import detrended_log_price_residual, factor_residual_model, rolling_zscore_benchmark
from src.models.common import ResidualModelResult
from src.models.pairs import PairModelResult, peer_comparison, rolling_pair_model
from src.models.ou import OUComparisonResult, OUFitResult, conditional_band, fit_ou_both
from src.consensus import EvidenceDirection, ModelEvidence, build_consensus
from src.costs import CostModel
from src.decision import CriticalGates, PositionState, ResearchAction, StoppingConfig, build_final_decision, build_policy, regions
from src.first_passage import BoundaryDefinition, ResidualDirection
from src.monte_carlo import simulate_ou_first_passage
from src.config import Frequency
from src.prediction import FrozenPairModel, PredictionStore, classify_live_pair, make_live_evaluation, new_prediction_snapshot
from src.backtest import BenchmarkConfig, run_backtest
from src.backtest.engine import BacktestConfig
from src.data import SeriesMapping, data_quality_report, detect_csv_format, model_readiness_report, normalize_bloomberg_frame
from statsmodels.tsa.stattools import adfuller, kpss


def _format(value: float | None, decimals: int = 4) -> str:
    return "Not available" if value is None or not np.isfinite(value) else f"{value:.{decimals}f}"


@st.cache_data(show_spinner=False)
def _run_models(prices: pd.DataFrame, target: str, market: str, sector: str | None, benchmark_window: int, regression_window: int) -> tuple[ResidualModelResult, ResidualModelResult, ResidualModelResult]:
    return (
        rolling_zscore_benchmark(prices[target], window=benchmark_window),
        detrended_log_price_residual(prices[target], window=regression_window),
        factor_residual_model(prices, target_ticker=target, market_ticker=market, sector_ticker=sector, window=regression_window),
    )


@st.cache_data(show_spinner=False)
def _run_pair_model(prices: pd.DataFrame, target: str, peer: str, window: int, zscore_threshold: float) -> PairModelResult:
    return rolling_pair_model(prices, target_ticker=target, peer_ticker=peer, window=window, zscore_threshold=zscore_threshold)


@st.cache_data(show_spinner=False)
def _peer_table(prices: pd.DataFrame, target: str, peers: tuple[str, ...], window: int, zscore_threshold: float) -> pd.DataFrame:
    return peer_comparison(prices, target_ticker=target, candidate_peers=peers, window=window, zscore_threshold=zscore_threshold)


@st.cache_data(show_spinner=False)
def _fit_ou_models(state: pd.Series, window: int) -> OUComparisonResult:
    return fit_ou_both(state.iloc[-window:], dt=1.0, time_unit="trading days")


def _model_metric_card(result: ResidualModelResult, labels: tuple[str, str, str]) -> None:
    snapshot = result.current_snapshot()
    with st.container(border=True):
        st.subheader(result.model_name)
        if not snapshot.valid:
            st.warning(f"Invalid for the current date: {snapshot.invalid_reason}")
        with st.container(horizontal=True):
            st.metric(labels[0], _format(snapshot.current_observed_value))
            st.metric(labels[1], _format(snapshot.current_expected_value))
            st.metric(labels[2], _format(snapshot.current_zscore))


def _single_stock_models(prices: pd.DataFrame) -> None:
    st.subheader("Single stock models")
    st.caption("Research outputs use only prior observations for each fitted baseline. They are not trading recommendations.")
    tickers = list(prices.columns)
    with st.form("single_stock_models"):
        target = st.selectbox("Target ticker", tickers)
        market_choices = [ticker for ticker in tickers if ticker != target]
        market = st.selectbox("Market ticker", market_choices)
        sector_choices = ["No sector factor"] + [ticker for ticker in market_choices if ticker != market]
        sector_choice = st.selectbox("Sector ticker (optional)", sector_choices)
        with st.container(horizontal=True):
            benchmark_window = int(st.number_input("Benchmark window", min_value=2, value=60, step=1))
            regression_window = int(st.number_input("Trend and factor window", min_value=3, value=126, step=1))
        submitted = st.form_submit_button("Run research models")
    if not submitted:
        st.info("Select inputs and run the models. No analysis runs automatically.")
        return
    sector = None if sector_choice == "No sector factor" else sector_choice
    try:
        benchmark, trend, factors = _run_models(prices, target, market, sector, benchmark_window, regression_window)
    except (TypeError, ValueError) as exc:
        st.error(f"Model input is invalid: {exc}")
        return
    _model_metric_card(benchmark, ("Current log price", "Rolling statistical baseline", "Z-score"))
    _model_metric_card(trend, ("Current log price", "Trend-implied log price", "Residual z-score"))
    _model_metric_card(factors, ("Actual return", "Model-implied return", "Residual z-score"))
    factor_snapshot = factors.current_snapshot()
    with st.container(border=True):
        st.subheader("Factor regression details")
        date = factor_snapshot.as_of_date
        st.write({
            "market beta": _format(factors.parameters.loc[date, "beta_market"]),
            "sector beta": _format(factors.parameters.loc[date, "beta_sector"]),
            "R-squared": _format(factors.r_squared_series.loc[date]),
            "factor residual": _format(factor_snapshot.current_residual),
        })
    with st.container(border=True):
        st.subheader("Price and trend statistical baseline")
        chart = pd.DataFrame({"Price": prices[target], "Trend-implied price": trend.metadata["expected_price"]})
        st.line_chart(chart)
    with st.container(border=True):
        st.subheader("Benchmark z-score history")
        st.line_chart(benchmark.zscore_series)
    with st.container(border=True):
        st.subheader("Factor residual history")
        st.line_chart(factors.residual_series)
    with st.container(border=True):
        st.subheader("Accumulated factor residual state")
        st.caption("Rolling sum of factor residual returns; not a stationarity finding.")
        st.line_chart(factors.state_series)


def _pairs_research(prices: pd.DataFrame) -> None:
    st.subheader("Pairs research")
    st.caption("Rolling diagnostics use data through the prior date for every fitted pair relationship. Cointegration can break and is not guaranteed to persist.")
    tickers = list(prices.columns)
    with st.form("pairs_research"):
        target = st.selectbox("Pair target ticker", tickers)
        peer = st.selectbox("Pair peer ticker", tickers, index=1 if len(tickers) > 1 else 0)
        candidates = st.multiselect("Candidate peer universe", tickers, default=[ticker for ticker in tickers if ticker != target])
        with st.container(horizontal=True):
            window = int(st.number_input("Pair estimation window", min_value=20, value=252, step=1))
            threshold = float(st.number_input("Pair z-score threshold", min_value=0.1, value=2.0, step=0.1))
        submitted = st.form_submit_button("Run pair research")
    if not submitted:
        st.info("Select a target, peer, and candidate universe, then run pair research.")
        return
    if target == peer:
        st.error("Target and peer tickers must differ.")
        return
    try:
        result = _run_pair_model(prices, target, peer, window, threshold)
        table = _peer_table(prices, target, tuple(candidates), window, threshold)
    except (TypeError, ValueError) as exc:
        st.error(f"Pair inputs are invalid: {exc}")
        return
    snapshot = result.current_snapshot()
    with st.container(border=True):
        st.subheader("Pair regression and current research state")
        if not snapshot.valid:
            st.warning(f"Invalid for the current date: {snapshot.invalid_reason}")
        with st.container(horizontal=True):
            st.metric("Hedge ratio (regression beta)", _format(snapshot.beta))
            st.metric("Current spread", _format(snapshot.spread))
            st.metric("Spread z-score", _format(snapshot.spread_zscore))
            st.metric("Relative state", snapshot.relative_state.value)
        st.write({
            "alpha": _format(snapshot.alpha), "R-squared": _format(snapshot.r_squared),
            "cointegration statistic": _format(snapshot.cointegration_statistic),
            "cointegration p-value": _format(snapshot.cointegration_pvalue),
            "return correlation": _format(snapshot.return_correlation),
        })
        st.caption("Regression beta is not a dollar-neutral position size. The theoretical notation is +1 target regression unit and -beta peer regression units.")
    with st.container(border=True):
        st.subheader("Normalized target and peer prices")
        normalized = prices[[target, peer]].divide(prices[[target, peer]].iloc[0]).rename(columns={target: f"{target} normalized", peer: f"{peer} normalized"})
        st.line_chart(normalized)
    with st.container(border=True):
        st.subheader("Pair spread history")
        st.line_chart(result.spread_series)
    with st.container(border=True):
        st.subheader("Pair spread z-score history")
        st.line_chart(result.zscore_series)
    with st.container(border=True):
        st.subheader("Rolling hedge ratio")
        st.line_chart(result.beta_series)
    with st.container(border=True):
        st.subheader("Peer comparison research table")
        st.caption("The table is a transparent historical comparison. It does not automatically select a pair, and correlation alone is insufficient evidence of cointegration.")
        st.dataframe(table, hide_index=True)


def _ou_state(prices: pd.DataFrame, source: str, target: str, market: str, peer: str, source_window: int) -> tuple[pd.Series, str]:
    if source == "Detrended log-price residual":
        return detrended_log_price_residual(prices[target], window=source_window).residual_series.dropna(), source
    if source == "Accumulated factor residual state":
        return factor_residual_model(prices, target_ticker=target, market_ticker=market, window=source_window).state_series.dropna(), source
    if source == "Pair spread":
        return rolling_pair_model(prices, target_ticker=target, peer_ticker=peer, window=source_window).spread_series.dropna(), source
    raise ValueError("Unknown OU state source.")


def _ou_dynamics(prices: pd.DataFrame) -> None:
    st.subheader("OU dynamics")
    st.caption("Fits are in trading-day units (`dt = 1`). Conditional convergence is not a guaranteed realized path or exit time.")
    tickers = list(prices.columns)
    with st.form("ou_dynamics"):
        source = st.selectbox("Residual state source", ["Detrended log-price residual", "Accumulated factor residual state", "Pair spread"])
        target = st.selectbox("OU target ticker", tickers)
        market = st.selectbox("OU market ticker (factor source)", tickers, index=1 if len(tickers) > 1 else 0)
        peer = st.selectbox("OU peer ticker (pair source)", tickers, index=1 if len(tickers) > 1 else 0)
        with st.container(horizontal=True):
            source_window = int(st.number_input("Residual source window", min_value=20, value=126, step=1))
            ou_window = int(st.number_input("OU estimation window", min_value=20, value=252, step=1))
            horizon = int(st.number_input("Conditional-path horizon", min_value=1, value=20, step=1))
        estimator = st.selectbox("Estimator display", ["Both", "AR1", "Exact MLE"])
        submitted = st.form_submit_button("Fit OU dynamics")
    if not submitted:
        st.info("Select a residual state and fit the OU diagnostics. No model runs automatically.")
        return
    if source == "Accumulated factor residual state" and target == market:
        st.error("Factor target and market tickers must differ.")
        return
    if source == "Pair spread" and target == peer:
        st.error("Pair target and peer tickers must differ.")
        return
    try:
        state, state_label = _ou_state(prices, source, target, market, peer, source_window)
        if len(state) < ou_window:
            raise ValueError(f"{state_label} has {len(state)} usable observations; OU window requires {ou_window}.")
        comparison = _fit_ou_models(state, ou_window)
    except (TypeError, ValueError) as exc:
        st.error(f"OU inputs are invalid: {exc}")
        return
    table = comparison.as_table()
    if estimator != "Both":
        table = table.loc[table["method"] == ("AR1" if estimator == "AR1" else "MLE")]
    with st.container(border=True):
        st.subheader("OU estimator comparison")
        st.dataframe(table, hide_index=True)
        st.caption("Differences between AR(1) and exact MLE are diagnostics for later research; they do not automatically reject a model.")
    primary: OUFitResult = comparison.mle if comparison.mle.valid else comparison.ar1
    if not primary.valid:
        st.warning(f"No valid OU fit is available. AR(1): {comparison.ar1.invalid_reason}; Exact MLE: {comparison.mle.invalid_reason}")
        return
    with st.container(border=True):
        st.subheader("Current OU research output")
        with st.container(horizontal=True):
            st.metric("Theta", _format(primary.theta))
            st.metric("Kappa per trading day", _format(primary.kappa))
            st.metric("Sigma per sqrt(trading day)", _format(primary.sigma))
            st.metric("Half-life (trading days)", _format(primary.half_life))
            st.metric("75% / 90% expected decay", f"{_format(primary.reversion_75_time, 1)} / {_format(primary.reversion_90_time, 1)}")
    with st.container(border=True):
        st.subheader("Historical residual state and theta")
        history = pd.DataFrame({"Residual state": state, "Theta": primary.theta}, index=state.index)
        st.line_chart(history)
    horizons = np.arange(horizon + 1, dtype=float)
    current = float(state.iloc[-1])
    paths = [conditional_band(primary, current, float(step)) for step in horizons]
    future = pd.DataFrame(paths, columns=["Expected residual state", "Lower model-implied band", "Upper model-implied band"], index=pd.Index(horizons, name="Trading-day horizon"))
    with st.container(border=True):
        st.subheader("OU conditional expected path")
        st.line_chart(future)
        st.caption("Bands are OU model-implied conditional distribution bands, not guaranteed confidence intervals. Half-life describes expected displacement decay, not an expected realized exit time. First-passage analysis is deferred.")


def _decision_monitor(prices: pd.DataFrame) -> None:
    st.header("Decision monitor")
    st.caption("A research-state summary. It is not investment advice, an order, or evidence of future profitability.")
    tickers = list(prices.columns)
    with st.form("decision_monitor"):
        target = st.selectbox("Decision-monitor target", tickers)
        window = int(st.number_input("Residual and OU window", min_value=30, value=min(126, max(30, len(prices) // 2)), step=1))
        position = st.selectbox("Position state", list(PositionState), format_func=lambda item: item.value)
        submitted = st.form_submit_button("Evaluate final research decision")
    if not submitted:
        st.info("Choose a target and evaluate the frozen current-data research state.")
        return
    try:
        residual_model = detrended_log_price_residual(prices[target], window=window)
        snapshot = residual_model.current_snapshot()
        state = residual_model.residual_series.dropna()
        if not snapshot.valid or len(state) < max(window, 20):
            raise ValueError("The selected residual has insufficient valid history for the requested window.")
        ou = _fit_ou_models(state, min(window, len(state)))
        primary = ou.mle if ou.mle.valid else ou.ar1
        adf_p = float(adfuller(state.iloc[-window:], autolag="AIC")[1])
        kpss_p = float(kpss(state.iloc[-window:], regression="c", nlags="auto")[1])
        stationarity = adf_p < .05 and kpss_p > .05
        ou_agreement = ou.ar1.valid and ou.mle.valid and abs(ou.ar1.kappa - ou.mle.kappa) / max(ou.mle.kappa, 1e-12) <= 0.5
        std = float(state.iloc[-window:].std(ddof=1))
        current = float(snapshot.current_residual)
        direction = EvidenceDirection.LONG_RESIDUAL if snapshot.current_zscore is not None and snapshot.current_zscore <= -2 else EvidenceDirection.SHORT_RESIDUAL if snapshot.current_zscore is not None and snapshot.current_zscore >= 2 else EvidenceDirection.NEUTRAL
        consensus = build_consensus([ModelEvidence("Detrended residual", direction, snapshot.valid, snapshot.current_zscore, "trend residual", primary.valid, dependency_group="single-stock")])
        long_stop = float(primary.theta - 2.5 * std) if primary.valid else None
        short_stop = float(primary.theta + 2.5 * std) if primary.valid else None
        policy = build_policy(StoppingConfig(theta=float(primary.theta), kappa=float(primary.kappa), sigma=float(primary.sigma), grid_min=float(primary.theta - 3 * std), grid_max=float(primary.theta + 3 * std), long_stop=long_stop, short_stop=short_stop), CostModel()) if primary.valid else None
        policy_action = policy.action_at(current, position)[0] if policy else ResearchAction.NO_SIGNAL
        fp = None
        if primary.valid and direction is not EvidenceDirection.NEUTRAL:
            boundary = BoundaryDefinition(current, float(primary.theta), long_stop if direction is EvidenceDirection.LONG_RESIDUAL else short_stop, ResidualDirection.LONG_RESIDUAL if direction is EvidenceDirection.LONG_RESIDUAL else ResidualDirection.SHORT_RESIDUAL)
            fp = simulate_ou_first_passage(theta=float(primary.theta), kappa=float(primary.kappa), sigma=float(primary.sigma), boundaries=boundary, number_paths=2000, seed=7)
        action = policy_action
        final = build_final_decision(decision_id=f"monitor-{snapshot.as_of_date.date()}", as_of=snapshot.as_of_date.to_pydatetime(), model_cutoff=snapshot.as_of_date.to_pydatetime(), target_expression=target, position_state=position, current_prices={target: float(prices[target].iloc[-1])}, current_residual=current, current_zscore=snapshot.current_zscore, consensus=consensus, gates=CriticalGates(sufficient_data=snapshot.valid, residual_valid=snapshot.valid, stationarity_valid=stationarity, ou_valid=primary.valid, half_life_acceptable=primary.valid and primary.half_life is not None and primary.half_life <= 60, parameter_stable=ou_agreement, regime_status="NORMAL"), policy_action=action, ou_theta=primary.theta, ou_kappa=primary.kappa, ou_sigma=primary.sigma, ou_half_life=primary.half_life, long_entry_region=regions(policy.grid, policy.flat_actions, ResearchAction.ENTER_LONG)[0] if policy and regions(policy.grid, policy.flat_actions, ResearchAction.ENTER_LONG) else None, short_entry_region=regions(policy.grid, policy.flat_actions, ResearchAction.ENTER_SHORT)[0] if policy and regions(policy.grid, policy.flat_actions, ResearchAction.ENTER_SHORT) else None, exit_region=(float(primary.theta), float(primary.theta)) if primary.valid else None, stop_region=(long_stop, short_stop) if primary.valid else None, probability_outputs={} if fp is None else {"exit_before_stop": fp.exit_before_stop, "stop_before_exit": fp.stop_before_exit, "exit_5d": fp.exit_within_5, "exit_10d": fp.exit_within_10, "exit_20d": fp.exit_within_20, "median_holding_time": fp.median_exit_time}, stationarity_classification="ADF/KPSS pass" if stationarity else "ADF/KPSS gate failed", parameter_stability="AR1/MLE agreement" if ou_agreement else "Estimator disagreement", confidence_qualities={"stationarity_quality": float(stationarity), "ou_agreement": float(ou_agreement), "half_life_quality": float(primary.valid and primary.half_life <= 60), "parameter_stability": float(ou_agreement), "regime_quality": 1.0, "first_passage_quality": 0.0 if fp is None else fp.exit_before_stop, "economic_value_margin": 0.0, "boundary_robustness": 0.0})
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        st.error(f"Decision monitor cannot evaluate this input: {exc}")
        return
    with st.container(border=True):
        st.subheader("Current state")
        with st.container(horizontal=True):
            st.metric("Current price", _format(next(iter(final.current_prices.values())))); st.metric("Residual", _format(final.current_residual)); st.metric("Z-score", _format(final.current_zscore)); st.metric("OU half-life", _format(final.ou_half_life, 1)); st.metric("Position", final.position_state.value)
    with st.container(border=True):
        st.subheader("Final research action")
        st.markdown(f"## {final.research_action.value}")
        st.caption(final.invalid_reason or "All critical gates passed.")
    with st.container(border=True):
        st.subheader("Decision boundaries")
        st.write({"long entry": final.long_entry_region, "short entry": final.short_entry_region, "current residual": final.current_residual, "exit": final.exit_region, "stop": final.stop_region})
    with st.container(border=True):
        st.subheader("Probability / economics")
        st.write({"P(exit before stop)": final.p_exit_before_stop, "P(stop first)": final.p_stop_before_exit, "P(exit within 5d)": final.p_exit_5d, "P(exit within 10d)": final.p_exit_10d, "P(exit within 20d)": final.p_exit_20d, "Expected economic value": final.expected_economic_value})
    with st.container(border=True):
        st.subheader("Model validity")
        st.write({"Stationarity": final.stationarity_classification, "OU validity": primary.valid, "Half-life": final.ou_half_life, "Parameter stability": final.parameter_stability, "Regime": final.regime_status})
    with st.container(border=True):
        st.subheader("Consensus")
        st.write({"state": final.consensus["state"], "agreement": f"{final.agreement_percentage:.1f}%", "models": [(item["model_name"], item["direction"]) for item in final.consensus["evidences"]]})
    with st.container(border=True):
        st.subheader("Why this signal exists")
        st.dataframe(pd.DataFrame(final.why_signal["items"]), hide_index=True)
    st.session_state["latest_final_decision"] = final
    if st.button("Create new frozen prediction", key="create_frozen_prediction"):
        try:
            frozen = new_prediction_snapshot(target_ticker=target, current_price=float(prices[target].iloc[-1]), data_cutoff=final.model_cutoff, frequency=Frequency.DAILY, data_source="uploaded CSV", model_version="step-11", frozen_outputs=final.snapshot_outputs(), notes="Created intentionally from the Decision Monitor.")
            PredictionStore().save_snapshot(frozen)
            st.success(f"Created frozen prediction {frozen.prediction_id}.")
        except (TypeError, ValueError, OSError) as exc:
            st.error(f"Could not save frozen prediction: {exc}")


def _live_prediction_monitor() -> None:
    st.header("Live prediction monitor")
    st.caption("MODEL PARAMETERS ARE FROZEN. Live prices update only the observation; they never refit the model.")
    store = PredictionStore()
    identifiers = store.list_snapshots()
    if not identifiers:
        st.info("No saved predictions yet. Create one intentionally in Decision monitor.")
        return
    identifier = st.selectbox("Stored frozen prediction", identifiers, key="live_prediction_id")
    try:
        snapshot = store.load_snapshot(identifier)
    except ValueError as exc:
        st.error(str(exc)); return
    final = snapshot.model_outputs.get("final_decision", {})
    with st.container(border=True):
        st.subheader("Model frozen")
        st.write({"timestamp": snapshot.created_at.isoformat(), "data cutoff": snapshot.data_cutoff.isoformat(), "parameter hash": snapshot.parameter_hash})
    with st.container(border=True):
        st.subheader("Original model")
        st.write({"initial residual": final.get("current_residual"), "entry regions": (final.get("long_entry_region"), final.get("short_entry_region")), "exit": final.get("exit_region"), "stop": final.get("stop_region"), "action": final.get("research_action"), "P(exit before stop)": snapshot.probability_outputs.get("exit_before_stop"), "P(exit 5d)": snapshot.probability_outputs.get("exit_5d"), "P(exit 10d)": snapshot.probability_outputs.get("exit_10d"), "P(exit 20d)": snapshot.probability_outputs.get("exit_20d"), "expected economic value": final.get("expected_economic_value"), "confidence": snapshot.consensus_outputs.get("confidence")})
    pair_data = snapshot.model_outputs.get("pair")
    if not pair_data:
        st.info("This frozen model has no exact price mapping. Its stored residual and statistical boundaries remain the reference; do not fabricate a dollar target.")
        return
    try:
        model = FrozenPairModel(pair_data["target_ticker"], pair_data["peer_ticker"], float(pair_data["alpha"]), float(pair_data["beta"]), float(final["current_residual"]), tuple(final["long_entry_region"]) if final.get("long_entry_region") else None, tuple(final["short_entry_region"]) if final.get("short_entry_region") else None, float(final["exit_region"][0]) if final.get("exit_region") else None, float(final["stop_region"][0]) if final.get("stop_region") else None, float(final["stop_region"][1]) if final.get("stop_region") else None)
        with st.form("live_pair_prices"):
            target_price = float(st.number_input(f"Live {model.target_ticker} price", min_value=.000001, value=float(snapshot.current_price)))
            peer_price = float(st.number_input(f"Live {model.peer_ticker} price", min_value=.000001, value=100.0))
            elapsed = int(st.number_input("Elapsed trading days", min_value=0, value=0, step=1))
            position = st.selectbox("Original position state", ["FLAT", "LONG_RESIDUAL", "SHORT_RESIDUAL"])
            calculate = st.form_submit_button("Evaluate live observation")
        if not calculate: return
        live = classify_live_pair(model, target_price=target_price, peer_price=peer_price, position=position)
        with st.container(border=True):
            st.subheader("Live market")
            st.write({"current frozen-model residual": live.current_residual, "distance to entry": live.distance_to_entry, "distance to exit": live.distance_to_exit, "distance to stop": live.distance_to_stop, "current region": live.state.value, "fraction reverted": live.fraction_reverted, "implied target prices": live.implied_target_prices})
        if st.button("Save separate evaluation", key="save_live_evaluation"):
            evaluation = make_live_evaluation(snapshot, timestamp=datetime.now(timezone.utc), live_prices={model.target_ticker: target_price, model.peer_ticker: peer_price}, live_state=live, elapsed_trading_days=elapsed)
            store.save_evaluation(evaluation)
            st.success("Saved a separate evaluation; the frozen prediction remains unchanged.")
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"Stored pair model cannot be evaluated: {exc}")


def _backtest_research(prices: pd.DataFrame) -> None:
    st.header("Backtest research")
    st.caption("Strict walk-forward research: signals use information through close t and execute at close t+1. Results are not investment recommendations.")
    st.warning("Parameter sweeps are descriptive only. Do not select settings by maximum Sharpe, final wealth, or holdout performance without explicit precommitment. Multiple testing, data snooping, survivorship, lookahead, and selection bias remain material risks.")
    with st.form("backtest_research"):
        target=st.selectbox("Backtest target", list(prices.columns))
        window=int(st.number_input("Training window", min_value=20, value=min(60,max(20,len(prices)//2)), step=1))
        entry=float(st.number_input("Benchmark entry |z|", min_value=.1, value=2.0, step=.1)); exit=float(st.number_input("Benchmark exit |z|", min_value=.0, value=.5, step=.1)); stop=float(st.number_input("Benchmark stop |z|", min_value=.1, value=4.0, step=.1))
        submitted=st.form_submit_button("Run chronological benchmark research")
    if not submitted: return
    try: output=run_backtest(prices[target],config=BacktestConfig(BenchmarkConfig(window,entry,exit,stop)))
    except (TypeError,ValueError) as exc: st.error(f"Backtest cannot run: {exc}"); return
    result=output["benchmark"]
    with st.container(border=True):
        st.subheader("Benchmark results — in-sample / validation / out-of-sample must be configured chronologically before interpreting results")
        st.dataframe(pd.DataFrame([result.metrics]),hide_index=True)
    with st.container(border=True): st.subheader("Event study"); st.dataframe(output["events"],hide_index=True)
    with st.container(border=True): st.subheader("Equity / P&L progression and drawdown"); st.line_chart(pd.DataFrame({"Equity":result.equity,"Drawdown":result.drawdown}))
    with st.container(border=True): st.subheader("Calibration"); st.dataframe(result.calibration,hide_index=True); st.subheader("Bucket analysis"); st.dataframe(result.buckets,hide_index=True)
    st.info("Research questions are intentionally unanswered until supported by walk-forward evidence: residual filtering versus benchmark, half-life, gate quality, stopping after costs, consensus, and probability calibration.")


def _data_workspace() -> None:
    st.header("Data workspace")
    st.caption("Upload a CSV, inspect its raw layout, normalize deliberately, then choose model roles. Uploaded files stay in this browser session and are not saved to the repository.")
    upload=st.file_uploader("Upload wide or Bloomberg-style CSV",type=["csv"],key="data_workspace_upload")
    if upload is None: return
    try: raw=pd.read_csv(BytesIO(upload.getvalue()))
    except Exception as exc: st.error(f"CSV could not be read: {exc}"); return
    st.subheader("1. Raw preview"); st.dataframe(raw.head(20),hide_index=True)
    detected=detect_csv_format(raw); st.subheader("2. Format detection"); st.write("Wide format" if detected=="wide" else "Repeated date/price pairs — explicit mapping required")
    normalized=None
    if detected=="wide":
        if st.button("Normalize wide CSV",key="normalize_wide"):
            normalized=normalize_bloomberg_frame(raw)
    else:
        count=int(st.number_input("Number of series to map",min_value=1,max_value=max(1,len(raw.columns)//2),value=1,step=1))
        with st.form("repeated_pair_mapping"):
            mappings=[]
            for i in range(count):
                st.markdown(f"Series {i+1}")
                ticker=st.text_input("Ticker",key=f"mapping_ticker_{i}")
                date=st.selectbox("Date column",list(raw.columns),key=f"mapping_date_{i}")
                price=st.selectbox("Price column",list(raw.columns),key=f"mapping_price_{i}")
                mappings.append(SeriesMapping(ticker,date,price))
            normalize=st.form_submit_button("Normalize mapped series")
        if normalize: normalized=normalize_bloomberg_frame(raw,mappings=mappings)
    if normalized is not None:
        try: normalized=validate_price_frame(normalized)
        except ValueError as exc: st.error(f"Normalization completed, but prices are not model-safe: {exc}"); return
        st.session_state["uploaded_prices"]=normalized
        st.success("Normalized data is ready for this session. No missing values were filled.")
    prices=st.session_state.get("uploaded_prices")
    if prices is None: return
    st.subheader("3. Normalized data and quality")
    st.dataframe(prices.head(20)); report=data_quality_report(prices); st.json(report.to_dict())
    st.subheader("4. Model readiness"); st.dataframe(model_readiness_report(prices).to_frame(),hide_index=True)
    st.subheader("5. Roles and universes")
    columns=list(prices.columns); defaults_market=columns.index("SPY") if "SPY" in columns else 0
    with st.form("data_roles"):
        target=st.selectbox("Target ticker",columns)
        market=st.selectbox("Market factor",columns,index=defaults_market)
        sector=st.selectbox("Sector factor (optional)",["None"]+columns)
        peers=st.multiselect("Peer universe",columns,default=[ticker for ticker in columns if ticker!=target])
        pca=st.multiselect("PCA universe (manually exclude ETFs/factors if desired)",columns,default=columns)
        confirm=st.form_submit_button("Confirm roles")
    if confirm: st.session_state["data_roles"]={"target":target,"market":market,"sector":None if sector=="None" else sector,"peers":peers,"pca":pca}; st.success("Roles confirmed. Continue to Decision Monitor when ready.")
    clean=prices.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download clean normalized CSV",clean,file_name="normalized_prices.csv",mime="text/csv")


def _learn_models() -> None:
    st.header("Learn the models")
    st.caption("Plain-language reference for the research workstation. None of these concepts establishes a profitable or guaranteed outcome.")
    topics={
        "Z-score and detrending":"A z-score is a deviation from a prior statistical baseline measured in prior sample standard deviations. Detrending first removes a fitted historical log-price trend; its residual is not fundamental fair value.",
        "Factor residuals and cointegration":"A factor residual is a target return less a model-implied market/sector return. Cointegration tests a long-run pair relationship; correlation alone is not cointegration.",
        "Kalman and PCA":"A Kalman filter permits a pair alpha/beta to evolve through time. PCA reconstructs shared return variation from a training window; the remaining residual is model-specific.",
        "Stationarity, ADF, and KPSS":"ADF and KPSS are complementary diagnostics with assumptions and finite-sample limits. A gate can reject a state when the diagnostics conflict; it does not prove stationarity.",
        "OU and half-life":"The OU state is dX = κ(θ−X)dt + σdW. κ is per trading day, σ per square-root trading day, and ln(2)/κ is expected-displacement half-life—not a realized boundary-hitting time.",
        "First passage and optimal stopping":"First passage estimates which boundary a simulated OU path hits first. The finite-horizon policy compares explicit entry, hold, exit, and stop values after costs; it can create bounded research entry regions.",
        "Costs and consensus":"Basis-point spread/slippage and fixed costs enter the economic decision before action selection. Consensus records agreement among valid models while limiting duplicate evidence from dependent models.",
    }
    for title,text in topics.items():
        with st.expander(title): st.write(text)


def main() -> None:
    st.set_page_config(page_title="Mean Reversion Research Engine", layout="wide")
    st.title("Mean Reversion Research Engine")
    st.caption("Step 4 research foundation — educational quantitative research; not investment advice.")
    data_workspace, decision_monitor, single_stock, pairs, ou_dynamics, model_validity, live_monitor, backtest, learn, landing = st.tabs(["Data workspace", "Decision monitor", "Single stock models", "Pairs research", "OU dynamics", "Model validity", "Live prediction monitor", "Backtest research", "Learn the models", "Project overview"])
    with data_workspace:
        _data_workspace()
    with decision_monitor:
        prices = st.session_state.get("uploaded_prices")
        if prices is None:
            st.info("Upload and validate a CSV in Project overview first.")
        else:
            _decision_monitor(prices)
    with live_monitor:
        _live_prediction_monitor()
    with backtest:
        prices=st.session_state.get("uploaded_prices")
        if prices is None: st.info("Upload and validate a CSV in Project overview first.")
        else: _backtest_research(prices)
    with landing:
        st.info("Implemented: validated CSV inputs, immutable prediction-journal infrastructure, prior-window residual research models, rolling pair diagnostics, and reusable OU dynamics estimation.")
        st.subheader("CSV workflow")
        st.write("Use Data workspace to upload, normalize, review quality, select roles, and download a clean session-only CSV.")
        st.subheader("Prediction journal")
        st.write("A forward prediction snapshot is frozen at its data cutoff. Later realized outcomes are stored separately, preventing historical predictions from being rewritten.")
        st.subheader("Roadmap")
        st.write("Current scope is Step 4: residual research, static pairs, and OU dynamics. See `docs/ROADMAP.md` for later work.")
    with single_stock:
        prices = st.session_state.get("uploaded_prices")
        if prices is None:
            st.info("Upload and validate a CSV in Project overview first.")
        elif len(prices.columns) < 2:
            st.warning("At least a target and market ticker are required.")
        else:
            _single_stock_models(prices)
    with pairs:
        prices = st.session_state.get("uploaded_prices")
        if prices is None:
            st.info("Upload and validate a CSV in Project overview first.")
        elif len(prices.columns) < 2:
            st.warning("At least two ticker columns are required.")
        else:
            _pairs_research(prices)
    with ou_dynamics:
        prices = st.session_state.get("uploaded_prices")
        if prices is None:
            st.info("Upload and validate a CSV in Project overview first.")
        elif len(prices.columns) < 2:
            st.warning("At least two ticker columns are required.")
        else:
            _ou_dynamics(prices)
    with model_validity:
        prices=st.session_state.get("uploaded_prices")
        if prices is None: st.info("Upload and normalize data in Data workspace first.")
        else:
            st.header("Model validity")
            st.dataframe(model_readiness_report(prices).to_frame(),hide_index=True)
            st.caption("Readiness is a data-length check, not validation that a residual will converge.")
    with learn:
        _learn_models()


if __name__ == "__main__":
    main()
