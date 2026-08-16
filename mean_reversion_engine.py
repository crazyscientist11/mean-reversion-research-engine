"""Step 1 landing page plus Step 2 single-stock research outputs."""
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from src.data.validation import summarize_prices, validate_price_frame
from src.models import detrended_log_price_residual, factor_residual_model, rolling_zscore_benchmark
from src.models.common import ResidualModelResult
from src.models.pairs import PairModelResult, peer_comparison, rolling_pair_model
from src.models.ou import OUComparisonResult, OUFitResult, conditional_band, fit_ou_both
from src.consensus import EvidenceDirection, ModelEvidence, build_consensus
from src.costs import CostModel
from src.decision import CriticalGates, PositionState, ResearchAction, StoppingConfig, build_final_decision, build_policy, regions
from src.first_passage import BoundaryDefinition, ResidualDirection
from src.monte_carlo import simulate_ou_first_passage
from statsmodels.tsa.stattools import adfuller, kpss


def _format(value: float | None, decimals: int = 4) -> str:
    return "Not available" if value is None or not np.isfinite(value) else f"{value:.{decimals}f}"


def _uploaded_prices(upload: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    preview = pd.read_csv(BytesIO(upload.getvalue()))
    if "Date" not in preview.columns:
        raise ValueError("A Date column is required.")
    prices = preview.drop(columns=["Date"]).copy()
    prices.index = pd.DatetimeIndex(pd.to_datetime(preview["Date"], errors="raise"), name="Date")
    return validate_price_frame(prices.sort_index())


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


def main() -> None:
    st.set_page_config(page_title="Mean Reversion Research Engine", layout="wide")
    st.title("Mean Reversion Research Engine")
    st.caption("Step 4 research foundation — educational quantitative research; not investment advice.")
    decision_monitor, landing, single_stock, pairs, ou_dynamics = st.tabs(["Decision monitor", "Project overview", "Single stock models", "Pairs research", "OU dynamics"])
    with decision_monitor:
        prices = st.session_state.get("uploaded_prices")
        if prices is None:
            st.info("Upload and validate a CSV in Project overview first.")
        else:
            _decision_monitor(prices)
    with landing:
        st.info("Implemented: validated CSV inputs, immutable prediction-journal infrastructure, prior-window residual research models, rolling pair diagnostics, and reusable OU dynamics estimation.")
        st.subheader("CSV preview")
        upload = st.file_uploader("Upload a wide-form adjusted-price CSV (optional)", type=["csv"])
        if upload is not None:
            try:
                prices = _uploaded_prices(upload)
                st.dataframe(prices.head(20))
                summary = summarize_prices(prices, frequency="daily", source="uploaded CSV preview")
                st.subheader("Data summary")
                st.json({key: str(value) for key, value in summary.items()})
                st.session_state["uploaded_prices"] = prices
            except (TypeError, ValueError) as exc:
                st.error(f"CSV cannot be used: {exc}")
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


if __name__ == "__main__":
    main()
