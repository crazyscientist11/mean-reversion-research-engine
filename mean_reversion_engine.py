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
from src.models.pca_model import PCAResidualResult, rolling_pca_residuals
from src.models.cross_sectional import cross_sectional_residuals
from src.models.kalman_pair import kalman_pair_filter
from src.models.ou_pair_optimizer import optimize_ou_hedge_ratio
from src.first_passage import BoundaryDefinition, ResidualDirection
from src.monte_carlo import simulate_ou_first_passage
from src.costs import CostModel
from src.decision import DecisionGates, PositionState, ResearchAction, StoppingConfig, benchmark_action, build_policy, evaluate_policy, regions


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


@st.cache_data(show_spinner=False)
def _run_pca(prices: pd.DataFrame, window: int, components: int | float, state_window: int) -> PCAResidualResult:
    return rolling_pca_residuals(prices, window=window, n_components=components, residual_state_window=state_window)


def _model_metric_card(result: ResidualModelResult, labels: tuple[str, str, str]) -> None:
    snapshot = result.current_snapshot()
    kalman = kalman_pair_filter(prices, target_ticker=target, peer_ticker=peer)
    optimized = optimize_ou_hedge_ratio(prices, target_ticker=target, peer_ticker=peer, beta_min=-3.0, beta_max=3.0, grid_size=61)
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
        st.subheader("Pair construction comparison")
        comparison = pd.DataFrame({"method": ["OLS / cointegration regression", "OU likelihood optimized", "Kalman filtered"], "hedge_ratio": [snapshot.beta, optimized.optimal_beta, kalman.beta.iloc[-1]], "OU_log_likelihood": [None, optimized.objective, None], "warning": [None, "WEAKLY_IDENTIFIED_HEDGE_RATIO" if optimized.weakly_identified else None, None]})
        st.dataframe(comparison, hide_index=True)
        st.caption("OU optimization uses only the displayed training sample's likelihood; lowest half-life alone is not a selection rule.")
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


def _pca_cross_sectional(prices: pd.DataFrame) -> None:
    st.subheader("PCA / cross-sectional research")
    st.caption("PCA reconstructs common return movement from a prior training window. Cross-sectional extremes are research states, not reversal or trade signals.")
    tickers = list(prices.columns)
    with st.form("pca_cross_sectional"):
        target = st.selectbox("PCA target ticker", tickers)
        selection = st.selectbox("PCA component selection", ["Fixed component count", "Explained variance threshold"])
        with st.container(horizontal=True):
            fixed_components = int(st.number_input("Fixed component count", min_value=1, max_value=len(tickers), value=1, step=1))
            explained_threshold = float(st.number_input("Explained variance threshold", min_value=0.05, max_value=1.0, value=0.80, step=0.05))
            pca_window = int(st.number_input("PCA training window", min_value=3, value=252, step=1))
            state_window = int(st.number_input("Residual-state window", min_value=1, value=20, step=1))
        cross_threshold = float(st.number_input("Cross-sectional z-score threshold", min_value=0.1, value=2.0, step=0.1))
        submitted = st.form_submit_button("Run PCA / cross-sectional research")
    if not submitted:
        st.info("Select the universe settings and run the research models. No analysis runs automatically.")
        return
    components: int | float = fixed_components if selection == "Fixed component count" else explained_threshold
    try:
        pca = _run_pca(prices, pca_window, components, state_window)
        cross = cross_sectional_residuals(pca.residual_returns, minimum_universe_size=max(2, min(3, len(tickers))), zscore_threshold=cross_threshold)
    except (TypeError, ValueError) as exc:
        st.error(f"PCA inputs are invalid: {exc}")
        return
    date = prices.index[-1]
    if not bool(pca.valid_observations.loc[date]):
        st.warning(f"PCA is invalid for the current date: {pca.invalid_reasons.loc[date]}")
    with st.container(border=True):
        st.subheader("Current target PCA research output")
        with st.container(horizontal=True):
            st.metric("Actual target return", _format(pca.actual_returns.loc[date, target]))
            st.metric("PCA reconstructed return", _format(pca.reconstructed_returns.loc[date, target]))
            st.metric("Idiosyncratic residual", _format(pca.residual_returns.loc[date, target]))
            st.metric("Residual z-score", _format(pca.residual_zscores.loc[date, target]))
            st.metric("Accumulated residual state", _format(pca.accumulated_residual_state.loc[date, target]))
    explained_rows = pca.explained_variance_by_component.dropna(how="all")
    if not explained_rows.empty:
        with st.container(border=True):
            st.subheader("Explained variance by principal component")
            st.bar_chart(explained_rows.iloc[-1])
    with st.container(border=True):
        st.subheader("Target actual versus PCA reconstructed returns")
        st.line_chart(pd.DataFrame({"Actual return": pca.actual_returns[target], "PCA reconstructed return": pca.reconstructed_returns[target]}))
    with st.container(border=True):
        st.subheader("Target PCA residual history")
        st.line_chart(pca.residual_returns[target])
    with st.container(border=True):
        st.subheader("Target accumulated residual state")
        st.caption("Rolling sum of PCA residual returns; it is not a stationarity conclusion.")
        st.line_chart(pca.accumulated_residual_state[target])
    with st.container(border=True):
        st.subheader("Cross-sectional residual research table")
        st.caption("Time-series residual z-scores and cross-sectional z-scores answer different questions. Neither is a trade instruction.")
        st.dataframe(cross.current_table(), hide_index=True)

def _first_passage_analysis() -> None:
    st.subheader("First-passage analysis")
    st.caption("Uses supplied OU parameters only. Expected convergence is distinct from simulated first-boundary outcomes.")
    with st.form("first_passage"):
        direction=st.selectbox("Residual direction", ["LONG_RESIDUAL","SHORT_RESIDUAL"]); current=float(st.number_input("Current residual", value=0.0)); exit_boundary=float(st.number_input("Exit boundary", value=0.5)); stop_boundary=float(st.number_input("Stop boundary", value=-0.5)); theta=float(st.number_input("OU theta", value=0.0)); kappa=float(st.number_input("OU kappa per trading day", min_value=0.0001,value=0.1)); sigma=float(st.number_input("OU sigma", min_value=0.0,value=0.1)); paths=int(st.number_input("Simulation paths", min_value=100,value=5000,step=100)); submitted=st.form_submit_button("Run boundary simulation")
    if not submitted: return
    try:
        boundary=BoundaryDefinition(current,exit_boundary,stop_boundary,ResidualDirection(direction)); result=simulate_ou_first_passage(theta=theta,kappa=kappa,sigma=sigma,boundaries=boundary,number_paths=paths,maximum_horizon=20,seed=0)
    except ValueError as exc: st.error(str(exc)); return
    with st.container(border=True):
        with st.container(horizontal=True):
            st.metric("P(exit before stop)",_format(result.exit_before_stop)); st.metric("P(stop first)",_format(result.stop_before_exit)); st.metric("P(exit 5d / 10d / 20d)",f"{_format(result.exit_within_5)} / {_format(result.exit_within_10)} / {_format(result.exit_within_20)}"); st.metric("Monte Carlo SE",_format(result.monte_carlo_se_exit)); st.metric("Median exit time",_format(result.median_exit_time,1))
    st.line_chart(pd.DataFrame(result.sample_paths.T))
    st.caption("Sample paths use exact OU transitions. Boundary result records first crossing; a coarse-step tie is conservatively assigned to stop.")

def _trading_decision_research() -> None:
    st.subheader("Trading Decision Research")
    st.warning("Paper-inspired numerical optimal-stopping research output. Not personalized investment advice and not an analytical reproduction of Li (2015).")
    with st.form("decision"):
        position = PositionState(st.selectbox("Current position state", [state.value for state in PositionState]))
        current=float(st.number_input("Current residual state",value=-1.0)); zscore=float(st.number_input("Current residual z-score (benchmark only)",value=-2.0)); theta=float(st.number_input("Decision OU theta",value=0.0)); kappa=float(st.number_input("Decision OU kappa",min_value=.0001,value=.15)); sigma=float(st.number_input("Decision OU sigma",min_value=.0001,value=.2)); long_stop=float(st.number_input("Long residual stop",value=-2.5)); short_stop=float(st.number_input("Short residual stop",value=2.5)); entry_cost=float(st.number_input("Fixed entry cost",min_value=0.,value=.01)); exit_cost=float(st.number_input("Fixed exit cost",min_value=0.,value=.01)); spread_bps=float(st.number_input("Bid-ask cost (bps)",min_value=0.,value=0.)); slippage_bps=float(st.number_input("Slippage (bps)",min_value=0.,value=0.)); commission=float(st.number_input("Commission",min_value=0.,value=0.)); opportunity_rate=float(st.number_input("Annual opportunity rate",min_value=0.,value=0.)); paths=int(st.number_input("First-passage simulation paths",min_value=100,value=2000,step=100)); ou_gate=st.checkbox("OU validity gate",value=True); stationarity_gate=st.checkbox("Stationarity gate",value=True); regime_gate=st.checkbox("Regime gate",value=True); sufficient_data=st.checkbox("Sufficient-data gate",value=True); submitted=st.form_submit_button("Evaluate research policy")
    if not submitted:return
    config=StoppingConfig(theta=theta,kappa=kappa,sigma=sigma,grid_min=min(long_stop-1,theta-3*sigma),grid_max=max(short_stop+1,theta+3*sigma),long_stop=long_stop,short_stop=short_stop)
    costs=CostModel(fixed_entry_cost=entry_cost,fixed_exit_cost=exit_cost,bid_ask_bps=spread_bps,slippage_bps=slippage_bps,commission=commission,annual_opportunity_rate=opportunity_rate)
    policy=build_policy(config,costs); action=evaluate_policy(policy,current,position,DecisionGates(ou_gate,stationarity_gate,regime_gate,sufficient_data)); policy_action,policy_value,next_action,next_value=policy.action_at(current,position)
    with st.container(border=True):
        st.subheader("Final research action")
        with st.container(horizontal=True):
            st.metric("Gated action",action.value,border=True); st.metric("Policy action",policy_action.value,border=True); st.metric("Policy value",_format(policy_value),border=True); st.metric("Next-best action",next_action.value,border=True); st.metric("Value advantage",_format(policy_value-next_value),border=True); st.metric("Z-score benchmark",benchmark_action(zscore,position).value,border=True)
        st.write({"long entry regions":regions(policy.grid,policy.flat_actions,ResearchAction.ENTER_LONG),"short entry regions":regions(policy.grid,policy.flat_actions,ResearchAction.ENTER_SHORT)})
        st.write({"OU validity":"PASS" if ou_gate else "FAIL","Stationarity":"PASS" if stationarity_gate else "FAIL","Regime":"PASS" if regime_gate else "FAIL","Data":"PASS" if sufficient_data else "FAIL","Economic value":"PASS" if policy_value>0 else "FAIL / WAIT"})
    if action in {ResearchAction.ENTER_LONG, ResearchAction.HOLD_LONG}:
        boundary=BoundaryDefinition(current,theta,long_stop,ResidualDirection.LONG_RESIDUAL)
    elif action in {ResearchAction.ENTER_SHORT, ResearchAction.HOLD_SHORT}:
        boundary=BoundaryDefinition(current,theta,short_stop,ResidualDirection.SHORT_RESIDUAL)
    else:
        boundary=None
    if boundary is not None:
        result=simulate_ou_first_passage(theta=theta,kappa=kappa,sigma=sigma,boundaries=boundary,number_paths=paths,maximum_horizon=20,seed=0)
        with st.container(border=True):
            st.subheader("Selected-policy first-passage check")
            with st.container(horizontal=True):
                st.metric("P(exit first)",_format(result.exit_before_stop),border=True); st.metric("P(stop first)",_format(result.stop_before_exit),border=True); st.metric("P(exit within 5d / 10d / 20d)",f"{_format(result.exit_within_5)} / {_format(result.exit_within_10)} / {_format(result.exit_within_20)}",border=True); st.metric("Monte Carlo SE",_format(result.monte_carlo_se_exit),border=True)
            st.caption("This first-passage distribution is shown separately from the expected-value policy and does not replace it.")


def main() -> None:
    st.set_page_config(page_title="Mean Reversion Research Engine", layout="wide")
    st.title("Mean Reversion Research Engine")
    st.caption("Step 5 research foundation — educational quantitative research; not investment advice.")
    landing, single_stock, pairs, ou_dynamics, pca_cross_sectional, first_passage, decision = st.tabs(["Project overview", "Single stock models", "Pairs research", "OU dynamics", "PCA / Cross-Sectional", "First-Passage Analysis", "Trading Decision Research"])
    with landing:
        st.info("Implemented: validated CSV inputs, immutable prediction-journal infrastructure, prior-window residual research models, pair/OU diagnostics, PCA residual reconstruction, and cross-sectional research states.")
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
        st.write("Current scope is Step 5: residual research, static pairs, OU dynamics, PCA, and cross-sectional comparisons. See `docs/ROADMAP.md` for later work.")
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
    with pca_cross_sectional:
        prices = st.session_state.get("uploaded_prices")
        if prices is None:
            st.info("Upload and validate a CSV in Project overview first.")
        elif len(prices.columns) < 2:
            st.warning("At least two ticker columns are required.")
        else:
            _pca_cross_sectional(prices)
    with first_passage:
        _first_passage_analysis()
    with decision:
        _trading_decision_research()


if __name__ == "__main__":
    main()
