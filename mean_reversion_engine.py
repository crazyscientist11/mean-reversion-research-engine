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


def main() -> None:
    st.set_page_config(page_title="Mean Reversion Research Engine", layout="wide")
    st.title("Mean Reversion Research Engine")
    st.caption("Step 4 research foundation — educational quantitative research; not investment advice.")
    landing, single_stock, pairs, ou_dynamics = st.tabs(["Project overview", "Single stock models", "Pairs research", "OU dynamics"])
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


