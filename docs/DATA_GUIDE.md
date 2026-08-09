# Data guide

Step 1 accepts wide adjusted-price CSV files: `Date,AAPL,MSFT,...`. Dates must be unique; prices are numeric and, by default, strictly positive. The loader sorts dates, reports missing data, and can explicitly drop incomplete rows. It never forward-fills, backfills, or mutates a supplied DataFrame.

## Using the Project Alongside Bloomberg

1. Export historical adjusted daily data to a private local CSV.
2. Load it through `CSVDataProvider`.
3. Generate a frozen end-of-day `PredictionSnapshot`.
4. Observe future Bloomberg prices without modifying that snapshot, then create `PredictionEvaluation` records.
5. A future optional `BloombergDataProvider` may replace the manual export while producing the same `MarketDataBundle`.

Bloomberg integration does not currently exist. Do not include proprietary data or credentials in this repository.

## Frequency discipline

Daily is the initial frequency. The schema reserves hourly and intraday labels, but no intraday model is implemented. Do not combine frequencies without explicit resampling. A daily model fit through close *t* is frozen; hourly or later prices may only be evaluated against it, not silently used to refit it.

