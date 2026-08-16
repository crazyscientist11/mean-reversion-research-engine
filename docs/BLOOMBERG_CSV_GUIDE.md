# Bloomberg CSV workflow

This project does not connect to Bloomberg and includes no Bloomberg data or dependencies. It accepts local CSV exports only. Use adjusted historical prices when that is appropriate for the research question, since corporate actions can otherwise create mechanical jumps that distort returns and residuals.

The preferred CSV is wide: `Date,XLE,XOM,CVX,...`. The Data Workspace also supports spreadsheet exports with one date/price pair per ticker. Because Bloomberg and Excel exports vary, repeated-pair layouts require you to explicitly select the ticker, date column, and price column for every series; the application does not guess. Series merge on exact observed dates and never forward-fill, backward-fill, or create weekend prices.

After upload, inspect the quality report, choose your target, market/sector factors, peer universe, and PCA universe. ETF and factor inclusion is a manual choice, not a hard-coded classification. Downloading a normalized CSV is optional and local to the browser session.

Export methods and field labels can differ across Bloomberg Terminal and Excel configurations. Do not commit exports: `data/private/`, `data/bloomberg/`, and `*.bbg.csv` are ignored to protect licensed data.
