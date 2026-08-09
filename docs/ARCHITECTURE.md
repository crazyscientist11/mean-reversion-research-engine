# Architecture

```mermaid
flowchart TD
  P["Data providers"] --> C["CSV (implemented)"]
  P --> B["Future Bloomberg adapter"]
  C --> M["MarketDataBundle (implemented)"]
  B --> M
  M --> R["Residual engines (future)"] --> D["Diagnostics (future)"] --> O["OU layer (future)"]
  O --> F["First-passage layer (future)"] --> S["Optimal-stopping layer (future)"] --> K["Consensus (future)"]
  K --> H["Historical backtest (future)"]
  K --> FP["Forward prediction (future)"] --> J["Prediction journal (implemented)"] --> E["Future evaluation"]
```

The implemented CSV provider yields a validated `MarketDataBundle` independent of future source adapters. The JSON prediction journal is deliberately separate from model logic: snapshot files capture information available at creation; later evaluation files never alter them.

Future residual engines can include rolling z-score, detrended log price, market/sector, static and dynamic pairs, PCA, cross-sectional, and basket approaches. OU is a downstream dynamics layer—not an independent equilibrium engine.

