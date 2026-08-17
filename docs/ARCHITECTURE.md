# Architecture

```mermaid
flowchart TD
  D["CSV data"] --> N["Normalization and quality report"] --> R["Residual engines"]
  R --> G["Diagnostics and hard gates"] --> O["OU dynamics"]
  O --> F["First passage"] --> S["Optimal stopping"] --> C["Consensus and confidence"]
  C --> FD["Final research decision"] --> FP["Frozen prediction"] --> LE["Live evaluation"]
  R --> WF["Chronological walk-forward event study"]
  C --> WF
  WF --> BR["Benchmark versus full-model research results"]
```

Data normalization outer-merges explicit repeated-pair mappings on observed dates without filling values. Residual engines own their model-specific state; diagnostics and hard gates decide whether downstream OU/stopping research is eligible. A `PredictionSnapshot` freezes all model-time outputs and a `PredictionEvaluation` records later observations separately. The walk-forward branch creates a signal from data through close *t* and executes it at close *t+1*.
