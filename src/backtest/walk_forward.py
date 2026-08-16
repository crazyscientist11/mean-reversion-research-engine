"""Chronological train/validation/test split with no random shuffling."""
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.Index; validation: pd.Index; test: pd.Index

def chronological_split(index: pd.Index, train_fraction: float=.6, validation_fraction: float=.2) -> ChronologicalSplit:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 or train_fraction+validation_fraction >= 1: raise ValueError("fractions must leave a non-empty test period")
    n=len(index); train_end=int(n*train_fraction); validation_end=int(n*(train_fraction+validation_fraction))
    if train_end<1 or validation_end<=train_end or validation_end>=n: raise ValueError("not enough observations for chronological split")
    return ChronologicalSplit(index[:train_end], index[train_end:validation_end], index[validation_end:])
