"""Validated configuration for the research workstation."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Frequency(str, Enum):
    DAILY = "daily"
    HOURLY = "hourly"
    INTRADAY = "intraday"


@dataclass(frozen=True)
class DataParameters:
    frequency: Frequency = Frequency.DAILY
    date_column: str = "Date"
    minimum_observations: int = 2
    drop_incomplete_rows: bool = False
    require_positive_prices: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.frequency, Frequency):
            raise ValueError("frequency must be a Frequency value")
        if not self.date_column.strip():
            raise ValueError("date_column cannot be empty")
        if self.minimum_observations < 2:
            raise ValueError("minimum_observations must be at least 2")


@dataclass(frozen=True)
class ResearchWindowParameters:
    short_window: int = 60
    medium_window: int = 126
    annual_window: int = 252
    long_window: int = 504

    def __post_init__(self) -> None:
        windows = (self.short_window, self.medium_window, self.annual_window, self.long_window)
        if any(window <= 0 for window in windows):
            raise ValueError("research windows must be positive")
        if list(windows) != sorted(windows):
            raise ValueError("research windows must be non-decreasing")


@dataclass(frozen=True)
class PredictionStorageParameters:
    storage_directory: Path = Path("predictions/private")
    allow_overwrite: bool = False

