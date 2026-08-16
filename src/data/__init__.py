"""Market-data providers, validation, and non-mutating transforms."""
from .csv_provider import CSVDataProvider
from .base import MarketDataProvider

__all__ = ["CSVDataProvider", "MarketDataProvider"]
from .bloomberg_csv import detect_csv_format, load_bloomberg_csv, normalize_bloomberg_frame
from .normalization import SeriesMapping, normalize_repeated_pairs
from .quality import DataQualityReport, ModelReadinessReport, data_quality_report, model_readiness_report
