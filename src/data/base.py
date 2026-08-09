"""Provider contract. Future sources must return the same internal bundle."""
from pathlib import Path
from typing import Protocol

from src.schemas import MarketDataBundle


class MarketDataProvider(Protocol):
    source_name: str

    def load(self, path: str | Path) -> MarketDataBundle:
        """Load a source into a validated, aligned MarketDataBundle."""


class BloombergDataProvider:
    """Non-operational placeholder for a future licensed Bloomberg adapter."""
    source_name = "Bloomberg (not implemented)"

    def load(self, path: str | Path) -> MarketDataBundle:
        raise NotImplementedError(
            "BloombergDataProvider is intentionally not available in Step 1; use CSVDataProvider."
        )

