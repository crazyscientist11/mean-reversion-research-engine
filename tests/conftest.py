from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def prices() -> pd.DataFrame:
    return pd.DataFrame(
        {"AAA": [100.0, 110.0, 121.0], "BBB": [50.0, 55.0, 60.0]},
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "prices.csv"
    path.write_text("Date,AAA,BBB\n2025-01-03,110,55\n2025-01-02,100,50\n", encoding="utf-8")
    return path

