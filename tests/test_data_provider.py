import pytest

from src.data.csv_provider import CSVDataProvider


def test_csv_loads_and_sorts(csv_path) -> None:
    bundle = CSVDataProvider().load(csv_path)
    assert bundle.tickers == ("AAA", "BBB")
    assert bundle.prices.index.is_monotonic_increasing
    assert bundle.number_of_observations == 2


@pytest.mark.parametrize("value, message", [("0", "strictly positive"), ("-1", "strictly positive"), ("not-a-number", "nonnumeric")])
def test_invalid_price_values_rejected(tmp_path, value, message) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(f"Date,AAA\n2025-01-02,{value}\n2025-01-03,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        CSVDataProvider().load(path)


def test_duplicate_dates_rejected(tmp_path) -> None:
    path = tmp_path / "duplicate.csv"
    path.write_text("Date,AAA\n2025-01-02,10\n2025-01-02,11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        CSVDataProvider().load(path)


def test_missing_values_are_reported(tmp_path) -> None:
    path = tmp_path / "missing.csv"
    path.write_text("Date,AAA,BBB\n2025-01-02,10,20\n2025-01-03,11,\n", encoding="utf-8")
    bundle = CSVDataProvider().load(path)
    assert bundle.missing_values_by_ticker == {"AAA": 0, "BBB": 1}
    assert bundle.number_of_complete_observations == 1

