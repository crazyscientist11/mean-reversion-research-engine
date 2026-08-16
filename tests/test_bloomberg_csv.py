import pandas as pd
import pytest
from src.data.bloomberg_csv import detect_csv_format, normalize_bloomberg_frame
from src.data.normalization import SeriesMapping

def test_wide_csv_loads():
    raw=pd.DataFrame({"Date":["2024-01-01","2024-01-02"],"XLE":[10,11],"XOM":[20,21]})
    result=normalize_bloomberg_frame(raw)
    assert list(result.columns)==["XLE","XOM"] and result.index[0]==pd.Timestamp("2024-01-01")

def test_repeated_pairs_require_mapping_and_exact_merge_has_no_fill():
    raw=pd.DataFrame({"XLE Date":["2024-01-01","2024-01-03"],"XLE PX_LAST":[10,12],"XOM Date":["2024-01-02","2024-01-03"],"XOM PX_LAST":[20,21]})
    assert detect_csv_format(raw)=="repeated_pairs_requires_mapping"
    with pytest.raises(ValueError,match="require explicit"): normalize_bloomberg_frame(raw)
    result=normalize_bloomberg_frame(raw,mappings=[SeriesMapping("XLE","XLE Date","XLE PX_LAST"),SeriesMapping("XOM","XOM Date","XOM PX_LAST")])
    assert result.loc[pd.Timestamp("2024-01-01"),"XOM"] != result.loc[pd.Timestamp("2024-01-01"),"XOM"]
    assert len(result)==3

def test_duplicate_series_dates_are_rejected():
    raw=pd.DataFrame({"a":["2024-01-01","2024-01-01"],"p":[1,2]})
    with pytest.raises(ValueError,match="duplicate"): normalize_bloomberg_frame(raw,mappings=[SeriesMapping("A","a","p")])
