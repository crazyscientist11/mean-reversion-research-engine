import pandas as pd
from src.data.quality import data_quality_report, model_readiness_report

def test_quality_report_flags_missing_nonpositive_and_jumps_without_changing_data():
    frame=pd.DataFrame({"A":[1.,2.,0.,10.],"B":[1.,None,1.,1.]},index=pd.to_datetime(["2024-01-03","2024-01-01","2024-01-02","2024-01-04"]))
    report=data_quality_report(frame)
    assert report.unsorted_dates and report.nonpositive_prices["A"]==1 and report.missing_values_by_ticker["B"]==1
    assert report.large_one_day_jumps["A"]>=1

def test_readiness_explains_insufficient_data():
    frame=pd.DataFrame({"A":range(20),"B":range(20)},index=pd.date_range("2024-01-01",periods=20))
    report=model_readiness_report(frame)
    assert report.checks["Pair 252-day model"]["status"]=="insufficient"
