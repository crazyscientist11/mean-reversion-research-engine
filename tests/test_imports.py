import importlib


def test_core_imports() -> None:
    importlib.import_module("src")
    importlib.import_module("src.data")
    importlib.import_module("src.prediction")


def test_streamlit_entry_point_imports() -> None:
    importlib.import_module("mean_reversion_engine")
