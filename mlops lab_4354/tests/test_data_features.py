import pandas as pd
import sys
sys.path.insert(0, "src")
from features import engineer_features
from data_generation import create_records


def test_versions_have_enough_records():
    v1 = create_records("v1", 250, 4354)
    v2 = create_records("v2", 250, 4355)
    assert len(v1) == 250
    assert len(v2) == 250


def test_monsoon_conditions_increase_humidity():
    v1 = create_records("v1", 250, 4354)
    v2 = create_records("v2", 250, 4355)
    assert v2["relative_humidity_pct"].mean() > v1["relative_humidity_pct"].mean()


def test_feature_ranges():
    data = create_records("v1", 250, 4354)
    engineered = engineer_features(data)
    assert engineered["condensation_risk_index"].between(0, 1).all()
    assert engineered["ventilation_effectiveness_ratio"].ge(0).all()
