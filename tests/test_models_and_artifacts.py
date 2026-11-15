"""
Test suite for serialized model artifacts, probability ranges, and inference consistency.
"""
import os
import joblib
import pandas as pd
import numpy as np
import pytest
from src.models.pipeline import ALL_MODEL_FEATURES

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "test.csv")

def test_model_artifacts_exist():
    expected_files = [
        'best_model.joblib',
        'baseline_model.joblib',
        'logistic_model.joblib',
        'rf_model.joblib',
        'hgb_model.joblib',
        'model_metadata.json',
        'evaluation_metrics.json'
    ]
    for fname in expected_files:
        path = os.path.join(ARTIFACTS_DIR, fname)
        assert os.path.exists(path), f"Artifact {fname} not found in {ARTIFACTS_DIR}"

def test_predicted_probabilities_within_unit_interval():
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "best_model.joblib"))
    df = pd.read_csv(TEST_DATA_PATH).head(50)
    X = df[ALL_MODEL_FEATURES]
    probs = model.predict_proba(X)[:, 1]
    
    assert (probs >= 0.0).all(), "Found negative probability!"
    assert (probs <= 1.0).all(), "Found probability exceeding 1.0!"
    assert len(probs) == len(df)

def test_model_handles_same_day_vs_long_lead_differently():
    """Validates that lead time has a direct positive correlation with risk score."""
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "best_model.joblib"))
    
    base_record = {
        'lead_days': 0,
        'is_same_day': 1,
        'lead_weeks': 0.0,
        'booking_hour': 10,
        'booking_weekday': 1,
        'booking_month': 5,
        'appointment_weekday': 1,
        'appointment_month': 5,
        'Age': 35,
        'age_group': 'YoungAdult',
        'gender_female': 1,
        'Scholarship': 0,
        'neighbourhood': 'CENTRO',
        'prior_appointments_count': 0,
        'prior_no_shows_count': 0,
        'prior_no_show_rate': 0.0,
        'is_repeat_patient': 0
    }
    
    row_same = pd.DataFrame([base_record])[ALL_MODEL_FEATURES]
    
    long_record = base_record.copy()
    long_record['lead_days'] = 30
    long_record['is_same_day'] = 0
    long_record['lead_weeks'] = 4.29
    row_long = pd.DataFrame([long_record])[ALL_MODEL_FEATURES]
    
    prob_same = model.predict_proba(row_same)[:, 1][0]
    prob_long = model.predict_proba(row_long)[:, 1][0]
    
    assert prob_long > prob_same, f"Expected long lead time to yield higher risk ({prob_long:.3f}) than same day ({prob_same:.3f})"
