"""
Test suite for feature availability, ethical boundaries, and leakage guards.
"""
import pytest
from src.models.pipeline import ALL_MODEL_FEATURES

def test_sms_received_strictly_excluded():
    """Verifies that SMS_received is purged to avoid temporal data leakage."""
    assert 'SMS_received' not in ALL_MODEL_FEATURES
    assert 'sms_received' not in ALL_MODEL_FEATURES

def test_clinical_variables_strictly_excluded():
    """Verifies non-clinical boundary: no disease diagnoses or disability indicators."""
    forbidden = {'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'hipertension', 'diabetes', 'alcoholism', 'handcap'}
    for feature in ALL_MODEL_FEATURES:
        assert feature not in forbidden, f"Forbidden clinical variable found in feature set: {feature}"

def test_identifiers_strictly_excluded():
    """Verifies raw primary/foreign keys are excluded from modeling features."""
    assert 'PatientId' not in ALL_MODEL_FEATURES
    assert 'AppointmentID' not in ALL_MODEL_FEATURES

def test_required_operational_features_present():
    """Ensures essential scheduling dynamics are present."""
    required = {'lead_days', 'is_same_day', 'booking_hour', 'booking_weekday', 'appointment_weekday', 'Age', 'prior_appointments_count', 'prior_no_shows_count'}
    assert required.issubset(set(ALL_MODEL_FEATURES))
