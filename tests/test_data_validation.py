"""
Test suite for data validation, schema integrity, and chronological split consistency.
"""
import os
import pandas as pd
import pytest

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_FILE = os.path.join(DATA_DIR, "raw", "noshowappointments-kagglev2-may-2016.csv")

def test_raw_dataset_exists():
    assert os.path.exists(RAW_FILE), f"Raw dataset missing at {RAW_FILE}"
    df = pd.read_csv(RAW_FILE, nrows=10)
    expected_cols = {'PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Age', 'No-show'}
    assert expected_cols.issubset(set(df.columns)), f"Missing expected raw columns: {expected_cols - set(df.columns)}"

def test_processed_splits_exist():
    for split in ['train.csv', 'val.csv', 'test.csv']:
        path = os.path.join(PROCESSED_DIR, split)
        assert os.path.exists(path), f"Processed split missing: {path}"

def test_target_is_strictly_binary():
    for split in ['train.csv', 'val.csv', 'test.csv']:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, split))
        assert 'target' in df.columns
        unique_targets = set(df['target'].unique())
        assert unique_targets.issubset({0, 1}), f"Invalid target values in {split}: {unique_targets}"

def test_no_negative_lead_days():
    for split in ['train.csv', 'val.csv', 'test.csv']:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, split))
        assert (df['lead_days'] >= 0).all(), f"Found negative lead_days in {split}"

def test_no_negative_age():
    for split in ['train.csv', 'val.csv', 'test.csv']:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, split))
        assert (df['Age'] >= 0).all(), f"Found negative age in {split}"

def test_chronological_split_ordering():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test.csv'))
    
    train_max_sched = pd.to_datetime(train_df['ScheduledDay']).max()
    val_min_sched = pd.to_datetime(val_df['ScheduledDay']).min()
    val_max_sched = pd.to_datetime(val_df['ScheduledDay']).max()
    test_min_sched = pd.to_datetime(test_df['ScheduledDay']).min()
    
    assert train_max_sched <= val_min_sched, "Chronological train/val leak detected!"
    assert val_max_sched <= test_min_sched, "Chronological val/test leak detected!"
