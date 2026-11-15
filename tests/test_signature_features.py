"""
Test suite for Signature Feature 1 (Simulator) and Signature Feature 2 (Threshold Studio).
"""
import os
import pandas as pd
import numpy as np
import pytest
from src.evaluation.simulator import AppointmentOperationsSimulator
from src.threshold.tradeoff import ThresholdTradeoffStudio

TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "test.csv")

def test_operations_simulator_capacity_constraint():
    sim = AppointmentOperationsSimulator()
    df = pd.read_csv(TEST_DATA_PATH).head(100)
    
    capacity = 25
    res = sim.simulate_batch(df, daily_capacity=capacity)
    
    assert res['metrics']['appointments_reviewed'] == capacity
    assert len(res['selected_records']) == capacity
    assert len(res['remaining_records']) == (len(df) - capacity)
    # Check ranking is strictly descending
    risks = res['selected_records']['predicted_risk'].values
    assert (np.diff(risks) <= 0).all(), "Selected records are not sorted in descending risk order!"

def test_threshold_tradeoff_monotonicity():
    studio = ThresholdTradeoffStudio()
    
    low_res = studio.evaluate_threshold(0.15, dataset="validation")
    high_res = studio.evaluate_threshold(0.75, dataset="validation")
    
    # Lower threshold must produce >= Recall than higher threshold
    assert low_res['recall'] >= high_res['recall']
    # Higher threshold must flag <= volume than lower threshold
    assert high_res['flagged_for_review'] <= low_res['flagged_for_review']
    
    # Check confusion matrix consistency: TP + FP + TN + FN == Total
    cm = low_res['confusion_matrix']
    total_cm = cm['tp'] + cm['fp'] + cm['tn'] + cm['fn']
    assert total_cm == low_res['appointments_total']
