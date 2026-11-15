"""
Signature Feature 1: Appointment Operations Simulator for PULSE.

Simulates real clinic operations under capacity-constrained human review.
Ranks scheduled appointments by model-predicted risk, flags top K appointments
for proactive staff outreach, and calculates capture rates, precision, lift,
and operational cost savings on held-out historical data.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.pipeline import ALL_MODEL_FEATURES

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts", "best_model.joblib")

class AppointmentOperationsSimulator:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Train models first.")
        self.model = joblib.load(model_path)
        self.feature_columns = ALL_MODEL_FEATURES
        
    def simulate_batch(self, batch_df: pd.DataFrame, daily_capacity: int = 50, cost_per_noshow: float = 100.0, cost_per_call: float = 5.0) -> dict:
        """
        Runs operational review simulation for a batch of scheduled appointments.
        
        Parameters:
        -----------
        batch_df: pd.DataFrame containing appointment records with ALL_MODEL_FEATURES.
        daily_capacity: int, max number of appointments staff can review/call.
        cost_per_noshow: float, clinic revenue lost per unannounced empty slot (default $100).
        cost_per_call: float, administrative cost per outbound human review call (default $5).
        
        Returns:
        --------
        dict containing ranked dataframe, selected records, unflagged records, and performance metrics.
        """
        df = batch_df.copy()
        
        # Ensure all required features exist
        for col in self.feature_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required feature for simulation: {col}")
                
        # Generate real model risk probabilities
        X = df[self.feature_columns]
        df['predicted_risk'] = self.model.predict_proba(X)[:, 1]
        
        # Risk Tiers for staff guidance
        # High: >= 0.40, Medium: 0.20 - 0.40, Low: < 0.20
        df['risk_tier'] = pd.cut(
            df['predicted_risk'],
            bins=[-0.01, 0.20, 0.40, 1.0],
            labels=['Low Risk', 'Moderate Risk', 'High Risk']
        ).astype(str)
        
        # Recommended Operational Action
        df['recommended_action'] = np.where(
            df['predicted_risk'] >= 0.40,
            "Priority Outreach: Personal Phone Call & Confirm Slot",
            np.where(
                df['predicted_risk'] >= 0.20,
                "Standard Outreach: Automated Multi-channel SMS/Email",
                "Routine: Standard Automated Reminder"
            )
        )
        
        # Rank descending by predicted risk
        df_ranked = df.sort_values(by='predicted_risk', ascending=False).reset_index(drop=True)
        df_ranked['rank'] = df_ranked.index + 1
        
        # Apply capacity constraint K
        k = min(daily_capacity, len(df_ranked))
        selected_for_review = df_ranked.iloc[:k].copy()
        remaining_appointments = df_ranked.iloc[k:].copy()
        
        # Calculate operational metrics
        total_appointments = len(df_ranked)
        has_ground_truth = 'target' in df_ranked.columns
        
        metrics = {
            "total_appointments": total_appointments,
            "daily_capacity_set": daily_capacity,
            "appointments_reviewed": k,
            "review_coverage_pct": round((k / total_appointments) * 100, 2),
            "unreviewed_count": len(remaining_appointments),
            "has_ground_truth": has_ground_truth
        }
        
        if has_ground_truth:
            total_actual_noshows = int(df_ranked['target'].sum())
            base_rate = total_actual_noshows / total_appointments if total_appointments > 0 else 0.0
            
            captured_noshows = int(selected_for_review['target'].sum())
            capture_rate = (captured_noshows / total_actual_noshows * 100) if total_actual_noshows > 0 else 0.0
            precision_in_k = (captured_noshows / k * 100) if k > 0 else 0.0
            lift = (precision_in_k / (base_rate * 100)) if base_rate > 0 else 1.0
            
            # Simulated Financial Impact
            # Baseline cost with 0 calls = all no-shows are missed
            cost_baseline = total_actual_noshows * cost_per_noshow
            # With staff outreach:
            # Assume human phone call recovers 30% of contacted no-shows (patient confirms or reschedules early)
            recovery_rate = 0.30
            recovered_noshows = captured_noshows * recovery_rate
            cost_with_simulation = (total_actual_noshows - recovered_noshows) * cost_per_noshow + (k * cost_per_call)
            net_savings = cost_baseline - cost_with_simulation
            
            metrics.update({
                "total_actual_noshows": total_actual_noshows,
                "base_noshow_rate_pct": round(base_rate * 100, 2),
                "captured_noshows_in_top_k": captured_noshows,
                "capture_rate_pct": round(capture_rate, 2),
                "precision_in_top_k_pct": round(precision_in_k, 2),
                "lift_over_random": round(lift, 2),
                "estimated_net_savings_usd": round(net_savings, 2),
                "assumed_phone_recovery_rate_pct": round(recovery_rate * 100, 1)
            })
            
        return {
            "metrics": metrics,
            "selected_records": selected_for_review,
            "remaining_records": remaining_appointments,
            "ranked_all": df_ranked
        }

if __name__ == "__main__":
    test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "test.csv")
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path).head(200)  # Simulate a daily schedule batch of 200 appointments
        sim = AppointmentOperationsSimulator()
        res = sim.simulate_batch(df_test, daily_capacity=40)
        print("\n--- SIMULATION BENCHMARK (Batch of 200, Capacity = 40) ---")
        for k, v in res['metrics'].items():
            print(f"  {k}: {v}")
