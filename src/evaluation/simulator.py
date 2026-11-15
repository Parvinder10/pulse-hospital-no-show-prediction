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
        
    def simulate_batch(self, batch_df: pd.DataFrame, daily_capacity: int = 50, cost_per_noshow: float = 100.0, cost_per_call: float = 5.0, phone_recovery_rate: float = 0.30) -> dict:
        """
        Runs operational review simulation for a batch of scheduled appointments.
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
        df_ranked['selected_for_review'] = df_ranked['rank'] <= k
        
        selected_for_review = df_ranked.iloc[:k].copy()
        remaining_appointments = df_ranked.iloc[k:].copy()
        
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
            # Human-readable labels for audit display
            df_ranked['actual_status'] = np.where(df_ranked['target'] == 1, "Missed (No-Show)", "Attended")
            df_ranked['review_outcome'] = np.where(
                df_ranked['selected_for_review'],
                np.where(df_ranked['target'] == 1, "🎯 Caught No-Show (TP)", "⚠️ Outreach to Attendee (FP)"),
                np.where(df_ranked['target'] == 1, "❌ Missed No-Show (FN)", "✅ Normal Attendance (TN)")
            )
            
            selected_for_review = df_ranked.iloc[:k].copy()
            remaining_appointments = df_ranked.iloc[k:].copy()
            
            total_actual_noshows = int(df_ranked['target'].sum())
            total_actual_attended = total_appointments - total_actual_noshows
            base_rate = total_actual_noshows / total_appointments if total_appointments > 0 else 0.0
            
            captured_noshows = int(selected_for_review['target'].sum())
            missed_noshows = total_actual_noshows - captured_noshows
            contacted_attendees = k - captured_noshows
            uncontacted_attendees = total_actual_attended - contacted_attendees
            
            capture_rate = (captured_noshows / total_actual_noshows * 100) if total_actual_noshows > 0 else 0.0
            precision_in_k = (captured_noshows / k * 100) if k > 0 else 0.0
            lift = (precision_in_k / (base_rate * 100)) if base_rate > 0 else 1.0
            
            # Financial Impact Modeling
            # Baseline: all no-shows are missed with 0 intervention
            cost_baseline = total_actual_noshows * cost_per_noshow
            # Proactive outreach:
            # Contacted patients who are no-shows: assumed percentage recover (patient confirms or reschedules early)
            recovered_slots = captured_noshows * phone_recovery_rate
            labor_cost = k * cost_per_call
            unrecovered_cost = (total_actual_noshows - recovered_slots) * cost_per_noshow
            cost_with_simulation = unrecovered_cost + labor_cost
            net_savings = cost_baseline - cost_with_simulation
            
            metrics.update({
                "total_actual_noshows": total_actual_noshows,
                "total_actual_attended": total_actual_attended,
                "base_noshow_rate_pct": round(base_rate * 100, 2),
                "captured_noshows_in_top_k": captured_noshows,
                "uncontacted_noshows_outside_k": missed_noshows,
                "contacted_attendees": contacted_attendees,
                "uncontacted_attendees": uncontacted_attendees,
                "capture_rate_pct": round(capture_rate, 2),
                "precision_in_top_k_pct": round(precision_in_k, 2),
                "lift_over_random": round(lift, 2),
                "recovered_slots_count": round(recovered_slots, 1),
                "estimated_net_savings_usd": round(net_savings, 2),
                "labor_cost_usd": round(labor_cost, 2),
                "assumed_phone_recovery_rate_pct": round(phone_recovery_rate * 100, 1)
            })
            
        return {
            "metrics": metrics,
            "selected_records": selected_for_review,
            "remaining_records": remaining_appointments,
            "ranked_all": df_ranked
        }

if __name__ == "__main__":
    sim = AppointmentOperationsSimulator()
    print("Simulator ready.")
