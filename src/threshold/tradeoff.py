"""
Signature Feature 2: Threshold Trade-off Studio for PULSE.

Provides interactive threshold exploration and real-time operational tradeoff
computations for clinical decision-makers.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from src.models.pipeline import ALL_MODEL_FEATURES

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts", "best_model.joblib")
DEFAULT_VAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "val.csv")
DEFAULT_TEST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "test.csv")

class ThresholdTradeoffStudio:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, val_path: str = DEFAULT_VAL_PATH, test_path: str = DEFAULT_TEST_PATH):
        self.model = joblib.load(model_path)
        self.feature_columns = ALL_MODEL_FEATURES
        
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        
        # Precompute predicted probabilities
        self.val_probs = self.model.predict_proba(self.val_df[self.feature_columns])[:, 1]
        self.test_probs = self.model.predict_proba(self.test_df[self.feature_columns])[:, 1]
        
        self.y_val = self.val_df['target'].values
        self.y_test = self.test_df['target'].values
        
    def evaluate_threshold(self, threshold: float, dataset: str = "validation", cost_fn: float = 100.0, cost_fp: float = 5.0) -> dict:
        """
        Evaluates operational performance metrics for an arbitrary decision threshold.
        
        Parameters:
        -----------
        threshold: float between 0.01 and 0.99
        dataset: "validation" or "test"
        cost_fn: cost of a missed appointment (false negative), default $100
        cost_fp: cost of unnecessary staff phone call (false positive), default $5
        """
        probs = self.val_probs if dataset == "validation" else self.test_probs
        y_true = self.y_val if dataset == "validation" else self.y_test
        
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        
        p = float(precision_score(y_true, preds, zero_division=0))
        r = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        
        total = len(y_true)
        flagged = int(tp + fp)
        flagged_pct = (flagged / total * 100) if total > 0 else 0.0
        
        # Total cost = cost of all FN (unaddressed no-shows) + cost of all staff calls (flagged)
        total_cost = (fn * cost_fn) + (flagged * cost_fp)
        baseline_cost = int(y_true.sum()) * cost_fn
        net_savings = baseline_cost - total_cost
        
        # Human-readable trade-off diagnosis
        if threshold < 0.25:
            tradeoff_summary = "High-Recall / High-Workload: Flags a broad fraction of appointments, catching most no-shows at the expense of many false alerts."
        elif threshold > 0.55:
            tradeoff_summary = "High-Precision / Low-Workload: Minimizes staff call burden, flagging only high-confidence risks, but leaving many potential no-shows uncontacted."
        else:
            tradeoff_summary = "Balanced Operational: Balances outreach staff workload against no-show reduction."
            
        return {
            "dataset": dataset,
            "threshold": round(threshold, 3),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            },
            "appointments_total": total,
            "flagged_for_review": flagged,
            "flagged_percent": round(flagged_pct, 2),
            "missed_noshows_fn": int(fn),
            "wasted_calls_fp": int(fp),
            "total_operational_cost": round(total_cost, 2),
            "baseline_unmanaged_cost": round(baseline_cost, 2),
            "net_cost_savings": round(net_savings, 2),
            "tradeoff_summary": tradeoff_summary
        }
        
    def generate_tradeoff_curve(self, dataset: str = "validation", steps: int = 50) -> pd.DataFrame:
        """Generates continuous tradeoff metrics across threshold range."""
        thresholds = np.linspace(0.05, 0.95, steps)
        records = [self.evaluate_threshold(t, dataset=dataset) for t in thresholds]
        
        df = pd.DataFrame([{
            "threshold": r["threshold"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1_score": r["f1_score"],
            "flagged_percent": r["flagged_percent"],
            "tp": r["confusion_matrix"]["tp"],
            "fp": r["confusion_matrix"]["fp"],
            "fn": r["confusion_matrix"]["fn"],
            "net_cost_savings": r["net_cost_savings"]
        } for r in records])
        return df

if __name__ == "__main__":
    studio = ThresholdTradeoffStudio()
    print("\n--- THRESHOLD TRADE-OFF STUDIO BENCHMARK ---")
    for t in [0.25, 0.43, 0.65]:
        res = studio.evaluate_threshold(t, dataset="validation")
        print(f"Threshold tau = {t}: Precision={res['precision']:.3f}, Recall={res['recall']:.3f}, Flagged={res['flagged_percent']}%, Savings=${res['net_cost_savings']:,.0f}")
