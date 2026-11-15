"""
Comprehensive Validation, Calibration, Threshold Analysis, and Test Evaluation for PULSE.

Key Capabilities:
1. Multi-model ROC and Precision-Recall curve generation.
2. Calibration curve (reliability diagram) and Brier score calculation.
3. Threshold sweep [0.05 - 0.95] on Validation set for operational decision support:
   - Finds F1-optimal threshold.
   - Finds High-Recall (>= 75%) threshold.
   - Computes workload, TP, FP, TN, FN, review volume.
4. Final untouched Test Set evaluation using chosen validation thresholds.
5. Operational Cost-Benefit modeling ($100 per missed slot vs $5 per phone call).
6. Visualization generation to reports/figures/ and JSON serialization to artifacts/.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, precision_score, recall_score, f1_score,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from src.models.pipeline import ALL_MODEL_FEATURES

VAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "val.csv")
TEST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "test.csv")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports", "figures")

def run_evaluation():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_val, y_val = val_df[ALL_MODEL_FEATURES], val_df['target']
    X_test, y_test = test_df[ALL_MODEL_FEATURES], test_df['target']
    
    # Load candidate models
    models = {
        'Baseline (Prior)': joblib.load(os.path.join(ARTIFACTS_DIR, 'baseline_model.joblib')),
        'Logistic Regression': joblib.load(os.path.join(ARTIFACTS_DIR, 'logistic_model.joblib')),
        'Random Forest': joblib.load(os.path.join(ARTIFACTS_DIR, 'rf_model.joblib')),
        'HistGradientBoosting': joblib.load(os.path.join(ARTIFACTS_DIR, 'hgb_model.joblib'))
    }
    
    sns.set_theme(style="whitegrid", palette="muted")
    
    # =========================================================================
    # 1. ROC Curves & PR Curves on Validation Set
    # =========================================================================
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.5))
    val_probs = {}
    
    for name, model in models.items():
        probs = model.predict_proba(X_val)[:, 1]
        val_probs[name] = probs
        
        # ROC
        fpr, tpr, _ = roc_curve(y_val, probs)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)
        
        # PR
        prec, rec, _ = precision_recall_curve(y_val, probs)
        pr_auc = average_precision_score(y_val, probs)
        ax_pr.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", linewidth=2)
        
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Chance (0.500)')
    ax_roc.set_title("ROC Curves (Validation Set)", fontsize=12, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.legend(loc='lower right')
    
    base_rate = float(y_val.mean())
    ax_pr.plot([0, 1], [base_rate, base_rate], 'k--', label=f'Prior Baseline ({base_rate:.3f})')
    ax_pr.set_title("Precision-Recall Curves (Validation Set)", fontsize=12, fontweight='bold')
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc='upper right')
    
    roc_pr_path = os.path.join(FIGURES_DIR, "roc_pr_curves.png")
    fig.savefig(roc_pr_path, dpi=200)
    plt.close(fig)
    print(f"[EVALUATION] Saved: {roc_pr_path}")
    
    # =========================================================================
    # 2. Calibration Curve (Reliability Diagram)
    # =========================================================================
    fig, ax_cal = plt.subplots(figsize=(7.5, 5))
    ax_cal.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    brier_scores = {}
    for name, probs in val_probs.items():
        if name == 'Baseline (Prior)':
            continue
        prob_true, prob_pred = calibration_curve(y_val, probs, n_bins=10)
        brier = brier_score_loss(y_val, probs)
        brier_scores[name] = round(float(brier), 4)
        ax_cal.plot(prob_pred, prob_true, marker='o', label=f"{name} (Brier = {brier:.3f})")
        
    ax_cal.set_title("Probability Calibration Curve (Reliability Diagram)", fontsize=12, fontweight='bold')
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Empirical Fraction of Positives")
    ax_cal.legend(loc='upper left')
    cal_path = os.path.join(FIGURES_DIR, "calibration_curve.png")
    fig.savefig(cal_path, dpi=200)
    plt.close(fig)
    print(f"[EVALUATION] Saved: {cal_path}")
    
    # =========================================================================
    # 3. Threshold Sweep on Validation Set (Using Champion Model)
    # =========================================================================
    champion_name = 'HistGradientBoosting'  # Strongest nonlinear classifier with best calibration
    champion_model = models[champion_name]
    champ_probs_val = val_probs[champion_name]
    
    thresholds = np.linspace(0.05, 0.95, 91)
    threshold_records = []
    
    for t in thresholds:
        preds = (champ_probs_val >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        flagged_count = int(tp + fp)
        flagged_pct = (flagged_count / len(y_val)) * 100
        
        # Cost function: $100 per missed no-show (FN), $5 per reminder call (TP + FP)
        operational_cost = (fn * 100) + (flagged_count * 5)
        # Cost if no model / no reminders: all no-shows are missed (FN = total actual positives)
        cost_no_model = int(y_val.sum() * 100)
        cost_savings = cost_no_model - operational_cost
        
        threshold_records.append({
            "threshold": round(float(t), 3),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "flagged_count": flagged_count,
            "flagged_pct": round(float(flagged_pct), 2),
            "operational_cost": round(float(operational_cost), 2),
            "cost_savings": round(float(cost_savings), 2)
        })
        
    thresh_df = pd.DataFrame(threshold_records)
    
    # Optimal Thresholds:
    # 1. F1 Optimal
    f1_idx = thresh_df['f1_score'].idxmax()
    f1_opt_row = thresh_df.iloc[f1_idx].to_dict()
    
    # 2. High Recall (Recall >= 0.75 with highest Precision)
    high_rec_candidates = thresh_df[thresh_df['recall'] >= 0.75]
    if not high_rec_candidates.empty:
        high_rec_row = high_rec_candidates.sort_values(by='precision', ascending=False).iloc[0].to_dict()
    else:
        high_rec_row = f1_opt_row
        
    # 3. Maximum Cost Savings
    cost_opt_idx = thresh_df['cost_savings'].idxmax()
    cost_opt_row = thresh_df.iloc[cost_opt_idx].to_dict()
    
    # Plot Threshold Curves
    fig, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax_t1.plot(thresh_df['threshold'], thresh_df['precision'], label='Precision', color='#2b5c8f', linewidth=2)
    ax_t1.plot(thresh_df['threshold'], thresh_df['recall'], label='Recall', color='#d9534f', linewidth=2)
    ax_t1.plot(thresh_df['threshold'], thresh_df['f1_score'], label='F1-Score', color='#5cb85c', linewidth=2.5)
    ax_t1.axvline(f1_opt_row['threshold'], color='purple', linestyle='--', label=f"F1-Opt ({f1_opt_row['threshold']:.2f})")
    ax_t1.set_title("Performance Metrics vs. Decision Threshold", fontsize=12, fontweight='bold')
    ax_t1.set_xlabel("Probability Threshold")
    ax_t1.set_ylabel("Score")
    ax_t1.legend(loc='center right')
    
    ax_t2.plot(thresh_df['threshold'], thresh_df['flagged_pct'], label='% Appointments Flagged', color='#f0ad4e', linewidth=2)
    ax_t2.plot(thresh_df['threshold'], thresh_df['cost_savings'] / 1000, label='Net Savings ($1k)', color='#5cb85c', linewidth=2)
    ax_t2.axvline(cost_opt_row['threshold'], color='green', linestyle=':', label=f"Cost-Opt ({cost_opt_row['threshold']:.2f})")
    ax_t2.set_title("Operational Review Workload & Net Cost Savings", fontsize=12, fontweight='bold')
    ax_t2.set_xlabel("Probability Threshold")
    ax_t2.set_ylabel("Flagged % / Net Savings ($k)")
    ax_t2.legend(loc='upper right')
    
    thresh_fig_path = os.path.join(FIGURES_DIR, "threshold_tradeoff.png")
    fig.savefig(thresh_fig_path, dpi=200)
    plt.close(fig)
    print(f"[EVALUATION] Saved: {thresh_fig_path}")
    
    # =========================================================================
    # 4. Untouched Final Test Set Evaluation
    # =========================================================================
    print("\n" + "="*70)
    print("           FINAL UNTOUCHED TEST SET EVALUATION (N=16,579)              ")
    print("="*70)
    
    test_results = {}
    chosen_threshold = f1_opt_row['threshold']
    print(f"Applying Chosen Validation Threshold: tau = {chosen_threshold:.2f}")
    
    for name, model in models.items():
        t_probs = model.predict_proba(X_test)[:, 1]
        t_preds = (t_probs >= chosen_threshold).astype(int)
        
        roc_auc = float(roc_auc_score(y_test, t_probs))
        pr_auc = float(average_precision_score(y_test, t_probs))
        p = float(precision_score(y_test, t_preds, zero_division=0))
        r = float(recall_score(y_test, t_preds, zero_division=0))
        f1 = float(f1_score(y_test, t_preds, zero_division=0))
        brier = float(brier_score_loss(y_test, t_probs))
        tn, fp, fn, tp = confusion_matrix(y_test, t_preds).ravel()
        
        test_results[name] = {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "brier_score": round(brier, 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }
        print(f"{name:<22} | ROC: {roc_auc:.4f} | PR: {pr_auc:.4f} | Prec: {p:.4f} | Rec: {r:.4f} | F1: {f1:.4f}")
        
    # Plot Test Confusion Matrix for Champion Model
    champ_test_probs = models[champion_name].predict_proba(X_test)[:, 1]
    champ_test_preds = (champ_test_probs >= chosen_threshold).astype(int)
    cm = confusion_matrix(y_test, champ_test_preds)
    
    fig, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', cbar=False,
                xticklabels=['Pred Attended', 'Pred No-Show'],
                yticklabels=['Actual Attended', 'Actual No-Show'], ax=ax_cm)
    ax_cm.set_title(f"Test Confusion Matrix: {champion_name} (tau = {chosen_threshold:.2f})", fontsize=12, fontweight='bold')
    cm_path = os.path.join(FIGURES_DIR, "test_confusion_matrix.png")
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"[EVALUATION] Saved: {cm_path}")
    
    # Save all evaluation metrics
    eval_payload = {
        "evaluation_split": "Untouched Future Test Set (Chronological)",
        "test_samples": len(y_test),
        "test_actual_noshows": int(y_test.sum()),
        "test_base_rate": round(float(y_test.mean()), 4),
        "chosen_threshold": chosen_threshold,
        "threshold_profiles": {
            "f1_optimal": f1_opt_row,
            "high_recall": high_rec_row,
            "cost_optimal": cost_opt_row
        },
        "test_model_comparison": test_results,
        "threshold_sweep_sample": threshold_records[::5]  # Every 5th for compact viewing
    }
    
    metrics_path = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(eval_payload, f, indent=2)
    print(f"[SAVED] Evaluation metrics written to: {metrics_path}")
    
    return eval_payload

if __name__ == "__main__":
    run_evaluation()
