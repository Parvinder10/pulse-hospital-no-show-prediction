"""
Model Training and Serialization Script for PULSE.
Trains Dummy baseline, Logistic Regression, Random Forest, and HistGradientBoosting.
Evaluates on chronological validation split, handles class imbalance,
and serializes artifacts.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from src.models.pipeline import ALL_MODEL_FEATURES, get_model_pipeline

TRAIN_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "train.csv")
VAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "val.csv")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")

def train_and_evaluate_models(train_path: str = TRAIN_DATA_PATH, val_path: str = VAL_DATA_PATH, artifacts_dir: str = ARTIFACTS_DIR):
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"[TRAINING] Loading training set from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"[TRAINING] Loading validation set from: {val_path}")
    val_df = pd.read_csv(val_path)
    
    X_train = train_df[ALL_MODEL_FEATURES]
    y_train = train_df['target']
    
    X_val = val_df[ALL_MODEL_FEATURES]
    y_val = val_df['target']
    
    models_to_train = [
        ('dummy', 'baseline_model.joblib'),
        ('logistic', 'logistic_model.joblib'),
        ('random_forest', 'rf_model.joblib'),
        ('hist_gradient_boosting', 'hgb_model.joblib')
    ]
    
    results = {}
    best_pr_auc = -1.0
    best_model_name = None
    best_pipeline = None
    
    print("\n" + "="*70)
    print(f"{'MODEL':<25} | {'ROC-AUC':<10} | {'PR-AUC':<10} | {'BRIER':<10} | {'LOG-LOSS':<10}")
    print("="*70)
    
    for model_name, filename in models_to_train:
        print(f"Training {model_name}...")
        pipeline = get_model_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        
        # Predict probabilities for positive class (1 = no-show)
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        
        # Compute metrics
        roc_auc = float(roc_auc_score(y_val, val_probs))
        pr_auc = float(average_precision_score(y_val, val_probs))
        brier = float(brier_score_loss(y_val, val_probs))
        # Clip probabilities for stable log-loss
        eps = 1e-15
        clipped_probs = np.clip(val_probs, eps, 1 - eps)
        loss = float(log_loss(y_val, clipped_probs))
        
        print(f"{model_name:<25} | {roc_auc:<10.4f} | {pr_auc:<10.4f} | {brier:<10.4f} | {loss:<10.4f}")
        
        results[model_name] = {
            "model_type": model_name,
            "filename": filename,
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(loss, 4)
        }
        
        # Save model pipeline
        out_path = os.path.join(artifacts_dir, filename)
        joblib.dump(pipeline, out_path)
        
        # Track best model based on PR-AUC (key metric under class imbalance)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_name = model_name
            best_pipeline = pipeline
            
    print("="*70)
    print(f"[CHAMPION MODEL] Winner: {best_model_name} with PR-AUC = {best_pr_auc:.4f}")
    
    # Save best model copy as best_model.joblib
    best_model_path = os.path.join(artifacts_dir, "best_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"[SAVED] Champion model saved to: {best_model_path}")
    
    # Save metadata
    metadata = {
        "project": "PULSE - Hospital Appointment No-Show Prediction",
        "training_timestamp": datetime.utcnow().isoformat() + "Z",
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_target_rate": float(train_df['target'].mean()),
        "val_target_rate": float(val_df['target'].mean()),
        "features": ALL_MODEL_FEATURES,
        "feature_count": len(ALL_MODEL_FEATURES),
        "champion_model": best_model_name,
        "model_comparison": results
    }
    
    metadata_path = os.path.join(artifacts_dir, "model_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Metadata saved to: {metadata_path}")
    
    return results

if __name__ == "__main__":
    train_and_evaluate_models()
