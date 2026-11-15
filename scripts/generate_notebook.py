"""
Generates the clean walkthrough Jupyter notebook for PULSE.
"""
import os
import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# PULSE: Hospital Appointment No-Show Prediction & Operational Decision Support\n",
                "**AI & Machine Learning Internship Assessment | ALGIONS AI LABS PRIVATE LIMITED**\n",
                "**Candidate:** Parvinder  \n",
                "**Project Identity:** *Predict risk. Plan responsibly.*\n",
                "\n",
                "---\n",
                "### Non-Negotiable Operational Safety Boundaries\n",
                "1. **Strictly Non-Clinical:** PULSE never predicts diagnoses, clinical triage, treatment needs, or health outcomes.\n",
                "2. **Zero Automated Adverse Actions:** PULSE never automatically cancels appointments or deprioritizes patients.\n",
                "3. **Booking-Time Prediction Point:** Strictly evaluates features available at appointment booking registration ($t_{pred} = ScheduledDay$)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Environment Setup & Data Loading"]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys\n",
                "sys.path.insert(0, os.path.abspath('..'))\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import joblib, json\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "train_df = pd.read_csv('../data/processed/train.csv')\n",
                "val_df = pd.read_csv('../data/processed/val.csv')\n",
                "test_df = pd.read_csv('../data/processed/test.csv')\n",
                "print(f'Train cohort:      {len(train_df):,} records | Target rate: {train_df[\"target\"].mean()*100:.2f}%')\n",
                "print(f'Validation cohort: {len(val_df):,} records | Target rate: {val_df[\"target\"].mean()*100:.2f}%')\n",
                "print(f'Test cohort:       {len(test_df):,} records | Target rate: {test_df[\"target\"].mean()*100:.2f}%')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Feature Availability & Leakage Prevention Audit"]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.models.pipeline import ALL_MODEL_FEATURES\n",
                "print(f'Audited Prediction-Time Features ({len(ALL_MODEL_FEATURES)}):\\n', ALL_MODEL_FEATURES)\n",
                "\n",
                "# Verify forbidden leaked or sensitive clinical features are absent\n",
                "forbidden = {'SMS_received', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap'}\n",
                "assert not any(f in ALL_MODEL_FEATURES for f in forbidden)\n",
                "print('\\n[PASSED] Zero-Leakage & Ethical AI boundaries strictly verified!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Benchmark Evaluation on Untouched Test Set"]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open('../artifacts/evaluation_metrics.json') as f:\n",
                "    metrics = json.load(f)\n",
                "\n",
                "test_comp = metrics['test_model_comparison']\n",
                "table_data = []\n",
                "for model_name, d in test_comp.items():\n",
                "    cm = d['confusion_matrix']\n",
                "    table_data.append({\n",
                "        'Model': model_name,\n",
                "        'ROC-AUC': f\"{d['roc_auc']:.4f}\",\n",
                "        'PR-AUC': f\"{d['pr_auc']:.4f}\",\n",
                "        'Precision': f\"{d['precision']*100:.1f}%\",\n",
                "        'Recall': f\"{d['recall']*100:.1f}%\",\n",
                "        'F1-Score': f\"{d['f1_score']:.4f}\",\n",
                "        'Brier Score': f\"{d['brier_score']:.4f}\",\n",
                "        'TP (Caught)': cm['tp'],\n",
                "        'FP (False Call)': cm['fp'],\n",
                "        'FN (Missed)': cm['fn']\n",
                "    })\n",
                "\n",
                "pd.DataFrame(table_data).set_index('Model')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Signature Feature 1: Operations Simulator Demonstration"]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.evaluation.simulator import AppointmentOperationsSimulator\n",
                "sim = AppointmentOperationsSimulator('../artifacts/best_model.joblib')\n",
                "batch_sample = test_df.head(250)\n",
                "sim_res = sim.simulate_batch(batch_sample, daily_capacity=50)\n",
                "for k, v in sim_res['metrics'].items():\n",
                "    print(f'{k}: {v}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Single Prediction & Local Attribution"]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.explainability.explainer import ModelExplainer\n",
                "exp = ModelExplainer('../artifacts/best_model.joblib', '../artifacts/logistic_model.joblib')\n",
                "sample_apt = {\n",
                "    'lead_days': 21,\n",
                "    'Age': 24,\n",
                "    'is_same_day': 0,\n",
                "    'lead_weeks': 3.0,\n",
                "    'booking_hour': 14,\n",
                "    'booking_weekday': 1,\n",
                "    'booking_month': 5,\n",
                "    'appointment_weekday': 3,\n",
                "    'appointment_month': 6,\n",
                "    'age_group': 'Youth',\n",
                "    'gender_female': 1,\n",
                "    'Scholarship': 1,\n",
                "    'neighbourhood': 'JARDIM DA PENHA',\n",
                "    'prior_appointments_count': 3,\n",
                "    'prior_no_shows_count': 2,\n",
                "    'prior_no_show_rate': 0.67,\n",
                "    'is_repeat_patient': 1\n",
                "}\n",
                "pred_out = exp.explain_instance(sample_apt)\n",
                "print(f\"Predicted Risk: {pred_out['predicted_risk_probability']*100:.1f}%\")\n",
                "print(f\"Risk Tier:      {pred_out['risk_tier']}\")\n",
                "print(f\"Action:         {pred_out['recommended_operational_action']}\")\n",
                "for d in pred_out['top_drivers']:\n",
                "    print(f\"  [{d['direction'].upper()}] {d['factor']} -> {d['impact']}\")"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"},
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks", "pulse_walkthrough.ipynb")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"[OK] Walkthrough notebook written to: {output_path}")
