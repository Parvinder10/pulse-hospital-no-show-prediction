# PULSE: Hospital Appointment No-Show Prediction and Operational Decision Support

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-19%20passed%20(100%25)-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Project Identity:** PULSE – Predict risk. Plan responsibly.  
> **Candidate Assessment:** AI & Machine Learning Internship Assessment | ALGIONS AI LABS PRIVATE LIMITED  
> **Dashboard Link:** [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

## 1. Executive Summary & Non-Clinical Safety Boundary

**PULSE** is a production-grade, explainable machine learning decision-support platform engineered to predict individual outpatient appointment no-show probabilities strictly at the moment of booking registration ($t_{\text{pred}} = \text{ScheduledDay}$).

By generating calibrated risk probabilities upon scheduling, hospital coordinators can proactively prioritize human telephone reminders, coordinate transit assistance, and configure intelligent overbooking buffers.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       NON-NEGOTIABLE SAFETY BOUNDARY                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. NON-CLINICAL GUARANTEE: PULSE never predicts clinical diagnoses, triage   │
│    urgency, treatment plans, or health outcomes.                            │
│ 2. ZERO AUTOMATED ADVERSE ACTION: PULSE NEVER automatically cancels visits, │
│    denies care, or deprioritizes vulnerable patients.                       │
│ 3. HUMAN-IN-THE-LOOP: High-risk scores strictly route to hospital           │
│    coordinators for supportive outbound telephone contact.                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Technical Differentiators

* **Strict Booking-Time Prediction Point ($t_{\text{pred}}$):** Eliminates temporal data leakage by incorporating only data physically registered at or before the booking timestamp.
* **Leakage Audit & Simpson's Paradox Guard:** Excludes post-booking features (`SMS_received`) and sensitive clinical diagnoses (`Hipertension`, `Diabetes`, `Handcap`).
* **Point-in-Time Patient History:** Calculates cumulative prior visits, prior no-shows, and attendance ratios strictly on historical visits occurring prior to the current booking.
* **Chronological Train/Validation/Test Split:** Preserves temporal ordering to prevent data leakage from future clinical schedules.
* **Two Signature Operational Features:**
  * **Appointment Operations Simulator:** Ranks daily batches by risk and flags the top $K$ visits matching staff capacity, capturing 45.5% of all no-shows in the top 20% of appointments (a 2.27x lift).
  * **Threshold Trade-off Studio:** Interactive decision dashboard balancing Precision, Recall, review workload, and asymmetric financial costs ($C_{\text{FN}} = \$100$ vs $C_{\text{FP}} = \$5$).
* **Dual-Level Explainability:** Global log-odds / feature importance combined with local patient-level directional driver waterfalls.

---

## 3. System Architecture & Directory Layout

```
pulse-hospital-no-show-prediction/
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusion specifications
├── requirements.txt          # Production dependencies
├── README.md                 # System documentation & reproduction manual
├── data/
│   ├── raw/                  # Immutable verified raw dataset (110,527 rows)
│   └── processed/            # Chronologically split partitions (train, val, test)
├── src/
│   ├── __init__.py
│   ├── data/                 # Data ingestion, schema validation & EDA scripts
│   │   ├── download.py
│   │   ├── inspect_data.py
│   │   └── eda.py
│   ├── features/             # Point-in-time feature engineering pipeline
│   │   └── build_features.py
│   ├── models/               # Pipeline construction & model training
│   │   ├── pipeline.py
│   │   └── train.py
│   ├── evaluation/           # ROC/PR curves, calibration & Simulator
│   │   ├── evaluate.py
│   │   └── simulator.py
│   ├── threshold/            # Threshold Trade-off Studio engine
│   │   └── tradeoff.py
│   ├── explainability/       # Global & local feature attribution
│   │   └── explainer.py
│   └── api/                  # FastAPI backend service
│       ├── main.py
│       └── schemas.py
├── app/                      # Streamlit Clinical Decision Support Dashboard
│   └── main.py
├── artifacts/                # Serialized model binaries & evaluation metadata
│   ├── best_model.joblib
│   ├── baseline_model.joblib
│   ├── logistic_model.joblib
│   ├── rf_model.joblib
│   ├── hgb_model.joblib
│   ├── model_metadata.json
│   └── evaluation_metrics.json
├── reports/                  # Detailed markdown audits, EDA summary & figures
│   ├── feature_leakage_audit.md
│   ├── eda_summary.md
│   └── figures/
└── tests/                    # Pytest test suites (19 automated unit/integration tests)
    ├── test_data_validation.py
    ├── test_leakage_and_features.py
    ├── test_models_and_artifacts.py
    ├── test_signature_features.py
    └── test_api_endpoints.py
```

---

## 4. Final Benchmark Results (Untouched Test Set)

Evaluated on 16,579 untouched future appointments at the validation-selected operating threshold $\tau = 0.43$:

| Model Architecture | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Prior)** | 0.5000 | 0.1078 | 0.0000 | 0.0000 | 0.0000 | 0.1371 |
| **Logistic Regression** | 0.7731 | 0.2884 | 0.2509 | 0.6458 | 0.3614 | 0.1715 |
| **Random Forest** | 0.7812 | 0.2898 | 0.2403 | **0.7426** | 0.3631 | 0.1687 |
| **HistGradientBoosting (Champion)** | **0.7827** | **0.3073** | **0.2570** | 0.6290 | **0.3649** | **0.1655** |

* **Asymmetric Financial Modeling:** Under costs of \$100 per missed visit and \$5 per outbound phone call, deploying the champion model yields an estimated **\$90,535 in net savings (50.7% cost reduction)** over unmanaged scheduling.

---

## 5. Quickstart & Pipeline Reproduction

All steps are completely automated and reproducible with a single script or sequential commands:

### A. Environment Installation
```powershell
# Clone or navigate to the repository
cd c:\Users\Vishe\Downloads\parvac

# Install verified dependencies
pip install -r requirements.txt
```

### B. Execute Full End-to-End Pipeline
```powershell
# 1. Download & verify raw dataset integrity
python src/data/download.py

# 2. Run data quality & leakage inspection audit
python src/data/inspect_data.py

# 3. Clean anomalies, compute point-in-time features & chronological split
python src/features/build_features.py

# 4. Generate EDA visualizations & statistical summaries
python src/data/eda.py

# 5. Train baseline & candidate classifiers, saving serialized pipelines
python src/models/train.py

# 6. Run comprehensive calibration, threshold sweep & test evaluation
python src/evaluation/evaluate.py
```

### C. Run the Pytest Test Suite
```powershell
python -m pytest -v
```
*(All 19 test cases across schema integrity, leakage prevention, probability bounds, and API endpoints will execute).*

---

## 6. Running the Application Interfaces

### Launch FastAPI Backend
```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
* **Interactive Swagger UI:** http://127.0.0.1:8000/docs
* **Health Check:** `GET http://127.0.0.1:8000/health`
* **Model Introspection:** `GET http://127.0.0.1:8000/model/info`
* **Batch Operations Endpoint:** `POST http://127.0.0.1:8000/predict/batch`

#### Example Prediction Request (`curl` / PowerShell):
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{
  "lead_days": 14,
  "Age": 28,
  "gender_female": 1,
  "Scholarship": 1,
  "neighbourhood": "JARDIM DA PENHA",
  "prior_appointments_count": 2,
  "prior_no_shows_count": 1
}'
```

### Launch Interactive Streamlit Dashboard
```powershell
streamlit run app/main.py
```
* Access the 7-module clinical operations platform at [http://127.0.0.1:8501](http://127.0.0.1:8501) (or http://localhost:8501).

