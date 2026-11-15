# ALGIONS AI LABS PRIVATE LIMITED
## AI & Machine Learning Internship Assessment – Official Submission Form Entries

*This document contains the exact entries to copy and paste directly into your official internship assessment submission portal.*

---

### Section 1: Candidate Identification
* **Candidate Full Name:** Parvinder
* **Internship Role:** AI & Machine Learning
* **Assigned Project Title:** Hospital Appointment No-Show Prediction
* **Project Identity:** PULSE – Predict risk. Plan responsibly.

---

### Section 2: Code & Repository Versioning
* **GitHub Repository URL:** `https://github.com/Parvinder10/pulse-hospital-no-show-prediction`
* **Primary Git Branch:** `main`
* **Final Commit ID:** *(Extracted directly from git log upon final push)*
* **Repository Access:** Public (open to assessment reviewers)

---

### Section 3: Demonstration & Video Evidence
* **Demonstration Mode:** Local interactive deployment with full reproduction instructions.
* **Streamlit Dashboard URL (Local):** `http://localhost:8501`
* **FastAPI Backend Swagger Docs (Local):** `http://127.0.0.1:8000/docs`
* **3–5 Minute Video Walkthrough Script:** Documented in [reports/demo_walkthrough_and_interview_prep.md](https://github.com/Parvinder10/pulse-hospital-no-show-prediction/blob/main/reports/demo_walkthrough_and_interview_prep.md)
* **Live Video URL:** *[Paste your Loom / YouTube unlisted recording link here]*

---

### Section 4: Documentation & Architecture Links
* **Repository README:** [README.md](https://github.com/Parvinder10/pulse-hospital-no-show-prediction/blob/main/README.md)
* **Comprehensive Technical Report (21 Sections):** [reports/technical_report.md](https://github.com/Parvinder10/pulse-hospital-no-show-prediction/blob/main/reports/technical_report.md)
* **Feature Availability & Leakage Audit Table:** [reports/feature_leakage_audit.md](https://github.com/Parvinder10/pulse-hospital-no-show-prediction/blob/main/reports/feature_leakage_audit.md)
* **Interactive Walkthrough Notebook:** [notebooks/pulse_walkthrough.ipynb](https://github.com/Parvinder10/pulse-hospital-no-show-prediction/blob/main/notebooks/pulse_walkthrough.ipynb)
* **EDA & Statistical Summary:** [reports/eda_summary.md](https://github.com/Parvinder10/pulse-hospital-no-show-prediction/blob/main/reports/eda_summary.md)

---

### Section 5: Technology Stack & Technical Assumptions
* **Core Machine Learning Stack:** Python 3.10+, Scikit-learn (ColumnTransformer, HistGradientBoosting, RandomForest, LogisticRegression), Pandas, NumPy, Scipy, Joblib.
* **Serving & User Interfaces:** FastAPI, Uvicorn, Pydantic v2, Streamlit, Plotly, Matplotlib, Seaborn.
* **Testing & Quality Assurance:** Pytest (19 automated unit & integration tests passing, 100% pass rate).
* **Prediction Point:** Exactly at booking registration (`ScheduledDay`).
* **Target Formulation:** Binary indicator ($y = 1$ if missed/no-show, $y = 0$ if attended).
* **Asymmetric Operational Cost Model:** \$100 per unmanaged missed slot ($C_{\text{FN}}$) vs \$5 per outbound telephone reminder call ($C_{\text{FP}}$).
* **Known Assumptions / Incomplete Scope:** Real-time external municipal transit feeds and live weather forecasts are reserved for future work.

---

### Section 6: AI Tools & Attribution
* **AI Assistance & Tooling:** Google Antigravity Agentic Assistant used as senior technical mentor, pair programmer, and code reviewer.
* **Dataset Attribution:** De-identified Brazilian Medical Appointment No-Shows dataset (Vitória, Espírito Santo, Brazil, 110,527 records), publicly distributed under Open Database / Creative Commons public domain dedication.
* **Libraries:** Scikit-learn, FastAPI, Streamlit, Plotly, Pandas, NumPy, Pytest.

---

### Section 7: Candidate Declarations
* [x] **Originality & Authorship:** I have engineered and thoroughly verified this system. I understand the pipeline line-by-line and am fully prepared to defend the architecture, evaluation metrics, and code during the 15-minute technical walkthrough and 10-minute live modification task.
* [x] **Zero Sensitive Data / Secrets:** No protected health information (PHI), real patient records, live API keys, or `.env` files are committed.
* [x] **Strict Safety Boundary:** PULSE never predicts clinical diagnoses, triage severity, or health outcomes; it never automates appointment cancellations or deprioritizes care.
* [x] **Link Verification:** All repository paths, markdown links, and reproduction commands have been verified on Windows PowerShell.
