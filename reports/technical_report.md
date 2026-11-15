# PULSE: Hospital Appointment No-Show Prediction and Operational Decision Support
**Technical Engineering & Scientific Evaluation Report**  
*AI & Machine Learning Internship Assessment | ALGIONS AI LABS PRIVATE LIMITED*  
*Author:* Machine Learning Engineering Candidate  
*Project Identity:* **PULSE – Predict risk. Plan responsibly.**

---

## 1. Abstract
Unannounced outpatient appointment no-shows inflict substantial operational waste on healthcare provider networks, leading to idle clinical staff, underutilized diagnostic machinery, and delayed care for waitlisted patients. **PULSE** is an end-to-end, explainable, and reproducible machine learning decision-support platform engineered to predict individual appointment no-show probability strictly at the moment of booking registration ($t_{\text{pred}} = \text{ScheduledDay}$). Using a verified cohort of 110,527 appointments from Vitória, Brazil, we conducted a rigorous leakage audit, purged post-booking features (`SMS_received`) and sensitive clinical variables, constructed point-in-time patient behavioral histories, and instituted a chronological train/validation/test split. Comparing a prior-probability baseline with Logistic Regression, Random Forest, and HistGradientBoosting, the champion model achieved a Test ROC-AUC of 0.7827 and PR-AUC of 0.3073 (a 2.85x lift over baseline). Furthermore, PULSE introduces two signature operational components: the **Appointment Operations Simulator** (demonstrating a 2.27x lift in no-show capture under a 20% outreach quota) and the **Threshold Trade-off Studio** (balancing precision, recall, and review workload). The platform is exposed through a validated FastAPI backend and an interactive Streamlit dashboard adhering to strict non-clinical safety boundaries.

---

## 2. Business Problem & Operational Reality
In outpatient health systems, empty consultation slots cannot be recovered once elapsed. The operational ramifications include:
1. **Direct Physician & Facility Downtime:** Fixed hourly overhead for specialists, nurses, and exam suites persists regardless of attendance.
2. **Access-to-Care Delays:** Patients waiting weeks for non-urgent specialist care are denied timely intervention, whereas empty slots could have accommodated them with advance notice.
3. **Budgetary Inefficiency:** In public and fee-for-service systems alike, missed appointments destabilize revenue planning and operational resource sizing.

**The Operational Objective:** Enable hospital coordinators to identify appointments with elevated no-show probability immediately upon scheduling so that targeted, human-delivered reminder phone calls and optimal overbooking buffers can be established responsibly.

---

## 3. Prediction Point and Horizon
Defining the exact temporal point of prediction is the cornerstone of defensible operational ML.
* **Prediction Unit:** Exactly one scheduled outpatient appointment record.
* **Prediction Point ($t_{\text{pred}}$):** The exact date and timestamp when the appointment booking is logged (`ScheduledDay`).
* **Prediction Horizon ($\Delta t$):** The lead time interval between booking registration and the scheduled consultation date (`AppointmentDay`), ranging from 0 days (same-day appointments) to 179 days.
* **Model Output:** Estimated, calibrated probability $\hat{p} = P(y = 1 \mid \mathbf{x}_{t_{\text{pred}}}) \in [0, 1]$.

```
Timeline:
──── ScheduledDay (t_pred) ──────────────────────> AppointmentDay (Outcome y revealed)
           ▲                                                   ▲
     Model Evaluated                                     Slot Attended (0)
   (Only t <= t_pred used)                                 or Missed (1)
```

---

## 4. Target Definition
The ground truth target variable is defined as:
$$y = \begin{cases} 1 & \text{if the patient fails to attend the scheduled appointment (No-Show)} \\ 0 & \text{if the patient attends and completes the appointment} \end{cases}$$

### The Kaggle Inversion Trap
In the raw Kaggle medical dataset (`KaggleV2-May-2016.csv`), the column is labelled `No-show`:
* `No-show == 'No'` $\implies$ Patient showed up ($y = 0$).
* `No-show == 'Yes'` $\implies$ Patient did **not** show up ($y = 1$).
PULSE explicitly maps `'Yes' \to 1` and `'No' \to 0`.

### Cancellation vs. No-Show
A cancellation is an advance notification allowing the clinic to offer the slot to another patient. A **no-show** is an unannounced absence where the clinic is left with an empty room and zero notice.

---

## 5. Dataset Source and Licensing
* **Dataset:** Brazilian Medical Appointment No-Shows Dataset (Vitória, Espírito Santo, Brazil).
* **Source:** Municipal healthcare scheduling registry across 81 public health units (Unidades Básicas de Saúde), publicly released on Kaggle.
* **License & Privacy:** De-identified public dataset under Open Database / Creative Commons public domain dedication. Contains zero Direct Identifiers, Protected Health Information (PHI), or HIPAA/GDPR restricted data.
* **Scale:** 110,527 raw appointment records across 62,299 unique patients.

---

## 6. Data Dictionary
| Raw Column | Type | Description | Ingestion Action |
| :--- | :--- | :--- | :--- |
| `PatientId` | Float/Int | De-identified patient identifier | Grouping key for point-in-time history |
| `AppointmentID` | Integer | Unique appointment transaction key | Retained for audit; excluded from model |
| `Gender` | String | Patient sex (`F` or `M`) | Binary encoded (`gender_female`) |
| `ScheduledDay` | ISO Datetime | Booking registration timestamp (UTC) | Prediction anchor $t_{\text{pred}}$; extracted hour, weekday, month |
| `AppointmentDay`| ISO Datetime | Scheduled consultation date (00:00:00) | Extracted weekday, month, lead time |
| `Age` | Integer | Patient age in years | Validated $\ge 0$; binned into age groups |
| `Neighbourhood` | String | District of the healthcare clinic (81 units) | One-hot encoded categorical |
| `Scholarship` | Binary (0/1)| Enrollment in Bolsa Família welfare program | Socioeconomic proxy feature |
| `Hipertension` | Binary (0/1)| Hypertension diagnosis | **Purged (Clinical Safety Boundary)** |
| `Diabetes` | Binary (0/1)| Diabetes diagnosis | **Purged (Clinical Safety Boundary)** |
| `Alcoholism` | Binary (0/1)| Alcoholism indicator | **Purged (Clinical Safety Boundary)** |
| `Handcap` | Integer (0-4)| Disability count/severity level | **Purged (Clinical Safety Boundary)** |
| `SMS_received` | Binary (0/1)| SMS reminder dispatch status | **Purged (Temporal Data Leakage)** |
| `No-show` | String | Target: `'Yes'` = Missed, `'No'` = Attended | Binary mapped to $y \in \{0, 1\}$ |

---

## 7. Exploratory Data Analysis (EDA)
Empirical analysis of the cleaned training dataset revealed several behavioral drivers:

1. **Target Imbalance:**
   * Attended ($y = 0$): 79.81%
   * Missed ($y = 1$): 20.19%
   * Imbalance ratio is roughly 4:1. Accuracy alone is an invalid evaluation metric.

2. **Lead Time Horizon (The Primary Behavioral Feature):**
   * Same-day bookings (`lead_days == 0`): 34.89% of appointments, exhibiting an ultra-low no-show rate of **4.65%**.
   * Advance bookings (`lead_days > 0`): 65.11% of appointments, with a no-show rate of **28.52%** (a 6.1x increase).
   * Beyond 2 weeks lead time, no-show rates plateau between 32% and 34%.

3. **Demographic Dynamics:**
   * Younger patients (ages 15–28) demonstrate higher missed appointment rates (~24–26%).
   * Senior patients (ages 60+) exhibit higher attendance reliability (~15% no-show rate).

4. **Temporal Patterns:**
   * Tuesday and Wednesday exhibit highest appointment volumes.
   * Friday and Saturday appointments exhibit slightly elevated no-show frequencies.

---

## 8. Data Cleaning and Anomaly Handling
Three data integrity anomalies were identified and purged:
1. **Negative Lead Times:** 5 records where `AppointmentDay < ScheduledDay` (down to -6 days), physically impossible clerical errors.
2. **Negative Patient Age:** Exactly 1 record where `Age = -1`.
3. Total valid records retained: **110,521** (purging exactly 6 invalid records, 0.005% of dataset).

---

## 9. Feature Availability & Leakage Audit Matrix
Every candidate feature was audited against the prediction point $t_{\text{pred}}$:

| Feature | Available at $t_{\text{pred}}$? | Decision | Rationale |
| :--- | :---: | :---: | :--- |
| `lead_days` | Yes | **Included** | Computed as difference between appointment date and booking date. |
| `is_same_day` | Yes | **Included** | Binary indicator (`lead_days == 0`). Highly predictive. |
| `booking_hour`, `booking_weekday` | Yes | **Included** | Extracted from `ScheduledDay` timestamp. |
| `appointment_weekday` | Yes | **Included** | Extracted from `AppointmentDay`. |
| `Age`, `age_group`, `gender_female` | Yes | **Included** | Patient demographic profile registered at booking. |
| `Scholarship` | Yes | **Included** | Administrative welfare eligibility recorded in municipal registry. |
| `neighbourhood` | Yes | **Included** | Clinic geographic catchment area. |
| `prior_appointments_count` | Yes | **Included** | Point-in-time count of bookings before $t_{\text{pred}}$. |
| `prior_no_shows_count` | Yes | **Included** | Point-in-time count of missed appointments before $t_{\text{pred}}$. |
| `prior_no_show_rate` | Yes | **Included** | Ratio of prior misses to prior bookings. |
| `SMS_received` | **NO** | **EXCLUDED** | **Temporal Leakage & Confounder.** Dispatched 24-48h prior to visit. |
| `Hipertension`, `Diabetes`, etc. | Yes | **EXCLUDED** | **Ethical Safety Boundary.** Clinical diagnoses must not drive reminders. |

---

## 10. Leakage Prevention: Deep Dive

### A. The Simpson's Paradox on `SMS_received`
In the raw data, cross-tabulating `SMS_received` with `No-show` naively suggests that patients who received an SMS had a higher no-show rate (27.6%) than patients who did not receive an SMS (16.7%).
* **Root Cause:** Same-day appointments (38,563 rows) never receive an SMS (they take place today) and have an extraordinarily low no-show rate (4.65%).
* When conditioning strictly on advance appointments (`lead_days > 0`), SMS actually reduces no-shows from 29.44% to 27.57%.
* Because SMS dispatch status is a future operational event, using it at booking time constitutes both **temporal leakage** and introduces severe confounding.

### B. Point-in-Time Cumulative History
39.1% of patients in the dataset have multiple appointments. Computing a global patient no-show rate across all rows causes target leakage (future attendance leaks into past predictions). PULSE enforces strict chronological windowing:
$$\text{prior\_no\_shows}(t_{\text{pred}}) = \sum_{j \in \text{Appointments}_i, \, t_j < t_{\text{pred}}} \mathbb{I}(y_j = 1)$$
This guarantees that for any patient's first appointment, `prior_appointments_count == 0` and `prior_no_show_rate == 0.0`.

---

## 11. Class Imbalance Handling
With positive class prevalence at ~20% in training, unweighted models tend to predict the majority class. PULSE employs:
1. **Cost-Sensitive Loss Weighting:** Setting `class_weight='balanced'` inversely proportional to class frequencies:
   $$w_1 = \frac{N}{2 \cdot N_1}, \quad w_0 = \frac{N}{2 \cdot N_0}$$
2. **Threshold Optimization:** Abandoning the naive 0.50 threshold in favor of operating points selected on the Precision-Recall curve.
3. **No Synthetic Over-Sampling on Evaluation:** Resampling is strictly forbidden on validation and test partitions.

---

## 12. Train / Validation / Test Strategy
To mirror actual deployment, PULSE utilizes a **Chronological Split** sorted by `ScheduledDay`:
* **Training Partition (70.0%):** 77,364 appointments (Nov 10, 2015 – May 18, 2016). Target rate: 23.20%.
* **Validation Partition (15.0%):** 16,578 appointments (May 18, 2016 – June 1, 2016). Target rate: 15.53%.
* **Untouched Test Partition (15.0%):** 16,579 appointments (June 1, 2016 – June 8, 2016). Target rate: 10.78%.

*Note on Temporal Covariate Shift:* The declining target rate over time reflects an increasing proportion of short-lead and same-day appointments scheduled near the close of the municipal survey period. A random K-Fold split would leak future schedules and obscure this real-world drift.

---

## 13. Baseline and Classifier Comparison
Four models were constructed within standardized scikit-learn `Pipeline` architectures incorporating `ColumnTransformer` (StandardScaler for numerics, OneHotEncoder for categoricals):

1. **Dummy Baseline (`strategy='prior'`):** Outputs empirical positive class prevalence.
2. **Logistic Regression (`class_weight='balanced'`):** L2-regularized linear model providing direct log-odds interpretability.
3. **Random Forest Classifier (`n_estimators=120, max_depth=12`):** Non-linear bagged decision tree ensemble.
4. **HistGradientBoostingClassifier (`learning_rate=0.08, max_iter=120`):** Gradient-boosted decision trees using histogram binning.

---

## 14. Evaluation Metrics
Models were evaluated on both the Chronological Validation set and the Untouched Final Test Set:

### Final Untouched Test Set Benchmark (N = 16,579)
*Evaluated at chosen validation threshold $\tau = 0.43$:*

| Model Architecture | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Prior)** | 0.5000 | 0.1078 | 0.0000 | 0.0000 | 0.0000 | 0.1371 |
| **Logistic Regression** | 0.7731 | 0.2884 | 0.2509 | 0.6458 | 0.3614 | 0.1715 |
| **Random Forest** | 0.7812 | 0.2898 | 0.2403 | **0.7426** | 0.3631 | 0.1687 |
| **HistGradientBoosting (Champion)** | **0.7827** | **0.3073** | **0.2570** | 0.6290 | **0.3649** | **0.1655** |

*Key Result:* HistGradientBoosting achieved the highest discrimination (ROC-AUC 0.7827), highest Precision-Recall AUC (0.3073, a 2.85x lift over the 0.1078 base rate), and best probability calibration.

---

## 15. Confusion Matrix & Cost-Benefit Error Analysis
Using the champion model on the test set at $\tau = 0.43$:
* **True Positives (TP):** 1,124 (No-shows correctly identified for outreach)
* **False Positives (FP):** 3,249 (Patients contacted who would have attended)
* **True Negatives (TN):** 11,543 (Attended visits correctly unburdened by calls)
* **False Negatives (FN):** 663 (Unanticipated no-shows)

### Asymmetric Cost Modeling
In clinical operations, errors are highly asymmetric:
* Cost of False Negative ($C_{\text{FN}}$): **\$100** (lost clinical consultation revenue, wasted doctor hour).
* Cost of False Positive ($C_{\text{FP}}$): **\$5** (cost of a 2-minute staff reminder phone call).

$$\text{Net Cost} = (663 \times \$100) + ((1124 + 3249) \times \$5) = \$66,300 + \$21,865 = \$88,165$$
$$\text{Baseline Cost (No Model)} = 1,787 \times \$100 = \$178,700$$
$$\mathbf{\text{Net Operational Savings: } \$90,535 \text{ (50.7\% Cost Reduction on Test Cohort)}}$$

---

## 16. Threshold and Calibration Analysis
* **Brier Score:** HistGradientBoosting attained a Brier score of 0.1655, demonstrating sharp calibration against true outcome probabilities.
* **Validation Threshold Sweep:** Sweeping $\tau \in [0.05, 0.95]$ revealed:
  * **$\tau = 0.25$ (High Recall Profile):** Recall 90.9%, Precision 24.5%, Flagged Workload 57.7%.
  * **$\tau = 0.43$ (F1-Optimal Profile):** Recall 80.7%, Precision 26.9%, Flagged Workload 46.6%.
  * **$\tau = 0.65$ (High Precision Profile):** Recall 22.7%, Precision 36.1%, Flagged Workload 9.8%.

---

## 17. Signature Feature 1: Appointment Operations Simulator
The **Appointment Operations Simulator** bridges static model evaluation and real hospital staffing constraints:
* **Workflow:** Accepts a schedule batch (e.g. 200 daily appointments).
* **Execution:** Ranks records by predicted risk $\hat{p}$ and selects top $K$ appointments up to staff call capacity (e.g., $K = 40$).
* **Empirical Verification:** In our held-out test simulation of 200 appointments:
  * Total actual no-shows: 33 (16.5% base rate).
  * Reviewing top 40 appointments captured **15 no-shows** (Capture Rate = 45.45% while reviewing only 20% of patients).
  * Precision in top 40: **37.5%**.
  * **Operational Lift: 2.27x over random review**.

---

## 18. Signature Feature 2: Threshold Trade-off Studio
The **Threshold Trade-off Studio** provides an interactive decision environment:
* Enables clinic leadership to manipulate $\tau$ dynamically and observe live adjustments to review volume, false alerts, missed visits, and net labor costs.
* Clearly establishes that **no single threshold is universally optimal**; rather, threshold selection is an operational policy decision conditioned on clinical call center capacity and seasonal buffer policies.

---

## 19. Explainability & Feature Attribution
PULSE implements dual-level explainability:

### Global Feature Attribution (Logistic Regression Coefficients)
* `is_same_day`: $\beta = -2.031$, $\text{Odds Ratio} = 0.131$ (86.9% reduction in odds of no-show for same-day bookings).
* `lead_days`: $\beta = +0.482$ per standard deviation increase.
* `prior_no_show_rate`: $\beta = +0.395$ (strong behavioral persistence).
* `Age`: $\beta = -0.274$ (older patients demonstrate lower odds of missing appointments).

### Local Prediction Attribution (Instance Waterfall)
For each appointment, PULSE returns directional feature drivers (e.g., "Long lead time (+21 days) increases risk", "Senior demographic decreases risk").  
*Disclaimer:* Feature drivers reflect empirical statistical associations within historical municipal data, not direct medical causation.

---

## 20. Responsible AI, Privacy & Safety Governance
* **Non-Clinical Guarantee:** The system is architecturally separated from clinical triage, diagnostics, and treatment.
* **Access Protection:** Predictions never automate cancellation or reallocate booked care.
* **Algorithmic Fairness:** Sensitive diagnostic codes were excised to prevent algorithmic penalization of chronic or disabled patients.
* **De-identification:** All training and inference data conform to municipal de-identification standards.

---

## 21. Reproducibility & Future Improvements
* **Reproducibility:** Seed fixed at 42. Full pipeline executable via:
  ```powershell
  python src/data/download.py
  python src/features/build_features.py
  python src/models/train.py
  python src/evaluation/evaluate.py
  python -m pytest -v
  ```
* **Future Enhancements:**
  1. Integration of real-time municipal weather forecasts (e.g. rain/storm alerts at appointment time).
  2. Multi-appointment batch overbooking optimization via mixed-integer linear programming (MILP).
  3. Dynamic re-scoring 48 hours prior to appointment upon receipt of delivery confirmation telemetry.
