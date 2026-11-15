# PULSE: Demo Presentation Script & Technical Interview Defense Guide

---

## Part 1: The 3–5 Minute Executive Demonstration Script

*Use this exact script when introducing PULSE to the assessment panel. Deliver with calm, senior-level confidence.*

### Minute 0:00 – 0:45 | Problem Formulation & Ethical Safety Boundary
> "Good morning / afternoon. I am excited to present **PULSE**, an operational decision-support system designed to predict outpatient appointment no-shows and optimize hospital reminder workflows.
>
> In outpatient clinics, unannounced missed appointments leave expensive facilities and specialist doctors idle, while delaying care for waitlisted patients. 
> 
> Before showing the technology, I want to emphasize our non-negotiable safety boundary: PULSE is strictly an administrative decision tool. It **never** predicts clinical diagnosis, triage urgency, or treatment needs. Most importantly, it **never** automatically cancels appointments or denies care. Its sole purpose is to help staff prioritize human telephone reminder calls and size scheduling buffers."

### Minute 0:45 – 1:30 | Prediction Point, Dataset & Strict Leakage Audit
> "One of the most critical engineering decisions in this project is defining the exact prediction point: $t_{\text{pred}} = \text{Booking Registration Time}$ (`ScheduledDay`).
>
> Using 110,527 de-identified appointment records, we conducted a rigorous leakage audit. Many beginner solutions include the raw `SMS_received` column. In our audit, we proved this causes both temporal data leakage and a classic case of **Simpson's Paradox**. Clinics dispatch SMS reminders 24 to 48 hours before an appointment—weeks after booking. Furthermore, same-day appointments (35% of bookings) never receive SMS and have a 4.6% no-show rate, while advance bookings have a 28.5% no-show rate. Naively including SMS causes models to learn that reminders increase no-shows! We purged this leakage completely and also excised clinical diagnostic columns (`Hipertension`, `Diabetes`) to prevent algorithmic bias."

### Minute 1:30 – 2:30 | Chronological Splitting & Model Benchmarking
> "Rather than using random K-Fold cross validation—which would leak future patient attendance into the past—we implemented a strict **chronological split**: 70% train, 15% validation, and 15% untouched future test.
>
> We benchmarked a Dummy Prior baseline against Logistic Regression, Random Forest, and HistGradientBoosting pipelines with cost-sensitive `class_weight='balanced'`. 
>
> On our final untouched test set of 16,579 appointments, our champion HistGradientBoosting model achieved:
> * **ROC-AUC of 0.783**
> * **PR-AUC of 0.307**—a **2.85x lift** over the empirical baseline.
> * A low Brier calibration score of **0.165**, ensuring our probabilities represent real risk."

### Minute 2:30 – 3:45 | The Two Signature Features in Action
> "Now let's examine our two signature operational features in the interactive Streamlit dashboard:
>
> **1. The Appointment Operations Simulator:**
> In practice, clinics cannot call 1,000 patients a day. In our simulator, the user sets their daily staff review capacity—for example, 40 calls. The system scores the batch, ranks appointments by risk, and flags the top 40. On our historical held-out test data, calling just 20% of appointments captured **45.5% of all no-shows**, achieving a **2.27x lift over random calling** and generating projected net savings of \$250 daily.
>
> **2. The Threshold Trade-off Studio:**
> Rather than assuming an arbitrary 0.50 cutoff, the studio allows clinic directors to slide the threshold from 0.05 to 0.95. At $\tau = 0.25$, we achieve 90.9% recall for high-capacity teams. At $\tau = 0.43$, we maximize F1 and capture the bulk of cost savings, reducing unmanaged missed appointment losses by over 50%."

### Minute 3:45 – 4:30 | Explainability, FastAPI Backend & Architecture
> "Finally, PULSE delivers both global and local explainability. For any patient, coordinators see the exact directional drivers: lead time in days, prior point-in-time attendance history, and demographic patterns.
>
> The system is fully tested with 19 automated unit and integration tests, and exposed through a production-ready FastAPI service with `/predict`, `/predict/batch`, and `/metrics` endpoints.
>
> In summary, PULSE demonstrates that machine learning in healthcare operations must be technically rigorous, leakage-safe, explainable, and ethically grounded. Thank you, and I look forward to your questions."

---

## Part 2: The 15-Minute Technical Deep-Dive (Interview Q&A)

### Q1: "Why did you choose Booking Time as your prediction point instead of 24 hours before the appointment?"
**Candidate Answer:**
> "Booking time is the earliest and most operationally valuable moment. When an appointment is scheduled weeks in advance, the hospital can immediately decide its communication cadence—such as scheduling multi-tier reminder messages, flagging the appointment for a human phone call, or reserving a double-booking buffer. If you wait until 24 hours prior, you have missed the window to rebook open slots or plan staffing buffers. Furthermore, booking time gives us an unambiguous boundary: any feature timestamped after booking is strictly excluded from training."

### Q2: "Why didn't you use random K-Fold cross validation?"
**Candidate Answer:**
> "Random K-fold cross-validation on time-series appointment data introduces severe data leakage and produces overly optimistic evaluations. In our dataset, 39.1% of patients have multiple appointments. If you split randomly, future visits of a patient appear in training while past visits appear in testing. Even more critically, clinical scheduling undergoes temporal covariate shift: early survey periods had long lead times, while later weeks had more same-day walk-ins. A chronological split (training on earlier dates, validating on middle dates, testing on future dates) is the only defensible way to simulate how the model will perform in production."

### Q3: "Explain how you computed your historical patient features without data leakage."
**Candidate Answer:**
> "If you calculate a patient's total no-show rate across the entire dataset, you leak their future behavior into their first appointment! To prevent this, we sorted the dataset chronologically by `ScheduledDay`. We grouped by `PatientId` and used cumulative sums shifted by 1:
> $$\text{prior\_appointments} = \text{cumcount}()$$
> $$\text{prior\_no\_shows} = \text{cumsum}(\text{target}) - \text{target}$$
> For any patient's first appointment, `prior_appointments` is exactly 0 and `prior_no_show_rate` is 0.0. Every feature only incorporates events strictly prior to $t_{\text{pred}}$."

### Q4: "Why did you exclude clinical features like Diabetes and Hypertension?"
**Candidate Answer:**
> "We established a strict ethical and operational safety boundary: our objective is administrative decision support, not clinical triage. Using chronic disease diagnoses to prioritize reminder calls or overbooking introduces severe algorithmic fairness risks and regulatory liabilities under health privacy standards. Operational scheduling friction—such as lead time, day of the week, and past scheduling compliance—provides strong predictive power without penalizing patients based on their medical conditions."

### Q5: "How do you handle class imbalance, and why is accuracy a poor metric here?"
**Candidate Answer:**
> "Our positive class (missed appointments) accounts for only 20% in training and 10.8% in the test set. A naive model predicting 'Attended' for every appointment would achieve 89.2% accuracy on the test set while catching zero no-shows! We handled class imbalance through:
> 1. `class_weight='balanced'` in our loss functions.
> 2. Evaluating on PR-AUC (Precision-Recall AUC) and Brier calibration score rather than accuracy.
> 3. Systematic threshold tuning in our Threshold Trade-off Studio using asymmetric cost matrices ($C_{\text{FN}} = \$100$ vs $C_{\text{FP}} = \$5$)."

---

## Part 3: Live Coding & Debugging Task Preparation (10-Minute Challenge)

*Interviewers often ask candidates to make live modifications. Here are the three most common live tasks and how to execute them cleanly:*

### Task 1: "Change the decision threshold in the API to prioritize high recall."
* **Where to modify:** `src/api/main.py`
* **How to modify:** In `@app.post("/predict")`, the default threshold parameter is currently `threshold: float = Query(default=0.43)`.
* **Action:** Update default to `0.25` (the high-recall threshold from our validation sweep) or show how passing `?threshold=0.25` in the URL query overrides it immediately without code redeployment.

### Task 2: "Add a new operational feature: is_weekend_appointment."
* **Where to modify:** `src/features/build_features.py` and `src/models/pipeline.py`.
* **Code modification:**
  ```python
  # In build_features.py:
  df['is_weekend_appointment'] = df['appointment_weekday'].isin([5, 6]).astype(int)
  
  # In pipeline.py:
  BINARY_FEATURES.append('is_weekend_appointment')
  ```
* **Verify:** Re-run `python -m pytest tests/test_leakage_and_features.py` to confirm pipeline compatibility.

### Task 3: "Update the cost ratio in the Operations Simulator."
* **Where to modify:** `src/evaluation/simulator.py` or through the interactive slider in Streamlit.
* **Explanation:** Show that `simulate_batch(df, cost_per_noshow=150.0, cost_per_call=8.0)` accepts dynamic cost parameters directly, allowing instant financial recalculation.
