# PULSE: Feature Availability, Data Leakage & Ethical Safety Audit

## 1. Prediction Point Definition

* **Prediction Unit:** Exactly one scheduled outpatient appointment.
* **Prediction Point ($t_{\text{pred}}$):** The moment of booking registration (`ScheduledDay`).
* **Prediction Horizon:** From booking timestamp up to the start of the scheduled appointment (`AppointmentDay`).
* **Target Variable ($y$):** Missed appointment indicator (`no_show`), where:
  * $y = 1$: Patient did not attend the scheduled slot without prior cancellation (No-Show).
  * $y = 0$: Patient attended the scheduled appointment.

---

## 2. Comprehensive Feature Audit Table

| Feature Name | Source Column | Type | Available at $t_{\text{pred}}$? | Audit Decision | Transformation / Engineering | Rationale & Safety Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`PatientId`** | `PatientId` | Integer / ID | Yes | **Derived Only** | Point-in-time historical aggregation ($t < t_{\text{pred}}$) | Raw ID leads to memorization and severe data leakage. Aggregated historical counts (prior appointments, prior no-show rate) computed strictly prior to current booking time. |
| **`AppointmentID`** | `AppointmentID` | Integer / ID | Yes | **Excluded** | Drop | Arbitrary system transaction surrogate key. No operational or behavioral signal. |
| **`Gender`** | `Gender` | Categorical | Yes | **Included** | Binary encoding (`F` $\to$ 1, `M` $\to$ 0) | Demographic attribute available in patient registry at booking. |
| **`ScheduledDay`** | `ScheduledDay` | Datetime | Yes | **Transformed** | Extract `booking_hour`, `booking_weekday`, `booking_month` | Booking timestamp is the exact anchor for $t_{\text{pred}}$. Captures time-of-day behavioral tendencies. |
| **`AppointmentDay`** | `AppointmentDay` | Datetime | Yes | **Transformed** | Extract `appointment_weekday`, `appointment_month`, `lead_days` | Agreed date of appointment. Lead time = difference in days between appointment date and booking date. |
| **`lead_days`** | Derived | Numeric (Int) | Yes | **Included** | $(\text{AppointmentDay} - \text{ScheduledDay})_{\text{days}}$ | Primary behavioral driver. Empirical findings show same-day no-show rate is 4.65% vs 28.52% for advance bookings. |
| **`Age`** | `Age` | Numeric (Int) | Yes | **Included** | Filter $Age \ge 0$, standard scaling | Valid demographic feature recorded at booking. 1 erroneous record with $Age = -1$ purged during cleaning. |
| **`Neighbourhood`** | `Neighbourhood` | Categorical | Yes | **Included** | Frequency / Target encoding with out-of-fold regularization | Clinic location district (81 locations). Represents clinic operational geography and transit accessibility. |
| **`Scholarship`** | `Scholarship` | Binary | Yes | **Included** | Keep as 0/1 | Bolsa Família social welfare enrollment. Socioeconomic proxy available at booking. |
| **`Hipertension`** | `Hipertension` | Binary | Yes | **Excluded** | Drop (Clinical Boundary) | **Ethical Safety Exclusion:** Clinical diagnosis. Using clinical conditions to schedule operational reminders risks regulatory violation and algorithmic bias. |
| **`Diabetes`** | `Diabetes` | Binary | Yes | **Excluded** | Drop (Clinical Boundary) | **Ethical Safety Exclusion:** Clinical diagnosis. Excluded to adhere strictly to non-clinical operational scope. |
| **`Alcoholism`** | `Alcoholism` | Binary | Yes | **Excluded** | Drop (Clinical Boundary) | **Ethical Safety Exclusion:** Sensitive health variable. Excluded under patient privacy and fairness guidelines. |
| **`Handcap`** | `Handcap` | Integer (0-4) | Yes | **Excluded** | Drop (Clinical Boundary) | **Ethical Safety Exclusion:** Disability severity/count. Excluded to prevent biased treatment of disabled patients. |
| **`SMS_received`** | `SMS_received` | Binary | **NO** | **Excluded (Temporal Leakage)** | Drop | **Data Leakage Exclusion:** Automated SMS reminders are dispatched 24–48 hours before the appointment date, weeks after $t_{\text{pred}}$! Using it at booking time leaks future hospital communication state. |
| **`No-show`** | `No-show` | Binary (Target) | **NO** | **Target Only** | Binary mapping: `'Yes'` $\to$ 1, `'No'` $\to$ 0 | Outcome revealed only after appointment concludes. Note Kaggle naming inversion trap: `'Yes'` means missed ($y=1$). |

---

## 3. Strict Safety & Ethical AI Principles

1. **Non-Clinical Guarantee:** The PULSE system never infers diagnoses, triage severity, treatment urgency, or clinical outcomes.
2. **Access-to-Care Protection:** The system output is solely an operational risk score used to direct human-reviewed reminder phone calls and optimize scheduling buffer capacity. It is strictly prohibited from being used to automatically cancel appointments, deprioritize patients, or deny medical care.
3. **No Future Leakage:** All engineered features must be mathematically verifiable as known strictly before or at the moment the appointment is logged.
