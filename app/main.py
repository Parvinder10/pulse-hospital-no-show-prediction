"""
PULSE: Hospital Appointment No-Show Prediction & Operational Decision Support
Interactive Clinical Operations Dashboard (Streamlit)
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.pipeline import ALL_MODEL_FEATURES
from src.evaluation.simulator import AppointmentOperationsSimulator
from src.threshold.tradeoff import ThresholdTradeoffStudio
from src.explainability.explainer import ModelExplainer

st.set_page_config(
    page_title="PULSE - Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")

@st.cache_resource
def load_core_systems():
    explainer = ModelExplainer()
    simulator = AppointmentOperationsSimulator()
    studio = ThresholdTradeoffStudio()
    
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
    return explainer, simulator, studio, metrics, metadata

@st.cache_data
def load_sample_test_batch(n: int = 250):
    if os.path.exists(TEST_DATA_PATH):
        df = pd.read_csv(TEST_DATA_PATH)
        return df.head(n)
    return pd.DataFrame()

explainer, simulator, studio, metrics, metadata = load_core_systems()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/hospital.png", width=64)
st.sidebar.title("PULSE Navigation")
st.sidebar.caption("Predict Risk. Plan Responsibly.")

menu = st.sidebar.radio(
    "Select System Module:",
    [
        "1. Executive Overview & Governance",
        "2. EDA & Feature Leakage Audit",
        "3. Model Benchmarks & Calibration",
        "4. Single Appointment Predictor",
        "5. Operations Simulator (Signature)",
        "6. Threshold Trade-off Studio (Signature)",
        "7. Safety & Responsible AI Guidelines"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Operational Boundary Guarantee:**\n"
    "• Supports staff reminder prioritization only.\n"
    "• Never cancels care or triages patients.\n"
    "• Prediction Point: Booking Registration."
)

# =============================================================================
# MODULE 1: EXECUTIVE OVERVIEW & GOVERNANCE
# =============================================================================
if menu == "1. Executive Overview & Governance":
    st.title("🏥 PULSE: Hospital Appointment No-Show Prediction System")
    st.subheader("Operational Decision Support for Clinical Scheduling & Outreach")
    
    st.markdown("""
    ### Business & Operational Mission
    Unannounced patient no-shows waste high-value clinical facilities, leave physician hours idle, and elongate waitlists for patients requiring urgent consultations.
    
    **PULSE** (*Predict risk. Plan responsibly.*) equips hospital administrative teams with calibrated, point-in-time predictive probabilities of appointment no-shows. By forecasting risk at the exact moment of appointment registration, coordinators can proactively target human telephone confirmation and configure overbooking buffers.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical Training Records", "77,364", "Chronological Split")
    col2.metric("Untouched Future Test Set", "16,579", "Zero Data Leakage")
    col3.metric("Champion Test ROC-AUC", "0.783", "HistGradientBoosting")
    col4.metric("Operational Lift in Top 20%", "2.27x", "Over Random Outreach")
    
    st.markdown("---")
    st.markdown("""
    ### 🛡️ Strict Operational Safety Boundaries
    1. **Strictly Non-Clinical:** PULSE never predicts diagnosis, clinical severity, treatment urgency, or clinical outcomes.
    2. **Zero Automated Adverse Actions:** PULSE **never** automatically cancels appointments, denies access to care, or deprioritizes vulnerable patients.
    3. **Human-in-the-Loop:** All high-risk flags are routed to hospital administrative staff solely for personal outreach (e.g. phone call to arrange transportation or confirm attendance).
    4. **Booking-Time Integrity:** All input features are mathematically verified as known at or before appointment scheduling (`ScheduledDay`).
    """)
    
    st.markdown("### System Architecture")
    st.code("""
    [ Raw EMR Scheduling Log ] 
              │
              ▼
    [ Cleaning & Point-in-Time Historical Engine ] (Purges anomalies, calculates past no-show rates strictly prior to t_pred)
              │
              ▼
    [ Feature Availability & Leakage Guard ] (Drops SMS_received, drops clinical diagnosis codes)
              │
              ▼
    [ Scikit-Learn Preprocessing & Pipeline ] (StandardScaler + OneHotEncoder)
              │
              ▼
    [ Champion Classifier: HistGradientBoosting ] (Calibrated risk probability: p_hat in [0, 1])
              │
              ├─────────────────────────────┬─────────────────────────────┐
              ▼                             ▼                             ▼
    [ FastAPI Backend ]        [ Operations Simulator ]      [ Threshold Studio ]
    (/predict & /batch)        (Capacity-constrained review)  (Workload trade-off)
    """, language="text")

# =============================================================================
# MODULE 2: EDA & FEATURE LEAKAGE AUDIT
# =============================================================================
elif menu == "2. EDA & Feature Leakage Audit":
    st.title("📊 Exploratory Data Analysis & Leakage Audit")
    
    tab1, tab2, tab3 = st.tabs(["Operational Patterns & Findings", "Simpson's Paradox on SMS", "Feature Availability Table"])
    
    with tab1:
        st.markdown("### Empirical Discoveries in Historical Appointment Data")
        c1, c2 = st.columns(2)
        target_fig_path = os.path.join(FIGURES_DIR, "target_distribution.png")
        if os.path.exists(target_fig_path):
            c1.image(target_fig_path, caption="Target Distribution (Chronological Training Set)")
            
        lead_fig_path = os.path.join(FIGURES_DIR, "lead_time_vs_noshow.png")
        if os.path.exists(lead_fig_path):
            c2.image(lead_fig_path, caption="Lead Time Horizon vs. No-Show Rate")
            
        c3, c4 = st.columns(2)
        age_fig_path = os.path.join(FIGURES_DIR, "age_distribution.png")
        if os.path.exists(age_fig_path):
            c3.image(age_fig_path, caption="Patient Age Density by Appointment Attendance")
            
        day_fig_path = os.path.join(FIGURES_DIR, "weekday_pattern.png")
        if os.path.exists(day_fig_path):
            c4.image(day_fig_path, caption="Appointment Attendance Across Weekdays")
            
    with tab2:
        st.markdown("### ⚠️ The Simpson's Paradox & Temporal Leakage Trap on `SMS_received`")
        st.warning(
            "**Why SMS_received MUST be purged from a booking-time prediction model:**\n\n"
            "1. **Temporal Leakage:** Hospitals send automated SMS reminders 24-48 hours before the appointment, weeks after booking time. Using it at booking leaks future events.\n"
            "2. **Confounding & Inversion:** In the raw dataset, same-day appointments (35% of all bookings) NEVER receive an SMS because they occur immediately, and have an ultra-low 4.65% no-show rate. Advance bookings (which have a 28.5% no-show rate) receive SMS. A naive model interprets SMS as causing missed appointments!"
        )
        
        simpson_data = pd.DataFrame({
            "Booking Type": ["Same-Day (Lead = 0 days)", "Advance (Lead > 0 days, No SMS)", "Advance (Lead > 0 days, SMS Sent)"],
            "Appointments": [38563, 36482, 35482],
            "SMS Dispatched": ["0%", "0%", "100%"],
            "Observed No-Show Rate": ["4.65%", "29.44%", "27.57%"]
        })
        st.table(simpson_data)
        st.caption("When evaluated correctly within advance bookings, SMS actually slightly decreases no-shows (29.4% -> 27.6%), but naive correlation across the whole dataset inverts this relationship completely.")

    with tab3:
        st.markdown("### Formal Feature Availability & Leakage Audit Matrix")
        audit_file = os.path.join(BASE_DIR, "reports", "feature_leakage_audit.md")
        if os.path.exists(audit_file):
            with open(audit_file, 'r', encoding='utf-8') as f:
                st.markdown(f.read())

# =============================================================================
# MODULE 3: MODEL BENCHMARKS & CALIBRATION
# =============================================================================
elif menu == "3. Model Benchmarks & Calibration":
    st.title("📈 Model Comparison, Calibration & Test Set Benchmark")
    
    st.markdown("""
    We rigorously trained a **Dummy Baseline** and three distinct supervised classifiers on the chronological training set (77,364 appointments) and evaluated them on the **untouched chronological test set (16,579 appointments)**.
    """)
    
    if metrics and "test_model_comparison" in metrics:
        comp_df = pd.DataFrame(metrics["test_model_comparison"]).T
        st.dataframe(comp_df.style.highlight_max(subset=['roc_auc', 'pr_auc', 'f1_score'], color='#d4edda'), use_container_width=True)
        
    c1, c2 = st.columns(2)
    roc_pr_path = os.path.join(FIGURES_DIR, "roc_pr_curves.png")
    if os.path.exists(roc_pr_path):
        c1.image(roc_pr_path, caption="ROC Curves & Precision-Recall Curves (Validation Set)")
        
    cal_path = os.path.join(FIGURES_DIR, "calibration_curve.png")
    if os.path.exists(cal_path):
        c2.image(cal_path, caption="Reliability Diagram (Probability Calibration Curves)")
        
    st.markdown("---")
    st.markdown("### Final Untouched Test Confusion Matrix")
    c3, c4 = st.columns([1, 1])
    cm_path = os.path.join(FIGURES_DIR, "test_confusion_matrix.png")
    if os.path.exists(cm_path):
        c3.image(cm_path, caption="Test Confusion Matrix at Chosen Threshold tau = 0.43")
        
    with c4:
        st.markdown("""
        **Operational Error Interpretation:**
        * **False Positive (FP):** Staff calls a patient who would have attended anyway.
          * *Consequence:* ~$5 of administrative call time spent. Patient receives helpful reminder.
        * **False Negative (FN):** Model fails to flag an appointment that ends up being missed.
          * *Consequence:* ~$100 clinic slot cost lost; physician and examination room sit idle.
        * **Conclusion:** Because $C_{\\text{FN}} \\gg C_{\\text{FP}}$, the optimal operational threshold must be tuned toward higher Recall rather than high Precision!
        """)

# =============================================================================
# MODULE 4: SINGLE APPOINTMENT PREDICTOR
# =============================================================================
elif menu == "4. Single Appointment Predictor":
    st.title("🔮 Single Appointment Risk Inference & Explainability")
    st.markdown("Simulate an appointment booking transaction in real-time to obtain the calibrated no-show risk score and feature breakdown.")
    
    with st.form("single_predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            lead_days = st.slider("Lead Time (Days between booking & visit):", min_value=0, max_value=90, value=14)
            is_same_day = 1 if lead_days == 0 else 0
            age = st.slider("Patient Age:", min_value=0, max_value=105, value=28)
            gender = st.selectbox("Gender:", ["Female (F)", "Male (M)"])
            gender_female = 1 if "Female" in gender else 0
            
        with col2:
            booking_hour = st.slider("Booking Hour (0-23):", 7, 20, 11)
            booking_day = st.selectbox("Booking Weekday:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5}
            booking_weekday = day_map[booking_day]
            appt_day = st.selectbox("Appointment Weekday:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
            appointment_weekday = day_map[appt_day]
            
        with col3:
            scholarship = st.checkbox("Bolsa Família Welfare Enrollment (Scholarship)", value=False)
            prior_count = st.number_input("Prior Scheduled Appointments for Patient:", min_value=0, max_value=50, value=2)
            prior_noshows = st.number_input("Prior Missed Appointments (No-Shows):", min_value=0, max_value=prior_count, value=1)
            neighbourhood = st.selectbox("Clinic Catchment District:", ["JARDIM DA PENHA", "JESUS DE NAZARETH", "MARIA ORTIZ", "CENTRO", "RESISTÊNCIA", "SANTO ANTÔNIO"])
            
        threshold_val = st.slider("Operational Alert Threshold (tau):", min_value=0.10, max_value=0.90, value=0.43, step=0.01)
        submit_btn = st.form_submit_button("Compute Appointment Risk", type="primary")
        
    if submit_btn:
        prior_rate = (prior_noshows / prior_count) if prior_count > 0 else 0.0
        lead_weeks = round(lead_days / 7.0, 2)
        
        # Age group
        if age <= 12:
            ag = 'Child'
        elif age <= 24:
            ag = 'Youth'
        elif age <= 45:
            ag = 'YoungAdult'
        elif age <= 65:
            ag = 'MiddleAged'
        else:
            ag = 'Senior'
            
        record = {
            'lead_days': lead_days,
            'is_same_day': is_same_day,
            'lead_weeks': lead_weeks,
            'booking_hour': booking_hour,
            'booking_weekday': booking_weekday,
            'booking_month': 5,
            'appointment_weekday': appointment_weekday,
            'appointment_month': 5,
            'Age': age,
            'age_group': ag,
            'gender_female': gender_female,
            'Scholarship': 1 if scholarship else 0,
            'neighbourhood': neighbourhood,
            'prior_appointments_count': prior_count,
            'prior_no_shows_count': prior_noshows,
            'prior_no_show_rate': prior_rate,
            'is_repeat_patient': 1 if prior_count > 0 else 0
        }
        
        explanation = explainer.explain_instance(record)
        prob = explanation["predicted_risk_probability"]
        flagged = prob >= threshold_val
        
        st.markdown("### Prediction Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Predicted No-Show Risk", f"{prob*100:.1f}%", f"{'+' if flagged else '-'} Threshold: {threshold_val*100:.0f}%")
        res_col2.metric("Risk Classification Tier", explanation["risk_tier"])
        res_col3.metric("Decision Flag", "FLAGGED FOR OUTREACH" if flagged else "ROUTINE APPOINTMENT")
        
        if flagged:
            st.warning(f"**Recommended Action:** {explanation['recommended_operational_action']}")
        else:
            st.success(f"**Recommended Action:** {explanation['recommended_operational_action']}")
            
        st.markdown("#### Local Feature Attribution Drivers")
        for d in explanation["top_drivers"]:
            icon = "🔺" if d["direction"] == "increases_risk" else "🔻"
            color = "#fff3cd" if d["direction"] == "increases_risk" else "#d4edda"
            st.markdown(f"> {icon} **{d['factor']}** — *{d['impact']}*")
            
        st.caption(explanation["disclaimer"])

# =============================================================================
# MODULE 5: SIGNATURE FEATURE 1 - OPERATIONS SIMULATOR
# =============================================================================
elif menu == "5. Operations Simulator (Signature)":
    st.title("⚙️ Signature Feature 1: Appointment Operations Simulator")
    st.markdown("""
    **Simulate real-world clinic workflows under constrained administrative capacity.**
    Hospitals cannot call every patient. Staff coordinators have limited outbound calling hours.
    This component ranks scheduled appointments by risk, flags the top $K$ appointments matching your daily staff capacity, and evaluates captured no-shows and operational lift.
    """)
    
    df_sample = load_sample_test_batch(200)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("#### Staff Capacity Settings")
        daily_capacity = st.slider("Daily Staff Call Capacity (K):", min_value=10, max_value=100, value=40, step=5)
        cost_noshow = st.number_input("Lost Revenue per Empty Slot ($):", min_value=20.0, max_value=500.0, value=100.0)
        cost_call = st.number_input("Cost per Outbound Review Call ($):", min_value=1.0, max_value=50.0, value=5.0)
        phone_recovery = st.slider("Assumed Phone Call Attendance Recovery Rate (%):", min_value=10, max_value=60, value=30)
        
    with c2:
        if not df_sample.empty:
            sim_res = simulator.simulate_batch(
                df_sample,
                daily_capacity=daily_capacity,
                cost_per_noshow=cost_noshow,
                cost_per_call=cost_call
            )
            m = sim_res["metrics"]
            
            st.markdown("#### Operational Simulation Outcomes (Batch of 200 Appointments)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Appointments Reviewed", f"{m['appointments_reviewed']} / {m['total_appointments']}", f"{m['review_coverage_pct']}% of schedule")
            m2.metric("No-Shows Captured in Top K", f"{m['captured_noshows_in_top_k']} / {m['total_actual_noshows']}", f"{m['capture_rate_pct']}% capture rate")
            m3.metric("Precision in Top K", f"{m['precision_in_top_k_pct']}%", f"{m['lift_over_random']}x Lift Over Random")
            
            st.success(f"**Net Projected Daily Cost Savings:** ${m['estimated_net_savings_usd']:,.2f} (Accounting for staff call labor and recovered clinic slots)")
            
    st.markdown("---")
    st.markdown("### Top Prioritized Appointments for Human Review")
    if not df_sample.empty:
        display_cols = ['rank', 'predicted_risk', 'risk_tier', 'lead_days', 'Age', 'prior_appointments_count', 'prior_no_shows_count', 'recommended_action']
        st.dataframe(sim_res["selected_records"][display_cols].style.format({'predicted_risk': '{:.1%}'}), use_container_width=True)

# =============================================================================
# MODULE 6: SIGNATURE FEATURE 2 - THRESHOLD TRADE-OFF STUDIO
# =============================================================================
elif menu == "6. Threshold Trade-off Studio (Signature)":
    st.title("🎚️ Signature Feature 2: Threshold Trade-off Studio")
    st.markdown("""
    Explore how changing the probability classification threshold $\\tau$ alters the operational trade-off between **identifying missed appointments (Recall)** and **wasting administrative outreach on patients who would show up anyway (Precision & Workload)**.
    """)
    
    col_ctrl, col_stats = st.columns([1, 2])
    with col_ctrl:
        dataset_choice = st.radio("Evaluation Split:", ["validation", "test"], index=0)
        tau = st.slider("Select Probability Threshold (tau):", min_value=0.05, max_value=0.95, value=0.43, step=0.01)
        cost_slot = st.number_input("Slot Cost ($):", value=100.0)
        cost_labor = st.number_input("Outreach Labor ($):", value=5.0)
        
    res_thresh = studio.evaluate_threshold(tau, dataset=dataset_choice, cost_fn=cost_slot, cost_fp=cost_labor)
    
    with col_stats:
        st.markdown(f"#### Performance at tau = {tau:.2f} ({dataset_choice.capitalize()} Split)")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Precision", f"{res_thresh['precision']*100:.1f}%")
        b2.metric("Recall", f"{res_thresh['recall']*100:.1f}%")
        b3.metric("F1-Score", f"{res_thresh['f1_score']:.3f}")
        b4.metric("Flagged Volume", f"{res_thresh['flagged_percent']}%", f"{res_thresh['flagged_for_review']:,} calls")
        
        st.info(f"**Trade-off Diagnosis:** {res_thresh['tradeoff_summary']}")
        
    c_cm, c_cost = st.columns(2)
    with c_cm:
        cm_data = res_thresh["confusion_matrix"]
        cm_matrix = np.array([[cm_data["tn"], cm_data["fp"]], [cm_data["fn"], cm_data["tp"]]])
        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Attendance", y="Actual Attendance", color="Count"),
            x=['Attended (0)', 'No-Show (1)'],
            y=['Attended (0)', 'No-Show (1)'],
            text_auto=True,
            color_continuous_scale='Blues',
            title=f"Confusion Matrix (tau = {tau:.2f})"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with c_cost:
        st.markdown("#### Continuous Trade-off Curves")
        tradeoff_curve_path = os.path.join(FIGURES_DIR, "threshold_tradeoff.png")
        if os.path.exists(tradeoff_curve_path):
            st.image(tradeoff_curve_path, caption="Precision, Recall, Workload, and Cost vs. Threshold")

# =============================================================================
# MODULE 7: SAFETY & RESPONSIBLE AI GUIDELINES
# =============================================================================
elif menu == "7. Safety & Responsible AI Guidelines":
    st.title("🛡️ Ethical AI, Patient Safety & Clinical Non-Interference")
    st.markdown("""
    ### PULSE Ethical Charter & Assessment Defense
    
    1. **Strictly Non-Clinical Decision Support**
       * The model predicts an administrative event (attendance vs unannounced vacancy).
       * It never predicts a disease diagnosis, clinical prognosis, triage urgency, or treatment plan.
       * Clinical care decisions remain 100% with qualified medical personnel.
       
    2. **Access-to-Care Guarantee (No Automated Deprioritization)**
       * Under no circumstances can model scores be used to deny service, cancel appointments, or de-prioritize care.
       * High-risk scores only trigger **supportive interventions** (e.g., proactive reminder calls, transit assistance inquiry).
       
    3. **Algorithmic Fairness & Bias Mitigation**
       * We explicitly excluded chronic disease flags (`Hipertension`, `Diabetes`, `Handcap`) to prevent algorithms from learning discriminatory associations between medical vulnerability and scheduling penalization.
       
    4. **Data Privacy & De-Identification**
       * Trained strictly on de-identified, publicly licensed data under municipal open data standards.
       * No private medical charts or protected health information (PHI) are accessed or stored.
    """)
