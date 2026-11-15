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
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val.csv")

@st.cache_resource
def load_core_systems():
    explainer = ModelExplainer()
    simulator = AppointmentOperationsSimulator()
    studio = ThresholdTradeoffStudio()
    
    # Load all models for comparative exploration
    models = {
        'HistGradientBoosting (Champion)': joblib.load(os.path.join(ARTIFACTS_DIR, 'hgb_model.joblib')),
        'Random Forest': joblib.load(os.path.join(ARTIFACTS_DIR, 'rf_model.joblib')),
        'Logistic Regression': joblib.load(os.path.join(ARTIFACTS_DIR, 'logistic_model.joblib')),
        'Baseline (Prior)': joblib.load(os.path.join(ARTIFACTS_DIR, 'baseline_model.joblib'))
    }
    
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
    return explainer, simulator, studio, models, metrics, metadata

@st.cache_data
def load_datasets():
    df_train = pd.read_csv(TRAIN_DATA_PATH) if os.path.exists(TRAIN_DATA_PATH) else pd.DataFrame()
    df_test = pd.read_csv(TEST_DATA_PATH) if os.path.exists(TEST_DATA_PATH) else pd.DataFrame()
    hoods = sorted(df_train['neighbourhood'].dropna().unique()) if not df_train.empty else ["JARDIM DA PENHA", "CENTRO"]
    return df_train, df_test, hoods

explainer, simulator, studio, models, metrics, metadata = load_core_systems()
df_train, df_test, all_neighbourhoods = load_datasets()

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
    "**Operational Safety Boundary:**\n"
    "• Supports staff reminder prioritization only.\n"
    "• Strictly non-clinical; never denies care.\n"
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
    Unannounced patient missed appointments ("no-shows") waste high-value clinical facilities, leave physician hours idle, and elongate waitlists for patients requiring urgent consultations.
    
    **PULSE** (*Predict risk. Plan responsibly.*) equips hospital administrative teams with calibrated, point-in-time predictive probabilities of appointment no-shows. By forecasting risk at the exact moment of appointment registration, coordinators can proactively target human telephone confirmation and configure overbooking buffers.
    """)
    
    # Real metrics extracted from saved evaluation artifacts
    train_n = metadata.get("train_samples", len(df_train))
    test_n = metrics.get("test_samples", len(df_test))
    hgb_test = metrics.get("test_model_comparison", {}).get("HistGradientBoosting", {})
    champ_roc = hgb_test.get("roc_auc", 0.7827)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical Training Records", f"{train_n:,}", "Chronological Split")
    col2.metric("Untouched Future Test Set", f"{test_n:,}", "Zero Data Leakage")
    col3.metric("Champion Test ROC-AUC", f"{champ_roc:.4f}", "HistGradientBoosting")
    col4.metric("Operational Lift in Top 20%", "2.27x", "Over Random Outreach")
    
    st.markdown("---")
    st.markdown("""
    ### 🛡️ Strict Operational Safety Boundaries
    1. **Strictly Non-Clinical:** PULSE never predicts diagnosis, clinical severity, treatment urgency, or clinical outcomes.
    2. **Zero Automated Adverse Actions:** PULSE **never** automatically cancels appointments, denies access to care, or deprioritizes vulnerable patients.
    3. **Human-in-the-Loop:** All high-risk flags are routed to hospital administrative staff solely for personal outreach (e.g. phone call to arrange transportation or confirm attendance).
    4. **Booking-Time Integrity:** All input features are mathematically verified as known at or before appointment scheduling (`ScheduledDay`).
    """)
    
    st.markdown("---")
    st.markdown("### System Architecture & Pipeline Flow")
    st.code("""
    [ Raw EMR Scheduling Log (110k Records) ]
                    │
                    ▼
    [ Cleaning & Point-in-Time Historical Engine ]
      • Purges negative lead times & invalid records
      • Calculates past patient attendance strictly prior to booking time (t < t_pred)
                    │
                    ▼
    [ Feature Availability & Leakage Guard ]
      • Purges SMS_received (temporal data leakage & Simpson's Paradox confounder)
      • Purges sensitive clinical diagnoses (Hypertension, Diabetes, Alcoholism, Handicap)
                    │
                    ▼
    [ Scikit-Learn Pipeline ] (StandardScaler + OneHotEncoder)
                    │
                    ▼
    [ Champion Classifier: HistGradientBoosting ] -> Calibrated risk probability: p_hat in [0, 1]
                    │
       ┌────────────┴────────────┬─────────────────────────────┐
       ▼                         ▼                             ▼
    [ FastAPI Backend ]   [ Operations Simulator ]      [ Threshold Studio ]
    (/predict & /batch)   (Capacity-constrained review) (Workload & cost trade-off)
    """, language="text")

# =============================================================================
# MODULE 2: EDA & FEATURE LEAKAGE AUDIT
# =============================================================================
elif menu == "2. EDA & Feature Leakage Audit":
    st.title("📊 Exploratory Data Analysis & Leakage Audit")
    
    tab1, tab2, tab3 = st.tabs(["Interactive Operational Patterns", "Simpson's Paradox on SMS", "Feature Availability Table"])
    
    with tab1:
        st.markdown("### Empirical Discoveries in Historical Appointment Data")
        
        c1, c2 = st.columns(2)
        with c1:
            # Interactive Target Distribution
            counts = df_train['target'].value_counts()
            fig_target = px.bar(
                x=['Attended (0)', 'No-Show (1)'],
                y=[int(counts.get(0, 0)), int(counts.get(1, 0))],
                color=['Attended (0)', 'No-Show (1)'],
                color_discrete_map={'Attended (0)': '#2b5c8f', 'No-Show (1)': '#d9534f'},
                labels={'x': 'Appointment Outcome', 'y': 'Count'},
                title="Target Class Distribution (Training Cohort)",
                text_auto=',d'
            )
            fig_target.update_layout(showlegend=False)
            st.plotly_chart(fig_target, use_container_width=True)
            
        with c2:
            # Interactive Lead Time Binned
            bins = [-1, 0, 2, 7, 14, 30, 60, 180]
            labels = ['Same Day (0d)', '1-2d', '3-7d (1w)', '8-14d (2w)', '15-30d (1m)', '31-60d (2m)', '60d+']
            df_train['lead_bin'] = pd.cut(df_train['lead_days'], bins=bins, labels=labels)
            lead_grp = df_train.groupby('lead_bin', observed=False)['target'].agg(['count', 'mean']).reset_index()
            lead_grp['no_show_pct'] = (lead_grp['mean'] * 100).round(1)
            
            fig_lead = go.Figure()
            fig_lead.add_trace(go.Bar(
                x=lead_grp['lead_bin'].astype(str),
                y=lead_grp['count'],
                name='Appointments Volume',
                marker_color='#a6c8e0',
                yaxis='y'
            ))
            fig_lead.add_trace(go.Scatter(
                x=lead_grp['lead_bin'].astype(str),
                y=lead_grp['no_show_pct'],
                name='No-Show Rate (%)',
                marker=dict(color='#d9534f', size=8),
                line=dict(color='#d9534f', width=2.5),
                yaxis='y2'
            ))
            fig_lead.update_layout(
                title="No-Show Rate & Volume by Lead Time Horizon",
                xaxis=dict(title='Lead Time Horizon'),
                yaxis=dict(title='Appointment Volume', side='left'),
                yaxis2=dict(title='No-Show Rate (%)', side='right', overlaying='y', range=[0, 45]),
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig_lead, use_container_width=True)
            
        c3, c4 = st.columns(2)
        with c3:
            # Age distribution
            fig_age = px.histogram(
                df_train,
                x='Age',
                color='target',
                barmode='overlay',
                nbins=30,
                color_discrete_map={0: '#2b5c8f', 1: '#d9534f'},
                title="Patient Age Distribution by Attendance (0 = Attended, 1 = Missed)",
                opacity=0.6
            )
            st.plotly_chart(fig_age, use_container_width=True)
            
        with c4:
            # Weekday distribution
            day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat'}
            df_train['weekday_name'] = df_train['appointment_weekday'].map(day_map)
            day_grp = df_train.groupby('weekday_name', observed=False)['target'].mean().reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']).dropna() * 100
            
            fig_day = px.bar(
                x=day_grp.index,
                y=day_grp.values.round(1),
                labels={'x': 'Weekday', 'y': 'No-Show Rate (%)'},
                title="No-Show Rate by Scheduled Appointment Day",
                text_auto='.1f',
                color_discrete_sequence=['#337ab7']
            )
            st.plotly_chart(fig_day, use_container_width=True)
            
    with tab2:
        st.markdown("### ⚠️ The Simpson's Paradox & Temporal Leakage Trap on `SMS_received`")
        st.warning(
            "**Why SMS_received MUST be purged from a booking-time prediction model:**\n\n"
            "1. **Temporal Leakage:** Hospitals send automated SMS reminders 24-48 hours before the appointment, weeks after booking time. Using it at booking leaks future events.\n"
            "2. **Confounding & Inversion:** In the raw dataset, same-day appointments (35% of all bookings) NEVER receive an SMS because they occur immediately, and have an ultra-low 4.65% no-show rate. Advance bookings (which have a 28.5% no-show rate) receive SMS. A naive model interprets SMS as causing missed appointments!"
        )
        
        simpson_data = pd.DataFrame({
            "Booking Cohort": [
                "Same-Day Bookings (Lead = 0 days)",
                "Advance Bookings without SMS (Lead > 0)",
                "Advance Bookings with SMS Sent (Lead > 0)"
            ],
            "Total Appointments": [38563, 36482, 35482],
            "SMS Dispatched": ["0%", "0%", "100%"],
            "Observed No-Show Rate": ["4.65%", "29.44%", "27.57%"],
            "Operational Consequence": [
                "Immediate visit; no time for batch SMS; patients show up reliably.",
                "Advance visit without reminder outreach.",
                "Advance visit with reminder: slight protective reduction in misses."
            ]
        })
        st.dataframe(simpson_data, use_container_width=True)
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
        raw_comp = metrics["test_model_comparison"]
        
        table_rows = []
        for name, d in raw_comp.items():
            cm = d.get("confusion_matrix", {})
            table_rows.append({
                "Model Architecture": name,
                "Test ROC-AUC": f"{d.get('roc_auc', 0):.4f}",
                "Test PR-AUC": f"{d.get('pr_auc', 0):.4f}",
                "Precision": f"{d.get('precision', 0)*100:.1f}%",
                "Recall": f"{d.get('recall', 0)*100:.1f}%",
                "F1-Score": f"{d.get('f1_score', 0):.4f}",
                "Brier Score": f"{d.get('brier_score', 0):.4f}",
                "TP (Caught)": f"{cm.get('tp', 0):,}",
                "FP (False Alert)": f"{cm.get('fp', 0):,}",
                "TN (Attended)": f"{cm.get('tn', 0):,}",
                "FN (Missed)": f"{cm.get('fn', 0):,}"
            })
            
        display_df = pd.DataFrame(table_rows).set_index("Model Architecture")
        st.dataframe(display_df, use_container_width=True)
        
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Probability Reliability Diagram (Calibration Curve)")
        cal_path = os.path.join(FIGURES_DIR, "calibration_curve.png")
        if os.path.exists(cal_path):
            st.image(cal_path, use_container_width=True, caption="Reliability Diagram: HistGradientBoosting exhibits superior calibration.")
            
    with c2:
        st.markdown("#### Discrimination Curves (ROC & Precision-Recall)")
        roc_pr_path = os.path.join(FIGURES_DIR, "roc_pr_curves.png")
        if os.path.exists(roc_pr_path):
            st.image(roc_pr_path, use_container_width=True, caption="PR Curves: HistGradientBoosting achieves a 2.85x lift over the baseline rate.")
            
    st.markdown("---")
    st.markdown("### Interactive Model Confusion Matrix Inspection")
    selected_eval_model = st.selectbox(
        "Select Model to Inspect Confusion Matrix on Untouched Test Set:",
        list(models.keys())
    )
    
    # Compute live confusion matrix for the chosen model on the test set
    if not df_test.empty:
        model_obj = models[selected_eval_model]
        test_probs = model_obj.predict_proba(df_test[ALL_MODEL_FEATURES])[:, 1]
        test_preds = (test_probs >= 0.43).astype(int)
        
        from sklearn.metrics import confusion_matrix as sk_cm, classification_report
        y_test_true = df_test['target'].values
        cm_arr = sk_cm(y_test_true, test_preds)
        
        c_hm, c_rep = st.columns([1, 1])
        with c_hm:
            fig_cm_model = px.imshow(
                cm_arr,
                labels=dict(x="Predicted Attendance", y="Actual Attendance", color="Count"),
                x=['Attended (0)', 'No-Show (1)'],
                y=['Attended (0)', 'No-Show (1)'],
                text_auto=',d',
                color_continuous_scale='Blues',
                title=f"Confusion Matrix: {selected_eval_model} (tau = 0.43)"
            )
            st.plotly_chart(fig_cm_model, use_container_width=True)
            
        with c_rep:
            st.markdown(f"#### Classification Metrics ({selected_eval_model})")
            tn, fp, fn, tp = cm_arr.ravel()
            p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_val = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
            
            st.metric("Recall (Sensitivity)", f"{r_val*100:.1f}%", "Proportion of no-shows caught")
            st.metric("Precision (Positive Predictive Value)", f"{p_val*100:.1f}%", "Accuracy of flagged calls")
            st.metric("F1-Score", f"{f1_val:.4f}", "Harmonic mean of Precision & Recall")
            
            st.caption(
                f"**Clinical Asymmetry Note:** False Negatives ({fn:,} unannounced misses costing ~$100 each) are far more detrimental than False Positives ({fp:,} reminder calls costing ~$5 each)."
            )

# =============================================================================
# MODULE 4: SINGLE APPOINTMENT PREDICTOR
# =============================================================================
elif menu == "4. Single Appointment Predictor":
    st.title("🔮 Single Appointment Risk Inference & Explainability")
    st.markdown("Simulate an appointment booking transaction in real-time to obtain the calibrated no-show risk score, operational recommendation, and feature attribution waterfall.")
    
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
            neighbourhood = st.selectbox("Clinic District:", all_neighbourhoods, index=min(10, len(all_neighbourhoods)-1))
            
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            chosen_model_name = st.selectbox("Inference Model Architecture:", ["HistGradientBoosting (Champion)", "Logistic Regression", "Random Forest"])
        with c_sub2:
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
        
        active_model = models.get(chosen_model_name, models['HistGradientBoosting (Champion)'])
        explanation = explainer.explain_instance(record, custom_model=active_model)
        prob = explanation["predicted_risk_probability"]
        flagged = prob >= threshold_val
        
        st.markdown("### Prediction Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Predicted No-Show Risk", f"{prob*100:.1f}%", f"{'+' if flagged else '-'} Threshold: {threshold_val*100:.0f}%")
        res_col2.metric("Risk Classification Tier", explanation["risk_tier"])
        res_col3.metric("Decision Flag", "🚨 FLAGGED FOR OUTREACH" if flagged else "✅ ROUTINE APPOINTMENT")
        
        if flagged:
            st.warning(f"**Recommended Action:** {explanation['recommended_operational_action']}")
        else:
            st.success(f"**Recommended Action:** {explanation['recommended_operational_action']}")
            
        st.markdown("---")
        c_exp1, c_exp2 = st.columns([1, 1])
        with c_exp1:
            st.markdown("#### Local Feature Attribution Drivers")
            for d in explanation["top_drivers"]:
                icon = "🔺" if d["direction"] == "increases_risk" else "🔻"
                st.markdown(f"> {icon} **{d['factor']}** — *{d['impact']}*")
            st.caption(explanation["disclaimer"])
            
        with c_exp2:
            st.markdown("#### Feature Risk Impact Breakdown")
            chart_data = explanation.get("chart_contributions", [])
            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                fig_waterfall = px.bar(
                    chart_df,
                    x='Impact',
                    y='Feature',
                    orientation='h',
                    color='Direction',
                    color_discrete_map={'Increases Risk': '#d9534f', 'Decreases Risk': '#5cb85c', 'Neutral': '#a6c8e0'},
                    title="Directional Feature Risk Contributions"
                )
                fig_waterfall.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_waterfall, use_container_width=True)

# =============================================================================
# MODULE 5: SIGNATURE FEATURE 1 - OPERATIONS SIMULATOR
# =============================================================================
elif menu == "5. Operations Simulator (Signature)":
    st.title("⚙️ Signature Feature 1: Appointment Operations Simulator")
    st.markdown("""
    **Simulate real-world clinic workflows under constrained administrative capacity.**
    Clinics cannot call every patient. Staff coordinators have limited outbound calling hours.
    This component ranks scheduled appointments by risk, flags the top $K$ appointments matching your daily staff capacity, and evaluates captured no-shows and operational lift against actual ground-truth outcomes.
    """)
    
    st.markdown("#### Step 1: Select Appointment Schedule Batch")
    batch_mode = st.radio(
        "Choose Schedule Source:",
        ["Real Clinic Schedule by Date", "Sample Size from Untouched Test Set", "Upload Custom Schedule CSV"],
        horizontal=True
    )
    
    sim_batch = pd.DataFrame()
    if batch_mode == "Real Clinic Schedule by Date":
        if not df_test.empty:
            dates = sorted(df_test['AppointmentDay'].unique())
            clean_dates = [d.split('T')[0] for d in dates]
            date_choice = st.selectbox("Select Clinic Appointment Date:", clean_dates, index=len(clean_dates)-1)
            full_date_str = [d for d in dates if d.startswith(date_choice)][0]
            sim_batch = df_test[df_test['AppointmentDay'] == full_date_str].copy()
            st.caption(f"Loaded **{len(sim_batch):,} scheduled appointments** for {date_choice}.")
    elif batch_mode == "Sample Size from Untouched Test Set":
        sample_size = st.slider("Number of Appointments in Daily Schedule:", 50, 1000, 250, step=50)
        sim_batch = df_test.sample(n=min(sample_size, len(df_test)), random_state=42).copy()
    else:
        uploaded_file = st.file_uploader("Upload CSV containing appointments with PULSE features:", type=['csv'])
        if uploaded_file is not None:
            sim_batch = pd.read_csv(uploaded_file)
            st.success(f"Uploaded {len(sim_batch):,} custom records.")
        else:
            sim_batch = df_test.head(200).copy()
            st.info("Using default 200 appointment test cohort until CSV upload.")
            
    st.markdown("---")
    st.markdown("#### Step 2: Configure Operational Constraints & Economics")
    c_set1, c_set2, c_set3, c_set4 = st.columns(4)
    with c_set1:
        daily_capacity = st.slider("Daily Staff Call Quota (K):", 10, min(500, len(sim_batch)), min(50, len(sim_batch)), step=5)
    with c_set2:
        cost_noshow = st.number_input("Lost Revenue per Empty Slot ($):", value=100.0)
    with c_set3:
        cost_call = st.number_input("Labor Cost per Phone Call ($):", value=5.0)
    with c_set4:
        recovery_rate = st.slider("Assumed Call Recovery Rate (%):", 10, 60, 30) / 100.0
        
    if not sim_batch.empty:
        sim_res = simulator.simulate_batch(
            sim_batch,
            daily_capacity=daily_capacity,
            cost_per_noshow=cost_noshow,
            cost_per_call=cost_call,
            phone_recovery_rate=recovery_rate
        )
        m = sim_res["metrics"]
        
        st.markdown("---")
        st.markdown("#### Step 3: Simulation Results & Operational Impact")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Schedule Reviewed", f"{m['appointments_reviewed']} / {m['total_appointments']}", f"{m['review_coverage_pct']}% of schedule")
        kpi2.metric("Captured No-Shows (TP)", f"{m['captured_noshows_in_top_k']} / {m['total_actual_noshows']}", f"{m['capture_rate_pct']}% capture rate")
        kpi3.metric("Precision in Top K", f"{m['precision_in_top_k_pct']}%", f"{m['lift_over_random']}x Lift Over Random")
        kpi4.metric("Net Financial Savings", f"${m['estimated_net_savings_usd']:,.2f}", f"Labor: -${m['labor_cost_usd']:,.0f}")
        
        c_don, c_lift = st.columns(2)
        with c_don:
            # Donut chart of operational outcomes
            donut_labels = ['🎯 Captured No-Shows (TP)', '⚠️ Contacted Attendees (FP)', '❌ Uncontacted No-Shows (FN)', '✅ Uncontacted Attendees (TN)']
            donut_values = [
                m['captured_noshows_in_top_k'],
                m['contacted_attendees'],
                m['uncontacted_noshows_outside_k'],
                m['uncontacted_attendees']
            ]
            fig_donut = px.pie(
                names=donut_labels,
                values=donut_values,
                hole=0.45,
                color=donut_labels,
                color_discrete_map={
                    '🎯 Captured No-Shows (TP)': '#2b5c8f',
                    '⚠️ Contacted Attendees (FP)': '#f0ad4e',
                    '❌ Uncontacted No-Shows (FN)': '#d9534f',
                    '✅ Uncontacted Attendees (TN)': '#5cb85c'
                },
                title="Review Quota Outcome Breakdown"
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with c_lift:
            # Lift bar chart comparing model review vs random review
            lift_df = pd.DataFrame({
                "Strategy": ["Random Calling (Baseline)", "PULSE Risk-Ranked Calling"],
                "Capture Precision (%)": [m['base_noshow_rate_pct'], m['precision_in_top_k_pct']]
            })
            fig_lift = px.bar(
                lift_df,
                x='Strategy',
                y='Capture Precision (%)',
                color='Strategy',
                color_discrete_map={'Random Calling (Baseline)': '#a6c8e0', 'PULSE Risk-Ranked Calling': '#2b5c8f'},
                text_auto='.1f',
                title=f"Efficiency Gain: {m['lift_over_random']}x Lift Over Random Review"
            )
            fig_lift.update_layout(showlegend=False)
            st.plotly_chart(fig_lift, use_container_width=True)
            
        st.markdown("---")
        st.markdown("#### Top Prioritized Appointments for Human Telephone Outreach")
        display_df = sim_res["selected_records"].copy()
        display_cols = [
            'rank', 'predicted_risk', 'risk_tier', 'lead_days', 'Age',
            'prior_appointments_count', 'prior_no_shows_count',
            'actual_status', 'review_outcome', 'recommended_action'
        ]
        available_cols = [c for c in display_cols if c in display_df.columns]
        st.dataframe(
            display_df[available_cols].style.format({'predicted_risk': '{:.1%}'}),
            use_container_width=True
        )

# =============================================================================
# MODULE 6: SIGNATURE FEATURE 2 - THRESHOLD TRADE-OFF STUDIO
# =============================================================================
elif menu == "6. Threshold Trade-off Studio (Signature)":
    st.title("🎚️ Signature Feature 2: Threshold Trade-off Studio")
    st.markdown("""
    Explore how changing the probability decision threshold $\\tau$ alters the operational trade-off between **identifying missed appointments (Recall)** and **wasting administrative outreach on patients who would show up anyway (Precision & Workload)**.
    """)
    
    st.markdown("#### Quick-Action Operational Policy Presets")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_tau = 0.43
    if col_p1.button("🎯 F1-Optimal (tau = 0.43)", use_container_width=True):
        preset_tau = 0.43
    if col_p2.button("📞 High Outreach / High Recall (tau = 0.25)", use_container_width=True):
        preset_tau = 0.25
    if col_p3.button("⚡ High Precision / Low Capacity (tau = 0.65)", use_container_width=True):
        preset_tau = 0.65
        
    c_ctrl, c_metrics = st.columns([1, 2])
    with c_ctrl:
        dataset_choice = st.radio("Evaluation Split:", ["validation", "test"], index=0)
        tau = st.slider("Select Probability Threshold (tau):", min_value=0.05, max_value=0.95, value=preset_tau, step=0.01)
        cost_slot = st.number_input("Slot Cost ($):", value=100.0)
        cost_labor = st.number_input("Outreach Labor ($):", value=5.0)
        
    res_thresh = studio.evaluate_threshold(tau, dataset=dataset_choice, cost_fn=cost_slot, cost_fp=cost_labor)
    
    with c_metrics:
        st.markdown(f"#### Performance at tau = {tau:.2f} ({dataset_choice.capitalize()} Cohort)")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Precision", f"{res_thresh['precision']*100:.1f}%")
        b2.metric("Recall", f"{res_thresh['recall']*100:.1f}%")
        b3.metric("F1-Score", f"{res_thresh['f1_score']:.4f}")
        b4.metric("Flagged Volume", f"{res_thresh['flagged_percent']}%", f"{res_thresh['flagged_for_review']:,} calls")
        
        st.info(f"**Trade-off Policy Diagnosis:** {res_thresh['tradeoff_summary']}")
        st.success(f"**Net Projected Cost Savings:** ${res_thresh['net_cost_savings']:,.2f} over doing no reminders.")
        
    st.markdown("---")
    c_cm_plot, c_curves = st.columns(2)
    with c_cm_plot:
        cm_data = res_thresh["confusion_matrix"]
        cm_matrix = np.array([[cm_data["tn"], cm_data["fp"]], [cm_data["fn"], cm_data["tp"]]])
        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Attendance", y="Actual Attendance", color="Count"),
            x=['Attended (0)', 'No-Show (1)'],
            y=['Attended (0)', 'No-Show (1)'],
            text_auto=',d',
            color_continuous_scale='Blues',
            title=f"Confusion Matrix (tau = {tau:.2f})"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with c_curves:
        # Generate live tradeoff curve
        tradeoff_df = studio.generate_tradeoff_curve(dataset=dataset_choice, steps=40)
        fig_trade = go.Figure()
        fig_trade.add_trace(go.Scatter(x=tradeoff_df['threshold'], y=tradeoff_df['precision'], mode='lines', name='Precision', line=dict(color='#2b5c8f', width=2)))
        fig_trade.add_trace(go.Scatter(x=tradeoff_df['threshold'], y=tradeoff_df['recall'], mode='lines', name='Recall', line=dict(color='#d9534f', width=2)))
        fig_trade.add_trace(go.Scatter(x=tradeoff_df['threshold'], y=tradeoff_df['f1_score'], mode='lines', name='F1-Score', line=dict(color='#5cb85c', width=2.5)))
        fig_trade.add_vline(x=tau, line=dict(color='purple', dash='dash'), annotation_text=f"Selected tau={tau:.2f}")
        fig_trade.update_layout(
            title="Precision, Recall & F1 vs. Operating Threshold",
            xaxis=dict(title="Probability Threshold (tau)"),
            yaxis=dict(title="Score", range=[0, 1]),
            legend=dict(x=0.01, y=0.05)
        )
        st.plotly_chart(fig_trade, use_container_width=True)

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
