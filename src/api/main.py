"""
FastAPI Backend Application for PULSE.
Exposes real-time appointment no-show prediction, batch ranking,
operational simulation, and model introspection endpoints.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.pipeline import ALL_MODEL_FEATURES
from src.explainability.explainer import ModelExplainer
from src.api.schemas import (
    AppointmentInput, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, BatchRankedItem, ModelInfoResponse,
    HealthResponse, DriverItem
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json")

# In-memory storage for loaded model and artifacts
state = {
    "model": None,
    "explainer": None,
    "metadata": {},
    "metrics": {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load artifacts
    print("[API] Loading model and serialized artifacts...")
    if os.path.exists(MODEL_PATH):
        state["model"] = joblib.load(MODEL_PATH)
        state["explainer"] = ModelExplainer(model_path=MODEL_PATH)
        print("[API] Model loaded successfully.")
    else:
        print(f"[API WARNING] Model file not found at {MODEL_PATH}")
        
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            state["metadata"] = json.load(f)
            
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            state["metrics"] = json.load(f)
            
    yield
    # Shutdown: Clean up if needed
    state.clear()

app = FastAPI(
    title="PULSE - Hospital Appointment No-Show Prediction API",
    description="Operational decision-support system predicting appointment missed risk to plan human-reviewed outreach and buffer capacity.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for dashboard and frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _enrich_appointment_features(apt: AppointmentInput) -> dict:
    """Computes derived features if not explicitly supplied."""
    data = apt.model_dump()
    
    # Lead time derivations
    if data.get('is_same_day') is None:
        data['is_same_day'] = 1 if data['lead_days'] == 0 else 0
    if data.get('lead_weeks') is None:
        data['lead_weeks'] = round(data['lead_days'] / 7.0, 2)
        
    # Age group derivation
    if data.get('age_group') is None:
        age = data['Age']
        if age <= 12:
            data['age_group'] = 'Child'
        elif age <= 24:
            data['age_group'] = 'Youth'
        elif age <= 45:
            data['age_group'] = 'YoungAdult'
        elif age <= 65:
            data['age_group'] = 'MiddleAged'
        else:
            data['age_group'] = 'Senior'
            
    # History derivations
    if data.get('is_repeat_patient') is None:
        data['is_repeat_patient'] = 1 if data['prior_appointments_count'] > 0 else 0
    if data.get('prior_no_show_rate') is None:
        if data['prior_appointments_count'] > 0:
            data['prior_no_show_rate'] = round(data['prior_no_shows_count'] / data['prior_appointments_count'], 4)
        else:
            data['prior_no_show_rate'] = 0.0
            
    return data

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Service health and model readiness probe."""
    return HealthResponse(
        status="healthy" if state["model"] is not None else "degraded",
        service="PULSE Prediction Service",
        model_loaded=state["model"] is not None,
        version="1.0.0"
    )

@app.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    """Returns model architecture, trained timestamp, features, and default operational thresholds."""
    if not state["metadata"]:
        raise HTTPException(status_code=503, detail="Model metadata is not loaded.")
        
    meta = state["metadata"]
    default_tau = 0.43
    if state["metrics"] and "chosen_threshold" in state["metrics"]:
        default_tau = float(state["metrics"]["chosen_threshold"])
        
    return ModelInfoResponse(
        project=meta.get("project", "PULSE"),
        model_architecture=meta.get("champion_model", "HistGradientBoosting"),
        trained_timestamp=meta.get("training_timestamp", "2026-11-15T00:00:00Z"),
        training_sample_size=meta.get("train_samples", 77364),
        validation_sample_size=meta.get("val_samples", 16578),
        feature_count=meta.get("feature_count", len(ALL_MODEL_FEATURES)),
        features=ALL_MODEL_FEATURES,
        default_operational_threshold=default_tau
    )

@app.get("/metrics")
def get_evaluation_metrics():
    """Returns validation and test evaluation metrics, ROC-AUC, PR-AUC, and threshold trade-offs."""
    if not state["metrics"]:
        raise HTTPException(status_code=503, detail="Evaluation metrics are not loaded.")
    return state["metrics"]

@app.post("/predict", response_model=PredictionResponse)
def predict_appointment(appointment: AppointmentInput, threshold: float = Query(default=0.43, ge=0.05, le=0.95)):
    """
    Predicts no-show risk probability for an individual appointment at booking time.
    Returns calibrated probability, risk tier, action recommendation, and local feature drivers.
    """
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not ready for inference.")
        
    enriched = _enrich_appointment_features(appointment)
    input_df = pd.DataFrame([enriched])[ALL_MODEL_FEATURES]
    
    # Predict real probability
    prob = float(state["model"].predict_proba(input_df)[:, 1][0])
    
    # Explain local instance
    explanation = state["explainer"].explain_instance(enriched)
    
    flagged = prob >= threshold
    risk_tier = "High Risk" if prob >= 0.40 else ("Moderate Risk" if prob >= 0.20 else "Low Risk")
    
    driver_objs = [DriverItem(**d) for d in explanation["top_drivers"]]
    
    return PredictionResponse(
        predicted_risk_probability=round(prob, 4),
        risk_tier=risk_tier,
        flagged_for_outreach=bool(flagged),
        threshold_applied=round(threshold, 3),
        recommended_action=explanation["recommended_operational_action"],
        top_drivers=driver_objs,
        model_version="1.0.0",
        disclaimer="Operational decision support only. Strictly non-clinical; never used to cancel appointments or deny medical care."
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction and operational scheduling simulator endpoint.
    Ranks appointments by risk and selects top K up to daily staff review capacity.
    """
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not ready for inference.")
        
    if not request.appointments:
        raise HTTPException(status_code=400, detail="Appointments batch cannot be empty.")
        
    enriched_rows = [_enrich_appointment_features(a) for a in request.appointments]
    df = pd.DataFrame(enriched_rows)
    
    # Real model scoring
    probs = state["model"].predict_proba(df[ALL_MODEL_FEATURES])[:, 1]
    df['predicted_risk'] = probs
    
    # Rank descending
    df_sorted = df.sort_values(by='predicted_risk', ascending=False).reset_index(drop=True)
    
    k = request.daily_capacity
    tau = request.decision_threshold
    
    ranked_items = []
    for idx, row in df_sorted.iterrows():
        rank = idx + 1
        p = float(row['predicted_risk'])
        selected = bool(rank <= k and p >= tau)
        
        tier = "High Risk" if p >= 0.40 else ("Moderate Risk" if p >= 0.20 else "Low Risk")
        action = (
            "Priority Outreach: Personal Phone Call" if selected
            else ("Automated SMS/Email" if p >= 0.20 else "Routine Calendar Reminder")
        )
        
        ranked_items.append(BatchRankedItem(
            rank=rank,
            AppointmentID=int(row['AppointmentID']) if pd.notnull(row.get('AppointmentID')) else None,
            predicted_risk=round(p, 4),
            risk_tier=tier,
            selected_for_review=selected,
            recommended_action=action
        ))
        
    flagged_count = sum(1 for item in ranked_items if item.selected_for_review)
    coverage = round(flagged_count / len(ranked_items) * 100, 2)
    avg_risk = round(float(np.mean(probs)), 4)
    
    return BatchPredictionResponse(
        total_appointments=len(ranked_items),
        daily_capacity=k,
        appointments_flagged_for_review=flagged_count,
        review_coverage_pct=coverage,
        average_batch_risk=avg_risk,
        ranked_appointments=ranked_items
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
