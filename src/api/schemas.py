"""
Pydantic Request & Response Validation Schemas for PULSE FastAPI Backend.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

class AppointmentInput(BaseModel):
    AppointmentID: Optional[int] = Field(default=None, description="Optional appointment ID")
    PatientId: Optional[float] = Field(default=None, description="Optional patient ID")
    lead_days: int = Field(..., ge=0, le=365, description="Lead time in calendar days between booking and appointment")
    is_same_day: Optional[int] = Field(default=None, ge=0, le=1, description="1 if appointment is same day as booking, else 0")
    lead_weeks: Optional[float] = Field(default=None, ge=0.0, description="Lead time in weeks")
    booking_hour: int = Field(default=10, ge=0, le=23, description="Hour of booking (0-23)")
    booking_weekday: int = Field(default=0, ge=0, le=6, description="Day of week booking occurred (0=Mon, 6=Sun)")
    booking_month: int = Field(default=5, ge=1, le=12, description="Month booking occurred (1-12)")
    appointment_weekday: int = Field(default=2, ge=0, le=6, description="Day of week appointment occurs (0=Mon, 6=Sun)")
    appointment_month: int = Field(default=5, ge=1, le=12, description="Month appointment occurs (1-12)")
    Age: int = Field(..., ge=0, le=120, description="Patient age in years (must be >= 0)")
    age_group: Optional[str] = Field(default=None, description="Age group: Child, Youth, YoungAdult, MiddleAged, Senior")
    gender_female: int = Field(default=1, ge=0, le=1, description="1 if Female, 0 if Male")
    Scholarship: int = Field(default=0, ge=0, le=1, description="1 if enrolled in Bolsa Familia welfare, else 0")
    neighbourhood: str = Field(default="JARDIM DA PENHA", description="Clinic location / district")
    prior_appointments_count: int = Field(default=0, ge=0, description="Number of prior appointments booked by this patient")
    prior_no_shows_count: int = Field(default=0, ge=0, description="Number of prior appointments missed by this patient")
    prior_no_show_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Prior no-show rate (0.0 to 1.0)")
    is_repeat_patient: Optional[int] = Field(default=None, ge=0, le=1, description="1 if repeat patient, else 0")

    @field_validator('lead_days')
    @classmethod
    def validate_lead_days(cls, v):
        if v < 0:
            raise ValueError("lead_days cannot be negative. Appointment cannot occur before booking.")
        return v

class DriverItem(BaseModel):
    factor: str
    direction: str
    impact: str

class PredictionResponse(BaseModel):
    predicted_risk_probability: float
    risk_tier: str
    flagged_for_outreach: bool
    threshold_applied: float
    recommended_action: str
    top_drivers: List[DriverItem]
    model_version: str
    disclaimer: str

class BatchPredictionRequest(BaseModel):
    appointments: List[AppointmentInput]
    daily_capacity: Optional[int] = Field(default=50, ge=1, description="Max outbound staff call capacity")
    decision_threshold: Optional[float] = Field(default=0.43, ge=0.05, le=0.95, description="Operational decision threshold")

class BatchRankedItem(BaseModel):
    rank: int
    AppointmentID: Optional[int]
    predicted_risk: float
    risk_tier: str
    selected_for_review: bool
    recommended_action: str

class BatchPredictionResponse(BaseModel):
    total_appointments: int
    daily_capacity: int
    appointments_flagged_for_review: int
    review_coverage_pct: float
    average_batch_risk: float
    ranked_appointments: List[BatchRankedItem]

class ModelInfoResponse(BaseModel):
    project: str
    model_architecture: str
    trained_timestamp: str
    training_sample_size: int
    validation_sample_size: int
    feature_count: int
    features: List[str]
    default_operational_threshold: float

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    version: str
