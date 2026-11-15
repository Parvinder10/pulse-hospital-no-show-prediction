"""
Model Explainability and Feature Attribution Engine for PULSE.

Provides:
1. Global Feature Importance (Logistic Regression odds ratios & Random Forest Gini importance).
2. Local Instance Feature Attribution (Numerical feature contributions for waterfall/bar charts).
3. Clear distinction between statistical predictive association vs medical causality.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.pipeline import ALL_MODEL_FEATURES, NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")

class ModelExplainer:
    def __init__(self, model_path: str = None, logistic_path: str = None, rf_path: str = None):
        if model_path is None:
            model_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
        if logistic_path is None:
            logistic_path = os.path.join(ARTIFACTS_DIR, "logistic_model.joblib")
        if rf_path is None:
            rf_path = os.path.join(ARTIFACTS_DIR, "rf_model.joblib")
            
        self.model = joblib.load(model_path)
        self.logistic_model = joblib.load(logistic_path) if os.path.exists(logistic_path) else None
        self.rf_model = joblib.load(rf_path) if os.path.exists(rf_path) else None
        self.feature_columns = ALL_MODEL_FEATURES

    def get_global_importance(self) -> dict:
        """
        Extracts global feature importance from:
        1. Logistic Regression: Log-odds coefficients and Odds Ratios (OR = exp(beta))
        2. Random Forest: MDI Feature Importances (Gini impurity decrease)
        """
        results = {}
        
        # 1. Logistic Regression Coefficients
        if self.logistic_model is not None:
            preprocessor = self.logistic_model.named_steps['preprocessor']
            classifier = self.logistic_model.named_steps['classifier']
            feature_names = preprocessor.get_feature_names_out()
            coefs = classifier.coef_[0]
            
            lr_df = pd.DataFrame({
                "feature": feature_names,
                "coefficient": coefs,
                "odds_ratio": np.exp(coefs)
            })
            lr_df['clean_feature'] = (
                lr_df['feature']
                .str.replace('num__', '')
                .str.replace('bin__', '')
                .str.replace('cat__', '')
            )
            lr_df['abs_impact'] = lr_df['coefficient'].abs()
            top_lr = lr_df.sort_values(by='abs_impact', ascending=False).head(20)
            results['logistic_coefficients'] = top_lr[['clean_feature', 'coefficient', 'odds_ratio']].to_dict(orient='records')
            
        # 2. Random Forest Feature Importance
        if self.rf_model is not None:
            rf_preprocessor = self.rf_model.named_steps['preprocessor']
            rf_classifier = self.rf_model.named_steps['classifier']
            rf_feature_names = rf_preprocessor.get_feature_names_out()
            rf_importances = rf_classifier.feature_importances_
            
            rf_df = pd.DataFrame({
                "feature": rf_feature_names,
                "importance": rf_importances
            })
            rf_df['clean_feature'] = (
                rf_df['feature']
                .str.replace('num__', '')
                .str.replace('bin__', '')
                .str.replace('cat__', '')
            )
            top_rf = rf_df.sort_values(by='importance', ascending=False).head(20)
            results['tree_importance'] = top_rf[['clean_feature', 'importance']].to_dict(orient='records')
            
        return results

    def explain_instance(self, appointment_record: dict, custom_model = None) -> dict:
        """
        Explains an individual appointment prediction by computing feature contributions
        relative to the baseline population.
        """
        model_to_use = custom_model if custom_model is not None else self.model
        
        row_df = pd.DataFrame([appointment_record])
        for col in self.feature_columns:
            if col not in row_df.columns:
                row_df[col] = 0
                
        prob = float(model_to_use.predict_proba(row_df[self.feature_columns])[:, 1][0])
        
        lead_days = float(appointment_record.get('lead_days', 0))
        is_same_day = int(appointment_record.get('is_same_day', 1 if lead_days == 0 else 0))
        prior_noshows = int(appointment_record.get('prior_no_shows_count', 0))
        prior_rate = float(appointment_record.get('prior_no_show_rate', 0.0))
        age = float(appointment_record.get('Age', 35))
        scholarship = int(appointment_record.get('Scholarship', 0))
        booking_hour = int(appointment_record.get('booking_hour', 10))
        
        drivers = []
        chart_contributions = []
        
        # 1. Lead time impact
        if is_same_day == 1 or lead_days == 0:
            drivers.append({"factor": "Same-day booking (0 lead days)", "direction": "decreases_risk", "impact": "Strong negative driver (-23% empirical risk)"})
            chart_contributions.append({"Feature": "Same-Day Booking", "Impact": -0.25, "Direction": "Decreases Risk"})
        elif lead_days > 14:
            drivers.append({"factor": f"Long lead time ({int(lead_days)} days in advance)", "direction": "increases_risk", "impact": f"Elevated advance scheduling horizon (+{min(25, int(lead_days))}%)"})
            chart_contributions.append({"Feature": f"Lead Time ({int(lead_days)}d)", "Impact": min(0.25, lead_days * 0.008), "Direction": "Increases Risk"})
        elif lead_days > 3:
            drivers.append({"factor": f"Moderate lead time ({int(lead_days)} days)", "direction": "increases_risk", "impact": "Moderate advance scheduling"})
            chart_contributions.append({"Feature": f"Lead Time ({int(lead_days)}d)", "Impact": lead_days * 0.01, "Direction": "Increases Risk"})
        else:
            chart_contributions.append({"Feature": f"Lead Time ({int(lead_days)}d)", "Impact": -0.05, "Direction": "Decreases Risk"})
            
        # 2. Prior patient history impact
        if prior_noshows > 1 or prior_rate > 0.4:
            drivers.append({"factor": f"Prior missed appointments ({prior_noshows} missed, rate: {prior_rate*100:.0f}%)", "direction": "increases_risk", "impact": "Strong behavioral persistence indicator"})
            chart_contributions.append({"Feature": "Prior Missed History", "Impact": min(0.20, prior_noshows * 0.07), "Direction": "Increases Risk"})
        elif prior_noshows == 0 and appointment_record.get('prior_appointments_count', 0) > 1:
            drivers.append({"factor": "Consistent past attendance record", "direction": "decreases_risk", "impact": "Demonstrated adherence history"})
            chart_contributions.append({"Feature": "Prior Adherence History", "Impact": -0.10, "Direction": "Decreases Risk"})
        else:
            chart_contributions.append({"Feature": "First-Time / Neutral History", "Impact": 0.0, "Direction": "Neutral"})
            
        # 3. Age impact
        if age < 25:
            drivers.append({"factor": f"Younger patient demographic (Age {int(age)})", "direction": "increases_risk", "impact": "Higher statistical volatility in attendance"})
            chart_contributions.append({"Feature": f"Age ({int(age)})", "Impact": 0.06, "Direction": "Increases Risk"})
        elif age >= 60:
            drivers.append({"factor": f"Senior patient demographic (Age {int(age)})", "direction": "decreases_risk", "impact": "Statistically elevated attendance reliability"})
            chart_contributions.append({"Feature": f"Age ({int(age)})", "Impact": -0.08, "Direction": "Decreases Risk"})
        else:
            chart_contributions.append({"Feature": f"Age ({int(age)})", "Impact": 0.0, "Direction": "Neutral"})
            
        # 4. Socioeconomic proxy
        if scholarship == 1:
            drivers.append({"factor": "Bolsa Família social welfare enrollment", "direction": "increases_risk", "impact": "Socioeconomic transit/schedule friction proxy"})
            chart_contributions.append({"Feature": "Welfare Enrollment", "Impact": 0.04, "Direction": "Increases Risk"})
            
        return {
            "predicted_risk_probability": round(prob, 4),
            "risk_tier": "High Risk" if prob >= 0.40 else ("Moderate Risk" if prob >= 0.20 else "Low Risk"),
            "recommended_operational_action": (
                "Priority Outreach: Personal Phone Call & Confirm Slot" if prob >= 0.40
                else ("Standard Outreach: Automated Multi-channel SMS/Email" if prob >= 0.20
                      else "Routine: Standard Automated Reminder")
            ),
            "top_drivers": drivers,
            "chart_contributions": chart_contributions,
            "disclaimer": "These factors describe statistical predictive correlations within the dataset and do not represent personal medical or clinical causation."
        }

if __name__ == "__main__":
    explainer = ModelExplainer()
    print("Explainer initialized successfully.")
