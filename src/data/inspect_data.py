"""
Dataset Inspection and Data Quality Audit for PULSE.
Analyzes schema, missing values, anomalies, target distributions,
lead times, and potential leakage points.
"""
import os
import pandas as pd
import numpy as np

def run_inspection(data_path: str = None) -> dict:
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "noshowappointments-kagglev2-may-2016.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw dataset not found at {data_path}. Run download.py first.")

    print(f"Loading raw data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # 1. Dimensions and columns
    total_records = len(df)
    cols = list(df.columns)
    
    # 2. Target Distribution
    target_counts = df['No-show'].value_counts().to_dict()
    target_pcts = (df['No-show'].value_counts(normalize=True) * 100).round(2).to_dict()
    
    # 3. Timestamps & Lead Time
    df['ScheduledDay_dt'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay_dt'] = pd.to_datetime(df['AppointmentDay'])
    
    # Lead time in calendar days
    lead_days = (df['AppointmentDay_dt'].dt.tz_localize(None).dt.normalize() - 
                 df['ScheduledDay_dt'].dt.tz_localize(None).dt.normalize()).dt.days
    
    negative_lead_count = int((lead_days < 0).sum())
    same_day_count = int((lead_days == 0).sum())
    same_day_pct = round(same_day_count / total_records * 100, 2)
    
    # 4. Patient & Appointment uniqueness
    unique_patients = int(df['PatientId'].nunique())
    unique_appointments = int(df['AppointmentID'].nunique())
    repeat_patients = int((df['PatientId'].value_counts() > 1).sum())
    
    # 5. Anomalies
    negative_age_count = int((df['Age'] < 0).sum())
    age_min = int(df['Age'].min())
    age_max = int(df['Age'].max())
    
    # 6. No-show rate by lead time condition
    df['target_numeric'] = (df['No-show'] == 'Yes').astype(int)
    same_day_noshow_rate = round(float(df[lead_days == 0]['target_numeric'].mean() * 100), 2)
    advance_noshow_rate = round(float(df[lead_days > 0]['target_numeric'].mean() * 100), 2)
    
    # 7. SMS breakdown
    advance_sms0_rate = round(float(df[(lead_days > 0) & (df['SMS_received'] == 0)]['target_numeric'].mean() * 100), 2)
    advance_sms1_rate = round(float(df[(lead_days > 0) & (df['SMS_received'] == 1)]['target_numeric'].mean() * 100), 2)
    
    results = {
        "total_records": total_records,
        "columns": cols,
        "target_counts": target_counts,
        "target_percentages": target_pcts,
        "unique_patients": unique_patients,
        "unique_appointments": unique_appointments,
        "repeat_patients": repeat_patients,
        "scheduled_min": str(df['ScheduledDay_dt'].min()),
        "scheduled_max": str(df['ScheduledDay_dt'].max()),
        "appointment_min": str(df['AppointmentDay_dt'].min()),
        "appointment_max": str(df['AppointmentDay_dt'].max()),
        "negative_lead_count": negative_lead_count,
        "same_day_count": same_day_count,
        "same_day_pct": same_day_pct,
        "negative_age_count": negative_age_count,
        "age_min": age_min,
        "age_max": age_max,
        "same_day_noshow_rate": same_day_noshow_rate,
        "advance_noshow_rate": advance_noshow_rate,
        "advance_sms0_rate": advance_sms0_rate,
        "advance_sms1_rate": advance_sms1_rate
    }
    
    print("\n=======================================================")
    print("       PULSE DATA AUDIT & INSPECTION REPORT           ")
    print("=======================================================")
    print(f"Total Appointments:         {total_records:,}")
    print(f"Unique Patients:            {unique_patients:,}")
    print(f"Patients with >1 Visit:     {repeat_patients:,} ({repeat_patients/unique_patients*100:.1f}%)")
    print(f"Target Breakdown (Raw):     {target_counts} ({target_pcts}%)")
    print(f"Scheduling Window:          {results['scheduled_min']} to {results['scheduled_max']}")
    print(f"Appointment Window:         {results['appointment_min']} to {results['appointment_max']}")
    print("-------------------------------------------------------")
    print("DATA INTEGRITY ANOMALIES:")
    print(f"  - Negative Lead Days (<0): {negative_lead_count} records (Must be purged)")
    print(f"  - Negative Age (<0):       {negative_age_count} records (Must be purged)")
    print(f"  - Age Range:               [{age_min}, {age_max}]")
    print("-------------------------------------------------------")
    print("LEAD TIME & BEHAVIORAL FINDINGS:")
    print(f"  - Same-Day Appointments:   {same_day_count:,} ({same_day_pct}%)")
    print(f"  - Same-Day No-Show Rate:   {same_day_noshow_rate}%")
    print(f"  - Advance No-Show Rate:    {advance_noshow_rate}%")
    print(f"  - Advance SMS=0 No-Show:   {advance_sms0_rate}%")
    print(f"  - Advance SMS=1 No-Show:   {advance_sms1_rate}%")
    print("=======================================================\n")
    return results

if __name__ == "__main__":
    run_inspection()
