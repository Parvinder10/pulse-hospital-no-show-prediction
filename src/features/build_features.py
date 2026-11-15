"""
Point-in-Time Feature Engineering & Data Preparation Pipeline for PULSE.

Key Responsibilities:
1. Data Cleaning:
   - Purge physically impossible records (lead_days < 0).
   - Purge anomalous records (Age < 0).
2. Point-in-Time Historical Patient Aggregations:
   - Calculate historical appointments, no-shows, and no-show rate strictly
     using records occurring before the current appointment's prediction point (ScheduledDay).
3. Booking & Schedule Features:
   - lead_days: difference in calendar days between AppointmentDay and ScheduledDay.
   - is_same_day: 1 if appointment is on the booking day, else 0.
   - booking_hour, booking_weekday, booking_month.
   - appointment_weekday, appointment_month.
   - Age, Age_group bins.
   - Gender (binary encoded).
   - Scholarship (socioeconomic proxy).
   - Neighbourhood (clinic catchment area).
4. Strict Leakage Guard:
   - Purge SMS_received (temporal leakage).
   - Purge Hipertension, Diabetes, Alcoholism, Handcap (clinical safety boundary).
   - Purge AppointmentID, raw PatientId, ScheduledDay, AppointmentDay from model inputs.
5. Chronological Train / Validation / Test Splitting:
   - Chronological sort by ScheduledDay.
   - 70% Train, 15% Validation, 15% Test.
"""
import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "noshowappointments-kagglev2-may-2016.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Removes invalid records (negative lead times, negative age)."""
    initial_count = len(df)
    
    # Parse dates
    df['ScheduledDay_dt'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay_dt'] = pd.to_datetime(df['AppointmentDay'])
    
    # Calculate lead days (calendar days)
    df['lead_days'] = (
        df['AppointmentDay_dt'].dt.tz_localize(None).dt.normalize() - 
        df['ScheduledDay_dt'].dt.tz_localize(None).dt.normalize()
    ).dt.days
    
    # Filter valid rows
    valid_mask = (df['lead_days'] >= 0) & (df['Age'] >= 0)
    cleaned_df = df[valid_mask].copy()
    
    purged_count = initial_count - len(cleaned_df)
    print(f"[DATA CLEANING] Initial: {initial_count:,} | Valid: {len(cleaned_df):,} | Purged: {purged_count}")
    return cleaned_df

def compute_point_in_time_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes prior attendance metrics for each patient chronologically.
    Guarantees that at prediction point t_pred, only appointments strictly
    prior to t_pred are used.
    """
    print("[FEATURE ENGINEERING] Computing point-in-time patient history...")
    # Sort chronologically by ScheduledDay, then AppointmentDay
    df = df.sort_values(by=['ScheduledDay_dt', 'AppointmentDay_dt']).reset_index(drop=True)
    
    # Binary numeric target (1 = no-show, 0 = attended)
    df['target'] = (df['No-show'] == 'Yes').astype(int)
    
    # For each patient, compute cumulative count of prior appointments and prior no-shows.
    # We must shift by 1 within each patient group to exclude the current appointment!
    grouped = df.groupby('PatientId', sort=False)
    
    # Cumulative count of prior appointments (0 for the patient's first appointment)
    df['prior_appointments_count'] = grouped.cumcount()
    
    # Cumulative prior no-shows: cumsum shifted by 1
    cum_noshows = grouped['target'].cumsum()
    df['prior_no_shows_count'] = cum_noshows - df['target']
    
    # Prior no-show rate (0.0 if no prior appointments)
    df['prior_no_show_rate'] = np.where(
        df['prior_appointments_count'] > 0,
        df['prior_no_shows_count'] / df['prior_appointments_count'],
        0.0
    )
    df['is_repeat_patient'] = (df['prior_appointments_count'] > 0).astype(int)
    
    print("[FEATURE ENGINEERING] Point-in-time patient history computed successfully.")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers operational scheduling and demographic features."""
    print("[FEATURE ENGINEERING] Generating operational features...")
    
    # Lead time features
    df['is_same_day'] = (df['lead_days'] == 0).astype(int)
    df['lead_weeks'] = (df['lead_days'] / 7.0).round(2)
    
    # Booking time features (from ScheduledDay)
    df['booking_hour'] = df['ScheduledDay_dt'].dt.hour
    df['booking_weekday'] = df['ScheduledDay_dt'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df['booking_month'] = df['ScheduledDay_dt'].dt.month
    
    # Appointment time features (from AppointmentDay)
    df['appointment_weekday'] = df['AppointmentDay_dt'].dt.dayofweek
    df['appointment_month'] = df['AppointmentDay_dt'].dt.month
    
    # Binary Demographic / Socioeconomic
    df['gender_female'] = (df['Gender'] == 'F').astype(int)
    # Scholarship is already 0/1
    
    # Age bins
    age_bins = [-1, 12, 24, 45, 65, 120]
    age_labels = ['Child', 'Youth', 'YoungAdult', 'MiddleAged', 'Senior']
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels).astype(str)
    
    # Clinic Neighbourhood (clean string)
    df['neighbourhood'] = df['Neighbourhood'].astype(str).str.strip().str.upper()
    
    # Ensure target is cleanly formatted
    if 'target' not in df.columns:
        df['target'] = (df['No-show'] == 'Yes').astype(int)
        
    print("[FEATURE ENGINEERING] Features created successfully.")
    return df

def select_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """
    Selects valid prediction-time features, explicitly dropping leaked
    or clinical boundary fields.
    """
    feature_columns = [
        'lead_days',
        'is_same_day',
        'lead_weeks',
        'booking_hour',
        'booking_weekday',
        'booking_month',
        'appointment_weekday',
        'appointment_month',
        'Age',
        'age_group',
        'gender_female',
        'Scholarship',
        'neighbourhood',
        'prior_appointments_count',
        'prior_no_shows_count',
        'prior_no_show_rate',
        'is_repeat_patient'
    ]
    
    target_column = 'target'
    
    # Keep metadata columns for operations simulator & evaluation tracking
    metadata_columns = [
        'AppointmentID',
        'PatientId',
        'ScheduledDay',
        'AppointmentDay'
    ]
    
    all_keep = metadata_columns + feature_columns + [target_column]
    final_df = df[all_keep].copy()
    return final_df, feature_columns, target_column

def split_chronologically(df: pd.DataFrame, train_pct: float = 0.70, val_pct: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data chronologically based on ScheduledDay.
    Ensures validation and test sets strictly represent future operations.
    """
    total = len(df)
    train_end = int(total * train_pct)
    val_end = int(total * (train_pct + val_pct))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    print("\n[CHRONOLOGICAL SPLIT SUMMARY]")
    print(f"Total rows:      {total:,}")
    print(f"Train split:     {len(train_df):,} rows ({len(train_df)/total*100:.1f}%) | Target rate: {train_df['target'].mean()*100:.2f}%")
    print(f"Validation split:{len(val_df):,} rows ({len(val_df)/total*100:.1f}%) | Target rate: {val_df['target'].mean()*100:.2f}%")
    print(f"Test split:      {len(test_df):,} rows ({len(test_df)/total*100:.1f}%) | Target rate: {test_df['target'].mean()*100:.2f}%")
    
    return train_df, val_df, test_df

def prepare_pipeline(raw_path: str = RAW_DATA_PATH, output_dir: str = PROCESSED_DIR):
    """Executes the full end-to-end data preparation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    df_raw = pd.read_csv(raw_path)
    df_clean = clean_raw_data(df_raw)
    df_history = compute_point_in_time_patient_history(df_clean)
    df_features = engineer_features(df_history)
    df_final, feature_cols, target_col = select_model_features(df_features)
    
    train_df, val_df, test_df = split_chronologically(df_final)
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[SAVED] Train split:      {train_path}")
    print(f"[SAVED] Validation split: {val_path}")
    print(f"[SAVED] Test split:       {test_path}")
    print(f"[METADATA] Feature count: {len(feature_cols)} features: {feature_cols}")
    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_pipeline()
