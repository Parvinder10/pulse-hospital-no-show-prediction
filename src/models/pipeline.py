"""
Machine Learning Preprocessing and Pipeline Construction for PULSE.
Builds scikit-learn ColumnTransformer and pipeline architectures.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

NUMERIC_FEATURES = [
    'lead_days',
    'lead_weeks',
    'booking_hour',
    'booking_weekday',
    'booking_month',
    'appointment_weekday',
    'appointment_month',
    'Age',
    'prior_appointments_count',
    'prior_no_shows_count',
    'prior_no_show_rate'
]

BINARY_FEATURES = [
    'is_same_day',
    'gender_female',
    'Scholarship',
    'is_repeat_patient'
]

CATEGORICAL_FEATURES = [
    'age_group',
    'neighbourhood'
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

def build_preprocessor() -> ColumnTransformer:
    """Builds scikit-learn ColumnTransformer for PULSE feature set."""
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    binary_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, NUMERIC_FEATURES),
            ('bin', binary_pipe, BINARY_FEATURES),
            ('cat', categorical_pipe, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def get_model_pipeline(model_type: str = 'hist_gradient_boosting', random_state: int = 42) -> Pipeline:
    """
    Returns an unfitted scikit-learn Pipeline with the specified classifier.
    Supported types: 'dummy', 'logistic', 'random_forest', 'hist_gradient_boosting'.
    """
    preprocessor = build_preprocessor()
    
    if model_type == 'dummy':
        classifier = DummyClassifier(strategy='prior', random_state=random_state)
    elif model_type == 'logistic':
        classifier = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=1.0,
            random_state=random_state
        )
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'hist_gradient_boosting':
        classifier = HistGradientBoostingClassifier(
            class_weight='balanced',
            max_iter=120,
            learning_rate=0.08,
            max_depth=8,
            min_samples_leaf=20,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return pipeline
