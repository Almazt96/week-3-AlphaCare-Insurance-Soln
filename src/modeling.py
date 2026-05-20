import numpy as np
# 1. Robust Modular Framework 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, classification_report

def prepare_insurance_features(df: pd.DataFrame, is_regression: bool = True):
    """Processes categorical attributes, handles empty cells, and creates model features."""
    df = df.copy()
    
    # Target Selection
    if is_regression:
        # Severity targets positive claims occurrences
        df = df[df['TotalClaims'] > 0]
        y = df['TotalClaims']
    else:
        # Frequency targets likelihood mapping
        y = np.where(df['TotalClaims'] > 0, 1, 0)
        
    # Isolate functional predictive elements
    feature_cols = [
        'VehicleType', 'RegistrationYear', 'Make', 'Cylinders', 
        'Cubiccapacity', 'Kilowatts', 'CustomValueEstimate', 
        'Gender', 'Province', 'MaritalStatus'
    ]
    # Filter features that exist in the dataset
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    
    # Handle missing numerical indicators using median values
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())
        
    # Handle categorical variables using label encoding
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate_regression(X_train, X_test, y_train, y_test):
    """Trains and compares Linear Regression, Random Forest, and XGBoost models."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    }
    
    performance_metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        performance_metrics[name] = {"RMSE": round(rmse, 2), "R2": round(r2, 4)}
        
    return performance_metrics, models["XGBoost Regressor"]