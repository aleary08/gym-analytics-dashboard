import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from data import generate_gym_data

def train_churn_model():
    df = generate_gym_data()
    
    features = [
        "membership_months",
        "avg_sessions_per_week", 
        "days_since_last_visit",
        "attendance_drop",
        "failed_payments",
        "monthly_fee"
    ]
    
    X = df[features]
    y = df["churned"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler, features

def predict_churn(df, model, scaler, features):
    X = df[features]
    X_scaled = scaler.transform(X)
    
    churn_proba = model.predict_proba(X_scaled)[:, 1]
    df = df.copy()
    df["churn_probability"] = (churn_proba * 100).round(1)
    
    df["risk_level"] = pd.cut(
        df["churn_probability"],
        bins=[0, 30, 60, 100],
        labels=["Low", "Medium", "High"]
    )
    
    return df

if __name__ == "__main__":
    model, scaler, features = train_churn_model()
    df = generate_gym_data()
    results = predict_churn(df, model, scaler, features)
    print("\nTop 5 At-Risk Members:")
    print(results.nlargest(5, "churn_probability")[
        ["name", "churn_probability", "risk_level", "days_since_last_visit"]
    ])