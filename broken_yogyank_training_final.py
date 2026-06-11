"""
Yogyank Entitlement Score - Baseline Training Script (Draft v1)
Author: Junior Data Scientist
Notes: Model is performing R2 is at 0.99.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import xgboost as xgb
import joblib


def load_and_prep_data(path="farmer_scoring_sample_yogyank_round1.csv"):
    # Ensure this CSV is in the same directory before running.
    return pd.read_csv(path)


def train_model():
    df = load_and_prep_data()

    
    print("Applying PM Kisan business policy...")
    df.loc[df["pm_kisan_status"] == "No", "target_entitlement_score"] -= 150

    features = [
        "land_area_acres",
        "crop_type",
        "pm_kisan_status",
        "historical_repayment_score",
        "defaulted_in_next_12_months",  
    ]

    X = df[features].copy()
    y = df["target_entitlement_score"]

    
    print("Encoding categorical variables...")
    
    encoder = LabelEncoder()
    X["crop_type"] = encoder.fit_transform(X["crop_type"])
    X["pm_kisan_status"] = encoder.fit_transform(X["pm_kisan_status"])

    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print("Training XGBoost...")
    model = xgb.XGBRegressor(n_estimators=60, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=1, tree_method="hist")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    print(f"Validation R2 Score: {score:.4f} (Wow!)")

    
    joblib.dump(model, "xgboost_baseline.pkl")
    print("Model saved to xgboost_baseline.pkl")


if __name__ == "__main__":
    train_model()
