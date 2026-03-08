from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(title="Customer Churn Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent
model = joblib.load(BASE / "churn_model.joblib")
with open(BASE / "model_meta.json") as f:
    meta = json.load(f)

THRESHOLD = meta["threshold"]

class CustomerData(BaseModel):
    gender: str
    age: int
    country: str
    city: str
    customer_segment: str
    tenure_months: int
    signup_channel: str
    contract_type: str
    monthly_logins: int
    weekly_active_days: int
    avg_session_time: float
    features_used: int
    usage_growth_rate: float
    last_login_days_ago: int
    monthly_fee: float
    total_revenue: float
    payment_method: str
    payment_failures: int
    discount_applied: str
    price_increase_last_3m: str
    support_tickets: int
    avg_resolution_time: float
    complaint_type: str
    csat_score: float
    escalations: int
    email_open_rate: float
    marketing_click_rate: float
    nps_score: int
    survey_response: str
    referral_count: int

@app.get("/")
def root():
    return {"status": "ok", "message": "Churn Predictor API is running"}

@app.get("/meta")
def get_meta():
    return {
        "cat_options": meta["cat_options"],
        "top_features": meta["top_features"],
        "auc": meta["auc"]
    }

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.model_dump()])
    prob = model.predict_proba(df)[0][1]
    churn = int(prob >= THRESHOLD)

    # Risk factors - top features with their values
    top = meta["top_features"][:6]
    risk_factors = []

    checks = {
        "csat_score": (data.csat_score <= 2, f"Low satisfaction score: {data.csat_score}/5"),
        "tenure_months": (data.tenure_months <= 6, f"New customer: {data.tenure_months} months"),
        "monthly_logins": (data.monthly_logins <= 5, f"Low engagement: {data.monthly_logins} logins/mo"),
        "payment_failures": (data.payment_failures >= 2, f"Payment issues: {data.payment_failures} failures"),
        "last_login_days_ago": (data.last_login_days_ago >= 30, f"Inactive: {data.last_login_days_ago} days since login"),
        "nps_score": (data.nps_score < 0, f"Negative NPS: {data.nps_score}"),
        "support_tickets": (data.support_tickets >= 3, f"High support load: {data.support_tickets} tickets"),
        "escalations": (data.escalations >= 2, f"Escalations: {data.escalations}"),
        "usage_growth_rate": (data.usage_growth_rate < -0.2, f"Usage declining: {data.usage_growth_rate:.0%}"),
        "price_increase_last_3m": (data.price_increase_last_3m == "Yes", "Price increased recently"),
        "survey_response": (data.survey_response == "Unsatisfied", "Survey: Unsatisfied"),
        "contract_type": (data.contract_type == "Monthly", "Month-to-month contract (easy to cancel)"),
    }

    for key, (condition, message) in checks.items():
        if condition:
            risk_factors.append(message)

    return {
        "churn_probability": round(float(prob), 4),
        "churn_prediction": churn,
        "risk_level": "High" if prob >= 0.6 else "Medium" if prob >= 0.35 else "Low",
        "risk_factors": risk_factors[:5],
        "retention_score": round((1 - prob) * 100, 1)
    }
