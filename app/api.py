from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

os.chdir(r'C:\Users\sunandha\Downloads\gitdemo\saas-churn-early-warning')

app = FastAPI(
    title="SaaS Churn Early Warning API",
    description="XGBoost churn prediction — returns churn probability and retention recommendation",
    version="1.0.0"
)

model = joblib.load('models/churn_model.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')
le_plan = joblib.load('models/le_plan.pkl')
le_industry = joblib.load('models/le_industry.pkl')

class Customer(BaseModel):
    current_logins: float = 5.0
    current_feature_usage: float = 4.0
    current_session_mins: float = 20.0
    current_api_calls: float = 100.0
    current_support_tickets: int = 0
    login_trend_3w: float = 0.0
    feature_trend_3w: float = 0.0
    session_trend_3w: float = 0.0
    api_trend_3w: float = 0.0
    avg_logins_4w: float = 5.0
    avg_session_4w: float = 20.0
    avg_api_4w: float = 100.0
    total_tickets_4w: int = 0
    login_volatility: float = 1.0
    session_volatility: float = 2.0
    total_weeks: int = 12
    plan: str = "growth"
    industry: str = "saas"
    monthly_revenue: float = 2999.0
    nps_score: int = 7
    feature_adoption_rate: float = 0.6

class ChurnResponse(BaseModel):
    churn_probability: float
    risk_level: str
    mrr_at_risk: float
    recommendation: str

@app.get("/")
def root():
    return {
        "message": "SaaS Churn Early Warning API",
        "model": "XGBoost",
        "roc_auc": 0.93,
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=ChurnResponse)
def predict(customer: Customer):
    try:
        plan_enc = le_plan.transform([customer.plan])[0]
    except:
        plan_enc = 0
    try:
        industry_enc = le_industry.transform([customer.industry])[0]
    except:
        industry_enc = 0

    features = {
        'current_logins': customer.current_logins,
        'current_feature_usage': customer.current_feature_usage,
        'current_session_mins': customer.current_session_mins,
        'current_api_calls': customer.current_api_calls,
        'current_support_tickets': customer.current_support_tickets,
        'login_trend_3w': customer.login_trend_3w,
        'feature_trend_3w': customer.feature_trend_3w,
        'session_trend_3w': customer.session_trend_3w,
        'api_trend_3w': customer.api_trend_3w,
        'avg_logins_4w': customer.avg_logins_4w,
        'avg_session_4w': customer.avg_session_4w,
        'avg_api_4w': customer.avg_api_4w,
        'total_tickets_4w': customer.total_tickets_4w,
        'login_volatility': customer.login_volatility,
        'session_volatility': customer.session_volatility,
        'total_weeks': customer.total_weeks,
        'plan_encoded': plan_enc,
        'industry_encoded': industry_enc,
        'monthly_revenue': customer.monthly_revenue,
        'nps_score': customer.nps_score,
        'feature_adoption_rate': customer.feature_adoption_rate,
    }

    X = pd.DataFrame([features])[feature_cols].fillna(0)
    churn_prob = float(model.predict_proba(X)[0][1])

    if churn_prob > 0.6:
        risk_level = "HIGH"
        recommendation = "Assign CSM for personal check-in this week"
    elif churn_prob > 0.3:
        risk_level = "MEDIUM"
        recommendation = "Proactive outreach within 2 weeks"
    else:
        risk_level = "LOW"
        recommendation = "Monitor monthly — no immediate action"

    return ChurnResponse(
        churn_probability=round(churn_prob, 4),
        risk_level=risk_level,
        mrr_at_risk=customer.monthly_revenue if churn_prob > 0.6 else 0,
        recommendation=recommendation
    )