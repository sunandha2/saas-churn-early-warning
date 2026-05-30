import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score,
                             precision_recall_curve)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'C:\Users\sunandha\Downloads\gitdemo\saas-churn-early-warning')

print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

customers = pd.read_csv('data/customers.csv')
signals = pd.read_csv('data/weekly_signals.csv')

print(f"Customers: {len(customers)}")
print(f"Weekly signals: {len(signals)}")
print(f"Churn rate: {customers['churned'].mean()*100:.1f}%")

# ── STEP 2: FEATURE ENGINEERING ──────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Feature Engineering")
print("=" * 60)

# Remove leakage columns — model must NOT see these
LEAKAGE_COLS = ['risk_score', 'is_warning_week', 'churned_this_week']
signals_clean = signals.drop(columns=LEAKAGE_COLS)

print(f"Removed leakage columns: {LEAKAGE_COLS}")
print(f"Clean signal columns: {signals_clean.columns.tolist()}")

# For each customer, take their LAST 4 weeks of signals
# This simulates real-time prediction: "given last 4 weeks, will they churn?"
def engineer_features(signals_df, customers_df):
    features = []

    for customer_id in signals_df['customer_id'].unique():
        cust_signals = signals_df[
            signals_df['customer_id'] == customer_id
        ].sort_values('week')

        # Take last 4 weeks (or all if less than 4)
        recent = cust_signals.tail(4)

        if len(recent) < 2:
            continue

        # ── CURRENT STATE FEATURES ──
        latest = recent.iloc[-1]
        current_logins = latest['login_frequency']
        current_features = latest['feature_usage_count']
        current_session = latest['session_duration_mins']
        current_api = latest['api_calls']
        current_tickets = latest['support_tickets']

        # ── TREND FEATURES (most important for early warning) ──
        # These capture DECLINING engagement before churn
        login_trend = recent['login_frequency'].diff().mean()
        feature_trend = recent['feature_usage_count'].diff().mean()
        session_trend = recent['session_duration_mins'].diff().mean()
        api_trend = recent['api_calls'].diff().mean()

        # ── AGGREGATED FEATURES ──
        avg_logins = recent['login_frequency'].mean()
        avg_session = recent['session_duration_mins'].mean()
        avg_api = recent['api_calls'].mean()
        total_tickets = recent['support_tickets'].sum()

        # ── VOLATILITY FEATURES ──
        login_std = recent['login_frequency'].std()
        session_std = recent['session_duration_mins'].std()

        # ── WEEKS OF DATA ──
        total_weeks = len(cust_signals)

        features.append({
            'customer_id': customer_id,
            # Current state
            'current_logins': current_logins,
            'current_feature_usage': current_features,
            'current_session_mins': current_session,
            'current_api_calls': current_api,
            'current_support_tickets': current_tickets,
            # Trends (negative = declining = bad sign)
            'login_trend_3w': round(login_trend, 3),
            'feature_trend_3w': round(feature_trend, 3),
            'session_trend_3w': round(session_trend, 3),
            'api_trend_3w': round(api_trend, 3),
            # Aggregates
            'avg_logins_4w': round(avg_logins, 2),
            'avg_session_4w': round(avg_session, 2),
            'avg_api_4w': round(avg_api, 2),
            'total_tickets_4w': total_tickets,
            # Volatility
            'login_volatility': round(login_std, 3) if not pd.isna(login_std) else 0,
            'session_volatility': round(session_std, 3) if not pd.isna(session_std) else 0,
            # Tenure
            'total_weeks': total_weeks,
        })

    features_df = pd.DataFrame(features)

    # Merge with customer info
    customer_info = customers_df[[
        'customer_id', 'plan', 'industry',
        'monthly_revenue', 'nps_score',
        'feature_adoption_rate', 'churned'
    ]].copy()

    merged = features_df.merge(customer_info, on='customer_id')
    return merged

print("Engineering features...")
feature_df = engineer_features(signals_clean, customers)
print(f"Feature matrix shape: {feature_df.shape}")
print(f"Features created: {[c for c in feature_df.columns if c not in ['customer_id', 'churned']]}")

# ── STEP 3: ENCODE CATEGORICALS ──────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Encoding categorical features")
print("=" * 60)

le_plan = LabelEncoder()
le_industry = LabelEncoder()

feature_df['plan_encoded'] = le_plan.fit_transform(feature_df['plan'])
feature_df['industry_encoded'] = le_industry.fit_transform(feature_df['industry'])

print(f"Plan encoding: {dict(zip(le_plan.classes_, le_plan.transform(le_plan.classes_)))}")
print(f"Industry encoding: {dict(zip(le_industry.classes_, le_industry.transform(le_industry.classes_)))}")

# Save encoders for the app
os.makedirs('models', exist_ok=True)
joblib.dump(le_plan, 'models/le_plan.pkl')
joblib.dump(le_industry, 'models/le_industry.pkl')

# ── STEP 4: PREPARE TRAIN/TEST SPLIT ─────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Train/test split")
print("=" * 60)

FEATURE_COLS = [
    'current_logins', 'current_feature_usage', 'current_session_mins',
    'current_api_calls', 'current_support_tickets',
    'login_trend_3w', 'feature_trend_3w', 'session_trend_3w', 'api_trend_3w',
    'avg_logins_4w', 'avg_session_4w', 'avg_api_4w', 'total_tickets_4w',
    'login_volatility', 'session_volatility', 'total_weeks',
    'plan_encoded', 'industry_encoded',
    'monthly_revenue', 'nps_score', 'feature_adoption_rate'
]

X = feature_df[FEATURE_COLS]
y = feature_df['churned'].astype(int)

print(f"Feature columns: {len(FEATURE_COLS)}")
print(f"Class distribution: {y.value_counts().to_dict()}")
print(f"Churn rate in features: {y.mean()*100:.1f}%")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ── STEP 5: HANDLE CLASS IMBALANCE WITH SMOTE ────────────────
print("\n" + "=" * 60)
print("STEP 5 — Handling class imbalance with SMOTE")
print("=" * 60)

print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# ── STEP 6: TRAIN XGBOOST ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Training XGBoost model")
print("=" * 60)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

model.fit(
    X_train_balanced, y_train_balanced,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("Model trained successfully")

# ── STEP 7: EVALUATE ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Model Evaluation")
print("=" * 60)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Active', 'Churned']))

auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {auc:.4f}")

# ── STEP 8: FEATURE IMPORTANCE ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Feature Importance")
print("=" * 60)

importance_df = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(importance_df.head(10).to_string(index=False))

# ── STEP 9: GENERATE RISK SCORES FOR ALL CUSTOMERS ───────────
print("\n" + "=" * 60)
print("STEP 9 — Generating risk scores for all customers")
print("=" * 60)

feature_df['churn_probability'] = model.predict_proba(
    feature_df[FEATURE_COLS]
)[:, 1]

feature_df['risk_level'] = pd.cut(
    feature_df['churn_probability'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low', 'Medium', 'High']
)

risk_summary = feature_df.groupby('risk_level').agg(
    customers=('customer_id', 'count'),
    avg_mrr=('monthly_revenue', 'mean'),
    total_mrr=('monthly_revenue', 'sum')
).reset_index()

print("\nRisk Level Distribution:")
print(risk_summary.to_string(index=False))

high_risk = feature_df[feature_df['risk_level'] == 'High']
print(f"\nHigh risk customers: {len(high_risk)}")
print(f"MRR at risk: ₹{high_risk['monthly_revenue'].sum():,.0f}")

# ── STEP 10: SAVE EVERYTHING ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — Saving model and outputs")
print("=" * 60)

joblib.dump(model, 'models/churn_model.pkl')
joblib.dump(FEATURE_COLS, 'models/feature_cols.pkl')

os.makedirs('outputs', exist_ok=True)
feature_df.to_csv('outputs/customer_risk_scores.csv', index=False)
importance_df.to_csv('outputs/feature_importance.csv', index=False)

print("Saved:")
print("  models/churn_model.pkl")
print("  models/le_plan.pkl")
print("  models/le_industry.pkl")
print("  models/feature_cols.pkl")
print("  outputs/customer_risk_scores.csv")
print("  outputs/feature_importance.csv")
print(f"\nROC-AUC: {auc:.4f}")
print(f"High risk customers: {len(high_risk)}")
print(f"MRR at risk: ₹{high_risk['monthly_revenue'].sum():,.0f}")
