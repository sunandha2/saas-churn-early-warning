import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

os.chdir(r'C:\Users\sunandha\Downloads\gitdemo\saas-churn-early-warning')

print("=" * 60)
print("MODEL COMPARISON — SaaS Churn Prediction")
print("=" * 60)

feature_cols = joblib.load('models/feature_cols.pkl')
risk_scores = pd.read_csv('outputs/customer_risk_scores.csv')

X = risk_scores[feature_cols].fillna(0)
y = risk_scores['churned'].astype(int)

print(f"Dataset: {len(X)} customers")
print(f"Churn rate: {y.mean()*100:.1f}%")
print(f"Features: {len(feature_cols)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=5,
        random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.1, random_state=42,
        eval_metric='logloss', verbosity=0
    ),
}

results = []

for name, m in models.items():
    print(f"\nTraining {name}...")
    m.fit(X_train_bal, y_train_bal)
    y_pred = m.predict(X_test)
    y_proba = m.predict_proba(X_test)[:, 1]

    results.append({
        'Model': name,
        'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1': round(f1_score(y_test, y_pred), 4),
    })
    print(f"  ROC-AUC: {results[-1]['ROC-AUC']}")

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

results_df = pd.DataFrame(results).sort_values(
    'ROC-AUC', ascending=False
)
print(results_df.to_string(index=False))

results_df.to_csv('outputs/model_comparison.csv', index=False)
print("\nSaved: outputs/model_comparison.csv")

print("\nWhy XGBoost wins for churn prediction:")
print("1. Captures non-linear behavioral patterns")
print("2. Handles correlated features (logins, sessions, API calls)")
print("3. Built-in feature importance + SHAP compatibility")
print("4. Robust to outliers in behavioral signals")