import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'C:\Users\sunandha\Downloads\gitdemo\saas-churn-early-warning')
os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("STEP 1 — Loading model and data")
print("=" * 60)

model = joblib.load('models/churn_model.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')
le_plan = joblib.load('models/le_plan.pkl')
le_industry = joblib.load('models/le_industry.pkl')

risk_scores = pd.read_csv('outputs/customer_risk_scores.csv')
customers = pd.read_csv('data/customers.csv')
signals = pd.read_csv('data/weekly_signals.csv')

print(f"Model loaded: {type(model)}")
print(f"Features: {len(feature_cols)}")
print(f"Customers with risk scores: {len(risk_scores)}")

print("\n" + "=" * 60)
print("STEP 2 — Building SHAP explainer")
print("=" * 60)

# Prepare feature matrix
X = risk_scores[feature_cols].fillna(0)

# TreeExplainer is optimized for XGBoost — fast and exact
explainer = shap.TreeExplainer(model)
print("SHAP TreeExplainer built")

# Calculate SHAP values for all customers
# Each value = how much that feature pushed prediction up/down
print("Calculating SHAP values for all customers...")
shap_values = explainer.shap_values(X)
print(f"SHAP values shape: {shap_values.shape}")
print(f"One row per customer, one column per feature")

print("\n" + "=" * 60)
print("STEP 3 — Global feature importance")
print("=" * 60)

# Mean absolute SHAP value = global importance
mean_shap = np.abs(shap_values).mean(axis=0)
global_importance = pd.DataFrame({
    'feature': feature_cols,
    'mean_shap': mean_shap
}).sort_values('mean_shap', ascending=False)

print("\nTop 10 features by SHAP importance:")
print(global_importance.head(10).to_string(index=False))

# Save global importance
global_importance.to_csv('outputs/shap_global_importance.csv', index=False)

print("\n" + "=" * 60)
print("STEP 4 — Per-customer SHAP explanations")
print("=" * 60)

def get_customer_explanation(customer_idx, top_n=5):
    """Get top SHAP factors for a specific customer"""
    shap_row = shap_values[customer_idx]
    customer_features = X.iloc[customer_idx]

    # Create explanation dataframe
    explanation = pd.DataFrame({
        'feature': feature_cols,
        'shap_value': shap_row,
        'feature_value': customer_features.values
    })

    # Sort by absolute SHAP value
    explanation['abs_shap'] = explanation['shap_value'].abs()
    explanation = explanation.sort_values('abs_shap', ascending=False)

    return explanation.head(top_n)

def format_explanation(customer_id, explanation_df, churn_prob, risk_level):
    """Format SHAP explanation into readable text"""
    lines = []
    lines.append(f"Customer: {customer_id}")
    lines.append(f"Churn Probability: {churn_prob:.1%}")
    lines.append(f"Risk Level: {risk_level}")
    lines.append(f"Top factors driving this prediction:")

    for _, row in explanation_df.iterrows():
        direction = "increases" if row['shap_value'] > 0 else "decreases"
        impact = abs(row['shap_value'])
        lines.append(
            f"  • {row['feature']}: value={row['feature_value']:.2f} "
            f"→ {direction} churn risk by {impact:.3f}"
        )

    return "\n".join(lines)

# Show explanations for 5 high-risk customers
high_risk = risk_scores[
    risk_scores['risk_level'] == 'High'
].head(5)

print("\nSample explanations for high-risk customers:")
for idx, row in high_risk.iterrows():
    customer_idx = risk_scores.index.get_loc(idx)
    explanation = get_customer_explanation(customer_idx)
    text = format_explanation(
        row['customer_id'],
        explanation,
        row['churn_probability'],
        row['risk_level']
    )
    print(f"\n{text}")
    print("-" * 50)

print("\n" + "=" * 60)
print("STEP 5 — Save SHAP values for all customers")
print("=" * 60)

# Create full SHAP dataframe
shap_df = pd.DataFrame(
    shap_values,
    columns=[f'shap_{c}' for c in feature_cols]
)
shap_df['customer_id'] = risk_scores['customer_id'].values
shap_df['churn_probability'] = risk_scores['churn_probability'].values
shap_df['risk_level'] = risk_scores['risk_level'].values

shap_df.to_csv('outputs/shap_values.csv', index=False)
print(f"Saved: outputs/shap_values.csv ({len(shap_df)} customers)")

print("\n" + "=" * 60)
print("STEP 6 — Weekly risk trajectory")
print("=" * 60)

# Show how risk evolved week by week for churned customers
churned_customers = customers[customers['churned'] == True]['customer_id'].tolist()
sample_churned = churned_customers[:3]

print("\nWeekly behavioral signals for 3 churned customers:")
for cust_id in sample_churned:
    cust_signals = signals[
        signals['customer_id'] == cust_id
    ].sort_values('week')

    print(f"\n{cust_id} (churned at week {cust_signals['week'].max()}):")
    print(cust_signals[[
        'week', 'login_frequency',
        'feature_usage_count', 'session_duration_mins',
        'support_tickets'
    ]].tail(6).to_string(index=False))

print("\n" + "=" * 60)
print("STEP 7 — Business impact summary")
print("=" * 60)

high_risk_customers = risk_scores[risk_scores['risk_level'] == 'High']
medium_risk_customers = risk_scores[risk_scores['risk_level'] == 'Medium']

print(f"\nRisk Summary:")
print(f"High risk customers:   {len(high_risk_customers):,}")
print(f"Medium risk customers: {len(medium_risk_customers):,}")
print(f"\nMRR at high risk:   ₹{high_risk_customers['monthly_revenue'].sum():,.0f}")
print(f"MRR at medium risk: ₹{medium_risk_customers['monthly_revenue'].sum():,.0f}")
print(f"Total MRR at risk:  ₹{(high_risk_customers['monthly_revenue'].sum() + medium_risk_customers['monthly_revenue'].sum()):,.0f}")

# Top features driving churn globally
print(f"\nTop 5 features driving churn risk globally:")
for _, row in global_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['mean_shap']:.4f} avg impact")

print("Outputs saved:")
print("  outputs/shap_global_importance.csv")
print("  outputs/shap_values.csv")