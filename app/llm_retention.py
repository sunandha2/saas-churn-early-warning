import pandas as pd
import numpy as np
import joblib
import os
from groq import Groq
from dotenv import load_dotenv
import json
import time

load_dotenv()
os.chdir(r'C:\Users\sunandha\Downloads\gitdemo\saas-churn-early-warning')

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

risk_scores = pd.read_csv('outputs/customer_risk_scores.csv')
shap_values = pd.read_csv('outputs/shap_values.csv')
customers = pd.read_csv('data/customers.csv')
signals = pd.read_csv('data/weekly_signals.csv')

print(f"Customers with risk scores: {len(risk_scores)}")
print(f"High risk customers: {len(risk_scores[risk_scores['risk_level']=='High'])}")

# ── BUSINESS CONTEXT ──────────────────────────────────────────
RAG_CONTEXT = """
You are a senior customer success analyst at a SaaS company.

PRODUCT PLANS:
- Starter (₹999/mo): 3 features, small teams, high churn risk
- Growth (₹2999/mo): 8 features, growing teams
- Pro (₹7999/mo): 15 features, established teams
- Enterprise (₹19999/mo): 25 features, large organizations

HEALTHY BENCHMARKS:
- Login frequency: 5-8 per week is healthy, below 3 is danger zone
- Session duration: 20-30 mins is healthy, below 10 is danger zone
- API calls: Active users make 80-150 API calls/week
- Feature usage: Should use at least 40% of available features
- Support tickets: 1-2/month is normal, 3+ in a week is a red flag

CHURN RISK LEVELS:
- High (>60%): Needs immediate intervention this week
- Medium (30-60%): Needs proactive outreach next 2 weeks
- Low (<30%): Monitor monthly

RETENTION PLAYBOOK:
- Login drop: Personal check-in call, feature walkthrough
- Session drop: Offer onboarding refresh, highlight new features
- API drop: Technical health check, integration support
- High tickets: Priority support escalation, product feedback session
- Starter plan high risk: Offer upgrade trial or renewal discount
- Long tenure + high risk: Executive business review
"""

print("\n" + "=" * 60)
print("STEP 2 — Building SHAP explanation per customer")
print("=" * 60)

def get_shap_explanation(customer_id):
    """Get top SHAP factors for a customer in readable format"""
    shap_row = shap_values[
        shap_values['customer_id'] == customer_id
    ]
    if shap_row.empty:
        return []

    shap_row = shap_row.iloc[0]
    shap_cols = [c for c in shap_values.columns
                 if c.startswith('shap_')]

    factors = []
    for col in shap_cols:
        feature_name = col.replace('shap_', '')
        shap_val = shap_row[col]
        factors.append({
            'feature': feature_name,
            'shap_value': shap_val,
            'direction': 'increases' if shap_val > 0 else 'decreases'
        })

    # Sort by absolute SHAP value
    factors.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    return factors[:5]

def get_recent_signals(customer_id):
    """Get last 4 weeks of behavioral signals"""
    cust_signals = signals[
        signals['customer_id'] == customer_id
    ].sort_values('week').tail(4)

    return cust_signals[[
        'week', 'login_frequency',
        'feature_usage_count',
        'session_duration_mins',
        'support_tickets',
        'api_calls'
    ]].to_dict('records')

def generate_retention_action(customer_data, shap_factors,
                               recent_signals):
    """Use Groq LLM to generate personalized retention action"""

    # Format SHAP factors
    shap_text = "\n".join([
        f"  - {f['feature']}: {f['direction']} churn risk "
        f"(impact: {abs(f['shap_value']):.3f})"
        for f in shap_factors
    ])

    # Format recent signals
    signals_text = "\n".join([
        f"  Week {s['week']}: logins={s['login_frequency']}, "
        f"features={s['feature_usage_count']}, "
        f"session={s['session_duration_mins']}min, "
        f"tickets={s['support_tickets']}, "
        f"api={s['api_calls']}"
        for s in recent_signals
    ])

    prompt = f"""
{RAG_CONTEXT}

CUSTOMER PROFILE:
- ID: {customer_data['customer_id']}
- Plan: {customer_data['plan']} (₹{customer_data['monthly_revenue']}/mo)
- Industry: {customer_data['industry']}
- Tenure: {customer_data['total_weeks']} weeks
- NPS Score: {customer_data['nps_score']}/10
- Churn Probability: {customer_data['churn_probability']:.1%}
- Risk Level: {customer_data['risk_level']}

TOP CHURN RISK FACTORS (SHAP):
{shap_text}

RECENT BEHAVIORAL SIGNALS (last 4 weeks):
{signals_text}

Write a retention action report with exactly 3 sections:
1. SITUATION (2 sentences): What is happening with this customer based on the data
2. ROOT CAUSE (1 sentence): The primary reason they are likely to churn
3. ACTION (2 sentences): Specific retention action the customer success team should take this week

Be specific — use the actual numbers. Sound like a senior analyst, not a bot.
Keep total response under 150 words.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

print("\n" + "=" * 60)
print("STEP 3 — Generating retention actions for high-risk customers")
print("=" * 60)

# Get top 10 highest risk customers
high_risk = risk_scores[
    risk_scores['risk_level'] == 'High'
].sort_values('churn_probability', ascending=False).head(10)

print(f"Generating retention actions for top 10 high-risk customers...")
print(f"MRR at stake: ₹{high_risk['monthly_revenue'].sum():,.0f}\n")

retention_reports = []

for _, customer in high_risk.iterrows():
    customer_id = customer['customer_id']

    # Get supporting data
    shap_factors = get_shap_explanation(customer_id)
    recent_signals = get_recent_signals(customer_id)

    print(f"Analyzing {customer_id} "
          f"({customer['churn_probability']:.1%} risk, "
          f"₹{customer['monthly_revenue']:,}/mo)...")

    # Generate LLM retention action
    action = generate_retention_action(
        customer, shap_factors, recent_signals
    )

    retention_reports.append({
        'customer_id': customer_id,
        'plan': customer['plan'],
        'monthly_revenue': customer['monthly_revenue'],
        'churn_probability': customer['churn_probability'],
        'risk_level': customer['risk_level'],
        'retention_action': action
    })

    time.sleep(0.5)  # Rate limiting

print("\n" + "=" * 60)
print("STEP 4 — Sample retention reports")
print("=" * 60)

for report in retention_reports[:3]:
    print(f"\n{'='*50}")
    print(f"Customer: {report['customer_id']}")
    print(f"Plan: {report['plan']} | "
          f"MRR: ₹{report['monthly_revenue']:,} | "
          f"Risk: {report['churn_probability']:.1%}")
    print(f"\n{report['retention_action']}")

print("\n" + "=" * 60)
print("STEP 5 — Saving retention reports")
print("=" * 60)

reports_df = pd.DataFrame(retention_reports)
reports_df.to_csv('outputs/retention_reports.csv', index=False)
print(f"Saved: outputs/retention_reports.csv")
print(f"Reports generated: {len(reports_df)}")
print(f"Total MRR covered: ₹{reports_df['monthly_revenue'].sum():,.0f}")

print("\n" + "=" * 60)
print("STEP 6 — Business summary")
print("=" * 60)

print(f"\nRetention Action Summary:")
print(f"Customers analyzed: {len(reports_df)}")
print(f"MRR at immediate risk: ₹{reports_df['monthly_revenue'].sum():,.0f}")
print(f"Avg churn probability: {reports_df['churn_probability'].mean():.1%}")

plan_breakdown = reports_df.groupby('plan').agg(
    customers=('customer_id', 'count'),
    mrr=('monthly_revenue', 'sum')
).reset_index()

print(f"\nBreakdown by plan:")
print(plan_breakdown.to_string(index=False))

print("The LLM has generated personalized retention")
print("actions for your highest-risk customers.")