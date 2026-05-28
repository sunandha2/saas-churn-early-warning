import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker('en_IN')
random.seed(42)
np.random.seed(42)

NUM_CUSTOMERS = 2000
START_DATE = datetime(2023, 1, 1)

PLANS = {
    'starter':    {'price': 999,   'features': 3,  'churn_base': 0.25},
    'growth':     {'price': 2999,  'features': 8,  'churn_base': 0.15},
    'pro':        {'price': 7999,  'features': 15, 'churn_base': 0.08},
    'enterprise': {'price': 19999, 'features': 25, 'churn_base': 0.04},
}

INDUSTRIES = ['fintech', 'edtech', 'healthtech', 'ecommerce', 'saas', 'logistics']

customers = []
weekly_signals = []

for i in range(NUM_CUSTOMERS):
    customer_id = f'CUST{10000+i}'
    plan = random.choices(
        list(PLANS.keys()),
        weights=[40, 30, 20, 10]
    )[0]
    plan_info = PLANS[plan]
    industry = random.choice(INDUSTRIES)
    signup_date = START_DATE + timedelta(days=random.randint(0, 180))
    tenure_weeks = random.randint(4, 52)

    # Churn probability influenced by plan + behavior
    base_churn = plan_info['churn_base']

    # Behavioral signals that increase churn risk
    support_tickets = random.randint(0, 10)
    nps_score = random.randint(1, 10)
    feature_adoption = random.uniform(0.1, 1.0)

    # High support tickets + low NPS + low adoption = higher churn
    churn_modifier = (
        (support_tickets / 10) * 0.3 +
        ((10 - nps_score) / 10) * 0.3 +
        ((1 - feature_adoption)) * 0.4
    )
    churn_probability = min(base_churn + churn_modifier * 0.4, 0.95)
    churned = random.random() < churn_probability

    # If churned, when did early warning signals appear?
    churn_week = tenure_weeks if churned else None
    warning_week = max(1, tenure_weeks - random.randint(2, 4)) if churned else None

    customers.append({
        'customer_id': customer_id,
        'plan': plan,
        'industry': industry,
        'monthly_revenue': plan_info['price'],
        'signup_date': signup_date.strftime('%Y-%m-%d'),
        'tenure_weeks': tenure_weeks,
        'support_tickets_total': support_tickets,
        'nps_score': nps_score,
        'feature_adoption_rate': round(feature_adoption, 2),
        'churned': churned,
        'churn_week': churn_week,
        'warning_week': warning_week,
    })

    # Generate weekly behavioral signals for each customer
    for week in range(1, tenure_weeks + 1):
        # Simulate declining engagement before churn
        if churned and warning_week and week >= warning_week:
            decay = (week - warning_week + 1) * 0.15
        else:
            decay = 0

        login_frequency = max(0, round(random.normalvariate(5, 1.5) - decay * 3))
        feature_usage = max(0, round(random.normalvariate(plan_info['features'] * 0.6, 2) - decay * 2))
        session_duration_mins = max(0, round(random.normalvariate(25, 8) - decay * 10))
        support_tickets_week = max(0, int(random.random() < (0.1 + decay * 0.3)))
        api_calls = max(0, int(random.normalvariate(100, 30) * (1 - decay * 0.5)))

        # Risk score increases as churn approaches
        if churned and warning_week and week >= warning_week:
            risk_score = min(0.9, 0.3 + decay * 0.5 + random.uniform(-0.05, 0.05))
        else:
            risk_score = max(0.05, random.normalvariate(0.15, 0.05))

        weekly_signals.append({
            'customer_id': customer_id,
            'week': week,
            'login_frequency': login_frequency,
            'feature_usage_count': feature_usage,
            'session_duration_mins': session_duration_mins,
            'support_tickets': support_tickets_week,
            'api_calls': api_calls,
            'risk_score': round(risk_score, 3),
            'is_warning_week': 1 if (churned and warning_week and week == warning_week) else 0,
            'churned_this_week': 1 if (churned and churn_week and week == churn_week) else 0,
        })

customers_df = pd.DataFrame(customers)
signals_df = pd.DataFrame(weekly_signals)

customers_df.to_csv('data/customers.csv', index=False)
signals_df.to_csv('data/weekly_signals.csv', index=False)

print(f"Customers: {len(customers_df)}")
print(f"Weekly signals: {len(signals_df)}")
print(f"\nChurn rate: {customers_df['churned'].mean()*100:.1f}%")
print(f"\nPlan distribution:")
print(customers_df['plan'].value_counts())
print(f"\nAvg MRR: ₹{customers_df['monthly_revenue'].mean():,.0f}")
print(f"Total MRR: ₹{customers_df['monthly_revenue'].sum():,.0f}")
print(f"\nChurn rate by plan:")
print(customers_df.groupby('plan')['churned'].mean().apply(lambda x: f"{x*100:.1f}%"))