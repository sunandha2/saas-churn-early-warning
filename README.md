# SaaS Churn Early Warning System

> Predicts which customers are going silent 3 weeks before
> they actually leave — with LLM-generated retention actions.

<img width="1810" height="792" alt="image" src="https://github.com/user-attachments/assets/dce5b66d-0346-4848-b771-fecf067bbc31" />
<img width="1806" height="793" alt="image" src="https://github.com/user-attachments/assets/7a07eac8-eb23-41ec-8cc1-6c56cbfc6000" />
<img width="1815" height="652" alt="image" src="https://github.com/user-attachments/assets/992b4c59-2ac5-4f16-8406-1f8f4b40c649" />
<img width="1878" height="720" alt="image" src="https://github.com/user-attachments/assets/b8462eb7-67d9-46d7-a2c1-6ad382e03230" />

## Live Demo
🔗https://saas-churn-early-warning-8txziep77uauxcgtzuddqq.streamlit.app/

## The Problem
Standard churn models predict "will they leave?" too late.
By the time a customer churns, it's too late to save them.
This system detects early warning signals 3 weeks before
churn — giving the retention team time to act.

## What It Does
- Tracks weekly behavioral signals per customer
  (logins, feature usage, session time, API calls)
- Predicts churn risk score per customer per week
- Flags customers entering the danger zone early
- SHAP explains WHY each customer is at risk
- Groq LLM writes personalized retention action per customer
- Live 4-page Streamlit dashboard

## Example Output
Customer CUST10676 — 100% churn risk

SITUATION: Login frequency dropped from 5 to 1 over
last 4 weeks, 0 features used in weeks 43, 45, 46.
NPS score is 1/10.

ROOT CAUSE: Drastically decreased engagement indicates
lack of value realization from the product.

ACTION: Conduct personal check-in call this week.
Offer Growth plan upgrade trial or renewal discount.

## Tech Stack
| Tool | Purpose |
|---|---|
| Python + Faker | Generated 2,000 customer dataset |
| XGBoost | Churn prediction model |
| SMOTE | Class imbalance handling |
| SHAP | Per-customer explainability |
| Groq API (Llama 3.3) | LLM retention action generation |
| Streamlit | Live 4-page dashboard |

## Model Performance
- ROC-AUC: 0.93
- Precision: 76%
- Recall: 81%
- High risk customers: 669
- MRR at risk: ₹22.9L

## Dataset
- 2,000 SaaS customers across 4 plans
- Starter (₹999) → Enterprise (₹19,999)
- 54,600 weekly behavioral signals
- 33.6% churn rate
- Original generated dataset

## Key Finding
Top churn driver: API usage decline (SHAP impact: 1.46)
followed by login frequency drop (1.19).
Customers who stop using the API are most likely to churn.

## Progress
- [x] Day 1 — Dataset generated (2,000 customers, 54,600 signals)
- [x] Day 2 — XGBoost trained (ROC-AUC 0.93)
- [x] Day 3 — SHAP explainability built
- [x] Day 4 — Groq LLM retention actions generated
- [x] Day 5 — Live Streamlit app deployed

## How to Run
```bash
git clone https://github.com/sunandha2/saas-churn-early-warning
cd saas-churn-early-warning
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python data/generate_data.py
python notebooks/train_model.py
streamlit run app/main.py
```

## Resume Bullet
"Built SaaS churn early warning system — XGBoost on
54,600 behavioral signals (ROC-AUC 0.93), SHAP
explainability, Groq LLM retention actions per customer.
669 high-risk customers flagged, ₹22.9L MRR at risk.
Live Streamlit dashboard deployed."

## Author
Built to demonstrate end-to-end ML + explainability +
LLM integration for SaaS customer retention.
