# SaaS Churn Early Warning System

> Predicts which customers are going silent 3 weeks before 
> they actually leave — with LLM-generated retention actions.

## The Problem
Standard churn models predict "will they leave?" too late.
By the time a customer churns, it's too late to save them.
This system detects the early warning signals 3 weeks before 
churn — giving the retention team time to act.

## What It Does
- Tracks weekly behavioral signals per customer
  (logins, feature usage, session time, API calls)
- Predicts churn risk score per customer per week
- Flags customers entering the danger zone 3 weeks early
- Groq LLM writes a personalized retention action per customer
- Live Streamlit dashboard for the customer success team

## Tech Stack
Python · XGBoost · SHAP · Groq API (Llama 3.3) · Streamlit

## Dataset
2,000 SaaS customers · 4 plans (Starter to Enterprise)
· Weekly behavioral signals · Original generated dataset

## Progress
- [x] Day 1 — Setup + dataset generated
- [x] Day 2 — XGBoost trained (ROC-AUC 0.93, precision 76%, recall 81%) — 669 high-risk customers flagged, ₹22.9L MRR at risk
- [x] Day 3 — SHAP explainability built — top churn driver: API usage decline
- [ ] Day 4 — Groq LLM retention actions
- [ ] Day 5 — Streamlit app + deployment