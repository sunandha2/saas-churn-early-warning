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
- [ ] Day 1 — Setup + dataset generated
- [ ] Day 2 — Feature engineering + XGBoost model
- [ ] Day 3 — Weekly risk scoring pipeline
- [ ] Day 4 — Groq LLM retention actions
- [ ] Day 5 — Streamlit app + deployment