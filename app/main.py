import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="SaaS Churn Early Warning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; }
    .main { background-color: #0a0a0f; }
    .metric-card {
        background: #111118;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #1e1e2e;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #4fc3f7; }
    .metric-label { font-size: 13px; color: #666; margin-top: 4px; }
    .high-risk { background: #2d0a0a; border-left: 4px solid #e74c3c;
                 border-radius: 6px; padding: 12px 16px; margin: 8px 0; }
    .low-risk { background: #0a2d0a; border-left: 4px solid #2ecc71;
                border-radius: 6px; padding: 12px 16px; margin: 8px 0; }
    .report-box { background: #111118; border-left: 4px solid #4fc3f7;
                  border-radius: 6px; padding: 16px; margin: 12px 0;
                  font-size: 14px; line-height: 1.7; color: #ddd; }
    h1, h2, h3 { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    risk = pd.read_csv('outputs/customer_risk_scores.csv')
    reports = pd.read_csv('outputs/retention_reports.csv')
    shap = pd.read_csv('outputs/shap_values.csv')
    signals = pd.read_csv('data/weekly_signals.csv')
    customers = pd.read_csv('data/customers.csv')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return risk, reports, shap, signals, customers, feature_cols

@st.cache_resource
def load_model():
    return joblib.load('models/churn_model.pkl')

risk, reports, shap, signals, customers, feature_cols = load_data()
model = load_model()

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## SaaS Churn Early Warning")
    st.markdown("*XGBoost + SHAP + Groq LLM*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Dashboard", "Customer Explorer", "Retention Reports", "Model Insights"],
        label_visibility="collapsed",
        key="nav"
    )

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"- {len(risk):,} customers")
    st.markdown(f"- {risk['churned'].sum()} churned")
    st.markdown(f"- {risk['churned'].mean()*100:.1f}% churn rate")
    st.markdown(f"- ₹{risk['monthly_revenue'].sum():,.0f} total MRR")
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.markdown("- ROC-AUC: 0.93")
    st.markdown("- Precision: 76%")
    st.markdown("- Recall: 81%")
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("XGBoost · SHAP · Groq · Streamlit")

# ── PAGE 1: DASHBOARD ──────────────────────────────────────────
if "Dashboard" in page:
    st.markdown("# SaaS Churn Early Warning Dashboard")
    st.markdown("*2,000 SaaS customers — behavioral churn prediction*")
    st.markdown("---")

    high_risk = risk[risk['risk_level'] == 'High']
    medium_risk = risk[risk['risk_level'] == 'Medium']
    low_risk = risk[risk['risk_level'] == 'Low']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#e74c3c">
                {len(high_risk)}
            </div>
            <div class="metric-label">High Risk Customers</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#f39c12">
                {len(medium_risk)}
            </div>
            <div class="metric-label">Medium Risk</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#e74c3c">
                ₹{high_risk['monthly_revenue'].sum():,.0f}
            </div>
            <div class="metric-label">MRR at High Risk</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        total_at_risk = (
            high_risk['monthly_revenue'].sum() +
            medium_risk['monthly_revenue'].sum()
        )
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#f39c12">
                ₹{total_at_risk:,.0f}
            </div>
            <div class="metric-label">Total MRR at Risk</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        risk_counts = risk['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        fig1 = px.pie(
            risk_counts,
            values='Count',
            names='Risk Level',
            color='Risk Level',
            color_discrete_map={
                'High': '#e74c3c',
                'Medium': '#f39c12',
                'Low': '#2ecc71'
            },
            title='Customer Risk Distribution',
            hole=0.4
        )
        fig1.update_layout(
            plot_bgcolor='#111118',
            paper_bgcolor='#0a0a0f',
            font=dict(color='white'),
            height=350
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        plan_churn = risk.groupby('plan').agg(
            total=('customer_id', 'count'),
            high_risk=('risk_level', lambda x: (x=='High').sum()),
            mrr=('monthly_revenue', 'sum')
        ).reset_index()
        plan_churn['high_risk_rate'] = (
            plan_churn['high_risk'] /
            plan_churn['total'] * 100
        ).round(1)

        fig2 = px.bar(
            plan_churn.sort_values('high_risk_rate', ascending=False),
            x='plan',
            y='high_risk_rate',
            color='high_risk_rate',
            color_continuous_scale=['#2ecc71', '#e74c3c'],
            title='High Risk Rate by Plan',
            text='high_risk_rate'
        )
        fig2.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig2.update_layout(
            plot_bgcolor='#111118',
            paper_bgcolor='#0a0a0f',
            font=dict(color='white'),
            height=350,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Top at-risk customers table
    st.markdown("### Top 20 Highest Risk Customers")
    display = high_risk.sort_values(
        'churn_probability', ascending=False
    )[[
        'customer_id', 'plan', 'monthly_revenue',
        'churn_probability', 'nps_score',
        'total_weeks', 'risk_level'
    ]].head(20).copy()

    display['churn_probability'] = display[
        'churn_probability'
    ].apply(lambda x: f"{x:.1%}")
    display.columns = [
        'Customer ID', 'Plan', 'MRR (₹)',
        'Churn Risk', 'NPS', 'Tenure (weeks)', 'Risk Level'
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)

# ── PAGE 2: CUSTOMER EXPLORER ──────────────────────────────────
elif "Explorer" in page:
    st.markdown("# Customer Explorer")
    st.markdown("*Drill into any customer's behavioral signals*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        plan_filter = st.selectbox(
            "Plan",
            ["All"] + sorted(risk['plan'].unique().tolist())
        )
    with col2:
        risk_filter = st.selectbox(
            "Risk Level",
            ["All", "High", "Medium", "Low"]
        )
    with col3:
        churned_filter = st.selectbox(
            "Status",
            ["All", "Churned", "Active"]
        )

    filtered = risk.copy()
    if plan_filter != "All":
        filtered = filtered[filtered['plan'] == plan_filter]
    if risk_filter != "All":
        filtered = filtered[filtered['risk_level'] == risk_filter]
    if churned_filter == "Churned":
        filtered = filtered[filtered['churned'] == True]
    elif churned_filter == "Active":
        filtered = filtered[filtered['churned'] == False]

    st.markdown(f"**{len(filtered):,} customers matching filters**")

    display = filtered[[
        'customer_id', 'plan', 'monthly_revenue',
        'churn_probability', 'nps_score',
        'feature_adoption_rate', 'risk_level', 'churned'
    ]].head(50).copy()
    display['churn_probability'] = display[
        'churn_probability'
    ].apply(lambda x: f"{x:.1%}")
    display['feature_adoption_rate'] = display[
        'feature_adoption_rate'
    ].apply(lambda x: f"{x:.0%}")
    display['churned'] = display['churned'].apply(
        lambda x: "Churned" if x else "Active"
    )
    display.columns = [
        'Customer ID', 'Plan', 'MRR (₹)', 'Churn Risk',
        'NPS', 'Feature Adoption', 'Risk Level', 'Status'
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Weekly signals for selected customer
    st.markdown("---")
    st.markdown("### Weekly Behavioral Signals")
    customer_ids = filtered['customer_id'].tolist()
    selected = st.selectbox("Select customer", customer_ids[:50])

    if selected:
        cust_signals = signals[
            signals['customer_id'] == selected
        ].sort_values('week')

        col1, col2 = st.columns(2)

        with col1:
            fig3 = px.line(
                cust_signals,
                x='week',
                y=['login_frequency', 'feature_usage_count'],
                title=f'{selected} — Login & Feature Usage',
                color_discrete_map={
                    'login_frequency': '#4fc3f7',
                    'feature_usage_count': '#2ecc71'
                }
            )
            fig3.update_layout(
                plot_bgcolor='#111118',
                paper_bgcolor='#0a0a0f',
                font=dict(color='white'),
                height=300
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            fig4 = px.line(
                cust_signals,
                x='week',
                y=['session_duration_mins', 'api_calls'],
                title=f'{selected} — Session & API Usage',
                color_discrete_map={
                    'session_duration_mins': '#f39c12',
                    'api_calls': '#e74c3c'
                }
            )
            fig4.update_layout(
                plot_bgcolor='#111118',
                paper_bgcolor='#0a0a0f',
                font=dict(color='white'),
                height=300
            )
            st.plotly_chart(fig4, use_container_width=True)

# ── PAGE 3: RETENTION REPORTS ──────────────────────────────────
elif "Retention" in page:
    st.markdown("# AI Retention Reports")
    st.markdown("*Groq LLM generates personalized retention actions*")
    st.markdown("---")

    st.markdown("### Pre-Generated Reports — Top 10 High Risk")

    for _, report in reports.iterrows():
        with st.expander(
            f"{report['customer_id']} — "
            f"{report['plan']} plan | "
            f"₹{report['monthly_revenue']:,}/mo | "
            f"{report['churn_probability']:.1%} risk"
        ):
            st.markdown(f"""
            <div class="report-box">
                {report['retention_action']}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Generate Fresh Retention Action")

    high_risk_ids = risk[
        risk['risk_level'] == 'High'
    ]['customer_id'].tolist()

    selected_customer = st.selectbox(
        "Select high-risk customer",
        high_risk_ids[:50]
    )

    if st.button("Generate Retention Action"):
        with st.spinner("Groq LLM analyzing customer..."):
            try:
                cust_data = risk[
                    risk['customer_id'] == selected_customer
                ].iloc[0]

                cust_signals = signals[
                    signals['customer_id'] == selected_customer
                ].sort_values('week').tail(4)

                signals_text = "\n".join([
                    f"Week {row['week']}: "
                    f"logins={row['login_frequency']}, "
                    f"session={row['session_duration_mins']}min, "
                    f"api={row['api_calls']}, "
                    f"tickets={row['support_tickets']}"
                    for _, row in cust_signals.iterrows()
                ])

                groq_key = st.secrets.get(
                    "GROQ_API_KEY",
                    os.getenv("GROQ_API_KEY")
                ) if hasattr(st, 'secrets') else \
                    os.getenv("GROQ_API_KEY")

                groq_client = Groq(api_key=groq_key)
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": f"""
You are a senior customer success analyst at a SaaS company.

Customer: {selected_customer}
Plan: {cust_data['plan']} (₹{cust_data['monthly_revenue']}/mo)
Churn Risk: {cust_data['churn_probability']:.1%}
NPS: {cust_data['nps_score']}/10
Tenure: {cust_data['total_weeks']} weeks

Recent signals (last 4 weeks):
{signals_text}

Write a retention action report:
1. SITUATION (2 sentences)
2. ROOT CAUSE (1 sentence)
3. ACTION (2 sentences)

Be specific with numbers. Under 120 words."""}],
                    max_tokens=200,
                    temperature=0.3
                )

                fresh_report = response.choices[
                    0
                ].message.content.strip()

                st.markdown(f"""
                <div class="report-box">
                    {fresh_report}
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ── PAGE 4: MODEL INSIGHTS ─────────────────────────────────────
elif "Insights" in page:
    st.markdown("# Model Insights")
    st.markdown("*How XGBoost predicts churn from behavioral signals*")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        shap_cols = [c for c in shap.columns if c.startswith('shap_')]
        mean_shap = shap[shap_cols].abs().mean()
        importance_df = pd.DataFrame({
            'feature': [c.replace('shap_', '') for c in shap_cols],
            'importance': mean_shap.values
        }).sort_values('importance', ascending=True).tail(15)

        fig5 = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale=['#3498db', '#e74c3c'],
            title='Top Features — SHAP Importance'
        )
        fig5.update_layout(
            plot_bgcolor='#111118',
            paper_bgcolor='#0a0a0f',
            font=dict(color='white'),
            height=450,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        fig6 = px.histogram(
            risk,
            x='churn_probability',
            color='risk_level',
            color_discrete_map={
                'High': '#e74c3c',
                'Medium': '#f39c12',
                'Low': '#2ecc71'
            },
            title='Churn Probability Distribution',
            nbins=50
        )
        fig6.update_layout(
            plot_bgcolor='#111118',
            paper_bgcolor='#0a0a0f',
            font=dict(color='white'),
            height=450
        )
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")
    st.markdown("### Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("ROC-AUC", "0.93", "#4fc3f7"),
        ("Precision", "76%", "#2ecc71"),
        ("Recall", "81%", "#f39c12"),
        ("Features", "21", "#4fc3f7"),
    ]

    for col, (label, value, color) in zip(
        [col1, col2, col3, col4], metrics
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">
                    {value}
                </div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)