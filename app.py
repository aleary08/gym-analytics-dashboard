import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data import generate_gym_data
from model import train_churn_model, predict_churn

st.set_page_config(
    page_title="Gym Analytics Dashboard",
    page_icon="🥊",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #111318; }
    .metric-card {
        background-color: #1a1d24;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2a2d35;
    }
    .stMetric { color: white; }
    h1, h2, h3 { color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return generate_gym_data()

@st.cache_resource
def load_model():
    return train_churn_model()

st.title("Gym Analytics Dashboard")
st.markdown("**Powered by Anthony Leary, Combat Sports Tech Specialist**")
st.divider()

with st.spinner("Loading data and training model..."):
    df = load_data()
    model, scaler, features = load_model()
    df = predict_churn(df, model, scaler, features)

# KPI Metrics Row
active_members = len(df[df["churned"] == 0])
monthly_revenue = df[df["churned"] == 0]["monthly_fee"].sum()
high_risk = len(df[df["risk_level"] == "High"])
avg_sessions = df["avg_sessions_per_week"].mean()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Active Members", active_members, delta="+12 this month")
with col2:
    st.metric("Monthly Revenue", f"${monthly_revenue:,}", delta="+$2,100")
with col3:
    st.metric("High Risk Members", high_risk, delta="-3 from last month")
with col4:
    st.metric("Avg Sessions/Week", f"{avg_sessions:.1f}", delta="+0.3")

st.divider()

# Charts Row
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Members by Martial Art")
    art_counts = df[df["churned"]==0]["martial_art"].value_counts().reset_index()
    art_counts.columns = ["Martial Art", "Members"]
    fig = px.bar(
        art_counts, x="Martial Art", y="Members",
        color="Members",
        color_continuous_scale=["#E63946", "#c1121f"],
        template="plotly_dark"
    )
    fig.update_layout(
        plot_bgcolor="#1a1d24",
        paper_bgcolor="#1a1d24",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("⚠️ Churn Risk Distribution")
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    colors = {"High": "#E63946", "Medium": "#f4a261", "Low": "#2a9d8f"}
    fig2 = px.pie(
        risk_counts, values="Count", names="Risk Level",
        color="Risk Level",
        color_discrete_map=colors,
        template="plotly_dark"
    )
    fig2.update_layout(
        plot_bgcolor="#1a1d24",
        paper_bgcolor="#1a1d24"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# Churn Risk Table
st.subheader("🚨 At-Risk Members, Immediate Action Required")

risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])

display_df = df[["name", "martial_art", "membership_months", 
                  "days_since_last_visit", "failed_payments", 
                  "churn_probability", "risk_level"]].copy()

display_df.columns = ["Name", "Martial Art", "Months as Member", 
                       "Days Since Last Visit", "Failed Payments",
                       "Churn Risk %", "Risk Level"]

if risk_filter != "All":
    display_df = display_df[display_df["Risk Level"] == risk_filter]

display_df = display_df.sort_values("Churn Risk %", ascending=False)

def color_risk(val):
    if val == "High":
        return "background-color: #E6394633; color: #E63946"
    elif val == "Medium":
        return "background-color: #f4a26133; color: #f4a261"
    return "background-color: #2a9d8f33; color: #2a9d8f"

styled_df = display_df.style.map(color_risk, subset=["Risk Level"])
st.dataframe(styled_df, use_container_width=True, height=400)

st.divider()

# Attendance Trend
st.subheader("📈 Attendance vs Churn Risk")
fig3 = px.scatter(
    df, 
    x="days_since_last_visit", 
    y="churn_probability",
    color="risk_level",
    color_discrete_map={"High": "#E63946", "Medium": "#f4a261", "Low": "#2a9d8f"},
    hover_data=["name", "martial_art"],
    labels={
        "days_since_last_visit": "Days Since Last Visit",
        "churn_probability": "Churn Risk %"
    },
    template="plotly_dark"
)
fig3.update_layout(
    plot_bgcolor="#1a1d24",
    paper_bgcolor="#1a1d24"
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.markdown("*Dashboard built by Anthony Leary — anthonyleary.dev | Combat Sports Tech Specialist*")