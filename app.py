import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- APPLE-INSPIRED THEME & STYLING ---
st.set_page_config(page_title="Burnout Analytics POC", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: #1d1d1f;
        background-color: #f5f5f7;
    }
    
    .main { background-color: #f5f5f7; }
    
    .stButton>button {
        border-radius: 20px;
        background-color: #0071e3;
        color: white;
        border: none;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #0077ed;
        transform: scale(1.02);
    }
    
    .card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        border-radius: 18px;
        padding: 24px;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }

    .highlight-box {
        background: linear-gradient(135deg, #f5f5f7 0%, #ffffff 100%);
        border-left: 5px solid #0071e3;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
    }

    h1, h2, h3 { font-weight: 600 !important; letter-spacing: -0.02em !important; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL CORE LOGIC ---
@st.cache_data
def load_and_train():
    # Loading the provided train.csv
    df = pd.read_csv('train.csv').dropna()
    
    # Preprocessing as per POC Section 3.1
    df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
    df['Company Type'] = df['Company Type'].map({'Product': 1, 'Service': 0})
    df['WFH Setup Available'] = df['WFH Setup Available'].map({'Yes': 1, 'No': 0})
    
    features = ['Gender', 'Company Type', 'WFH Setup Available', 'Designation', 'Resource Allocation', 'Mental Fatigue Score']
    X = df[features]
    y = df['Burn Rate']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return model, scaler, features

model, scaler, feature_names = load_and_train()

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=30)
    st.title("Employee Input")
    st.markdown("---")
    gen = st.selectbox("Gender", ["Female", "Male"])
    ct = st.selectbox("Company Type", ["Product", "Service"])
    wfh = st.selectbox("WFH Available", ["Yes", "No"])
    des = st.slider("Designation (Seniority)", 0.0, 5.0, 2.0)
    ra = st.slider("Resource Allocation (Hours)", 1.0, 10.0, 5.0)
    mf = st.slider("Mental Fatigue Score", 0.0, 10.0, 5.0)

# --- MAIN DASHBOARD ---
st.title("Employee Burnout Risk Dashboard")
st.caption("A Novel Multi-Modal Framework with Prospective Simulation")

# Predict Current State
input_data = pd.DataFrame([[
    1 if gen == "Female" else 0,
    1 if ct == "Product" else 0,
    1 if wfh == "Yes" else 0,
    des, ra, mf
]], columns=feature_names)

scaled_input = scaler.transform(input_data)
current_burn_rate = model.predict(scaled_input)[0]
current_burn_rate = np.clip(current_burn_rate, 0, 1)

# Risk Stratification Logic (POC Eq. 7)
if current_burn_rate < 0.35:
    risk_tier = "LOW"
    color = "#34c759"
elif current_burn_rate < 0.65:
    risk_tier = "MODERATE"
    color = "#ff9f0a"
else:
    risk_tier = "HIGH"
    color = "#ff3b30"

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"""
        <div class="card" style="text-align: center;">
            <p style="color: #86868b; font-size: 14px; margin-bottom: 5px;">CURRENT RISK LEVEL</p>
            <h1 style="color: {color}; font-size: 48px; margin: 0;">{risk_tier}</h1>
            <p style="font-size: 24px; font-weight: 300;">{current_burn_rate:.2f}</p>
        </div>
    """, unsafe_allow_html=True)

# --- HIGHLIGHTED UNIQUE ELEMENT: INTERVENTION SIMULATOR ---
st.markdown("---")
st.markdown("""
    <div class="highlight-box">
        <h3 style="margin-top:0; color:#0071e3;">✨ Unique Innovation: Intervention Simulation Engine</h3>
        <p>This module applies coefficient-based counterfactual adjustments (Eq. 8) to project 12-month trajectories. 
        Unlike static tools, this simulates the impact of <b>WFH access</b> and <b>Workload Caps</b> before deployment.</p>
    </div>
""", unsafe_allow_html=True)

st.subheader("Prospective Intervention Modeling")
sim_wfh = st.checkbox("Simulate: Enable WFH Access")
sim_ra_reduction = st.slider("Simulate: Workload Reduction (Resource Allocation Units)", 0.0, 3.0, 0.0)

# Simulation Math (Eq. 8)
months = np.arange(0, 13)
# S0: No Intervention Baseline (drifts +0.01/month per POC 9.1)
baseline_trajectory = [np.clip(current_burn_rate + (m * 0.01), 0, 1) for m in months]

# Intervention Trajectory
# β_WFH = -0.0177 (POC 4.5), RA coefficient ≈ 0.07 (standardized)
# For simplicity in this demo, we use the direct impact of reducing RA.
sim_trajectory = []
for m in months:
    # Apply WFH benefit
    reduction = 0.0177 if sim_wfh else 0
    # Apply gradual RA reduction over first 3 months
    ra_effect = (min(m, 3) / 3) * (sim_ra_reduction * 0.04) # 0.04 estimated impact
    val = current_burn_rate - reduction - ra_effect + (m * 0.005) # Slower drift with intervention
    sim_trajectory.append(np.clip(val, 0, 1))

fig = go.Figure()
fig.add_trace(go.Scatter(x=months, y=baseline_trajectory, name="S0: No Intervention", line=dict(color='#ff3b30', dash='dot')))
fig.add_trace(go.Scatter(x=months, y=sim_trajectory, name="S1: With Selected Intervention", fill='tozeroy', line=dict(color='#0071e3', width=4)))

# Add Threshold Lines
fig.add_hline(y=0.65, line_dash="dash", line_color="#ff9f0a", annotation_text="High Risk Threshold")

fig.update_layout(
    title="12-Month Burn Rate Projection",
    xaxis_title="Months From Today",
    yaxis_title="Predicted Burn Rate Score",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
    <div style="color: #86868b; font-size: 12px; text-align: center; margin-top: 40px;">
        © 2026 Employee Burnout Prediction & Risk Classification System | Based on POC v2.0
    </div>
""", unsafe_allow_html=True)
