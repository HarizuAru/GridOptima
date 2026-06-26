import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import os
import warnings
warnings.filterwarnings('ignore')

# Set page configuration for modern UI
st.set_page_config(
    page_title="GridOptima | Anomaly Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .kpi-card {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .kpi-title {
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-bottom: 10px;
    }
    .kpi-value {
        color: #F7FAFC;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Processing ---
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smart_meter_data.csv')
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Calculate Is_Peak (Peak Hours: 08:00 to 21:59 mapped as >= 8 and < 22)
    hours = df['Timestamp'].dt.hour
    df['Is_Peak'] = ((hours >= 8) & (hours < 22)).astype(int)
    
    # Calculate RP4 Cost (Sen to MYR conversion)
    # Peak: 28.52 sen/kWh, Off-Peak: 24.43 sen/kWh
    peak_rate_rm = 28.52 / 100.0
    off_peak_rate_rm = 24.43 / 100.0
    
    df['Cost (RM)'] = np.where(
        df['Is_Peak'] == 1,
        df['Electricity Consumed (kWh)'] * peak_rate_rm,
        df['Electricity Consumed (kWh)'] * off_peak_rate_rm
    )
    
    return df

# --- Sidebar Configuration ---
st.sidebar.title("⚡ GridOptima Engine")
st.sidebar.markdown("Configure ML Anomaly Parameters")

contamination_rate = st.sidebar.slider(
    "Isolation Forest Contamination",
    min_value=0.001,
    max_value=0.05,
    value=0.01,
    step=0.001,
    help="Proportion of outliers in the dataset."
)

z_score_window = st.sidebar.slider(
    "Z-Score Rolling Window (Periods)",
    min_value=12,
    max_value=96,
    value=48, # 24 hours at 30 min intervals
    step=12,
    help="Window size for rolling Z-score calculation."
)

# --- Main App ---
st.title("GridOptima: Smart Meter Telemetry Analyzer")
st.markdown("Advanced analytics dashboard for Grid RP4 operations and multivariable anomaly detection.")

df = load_data()

if df is None:
    st.error("Data file not found. Please run `python generate_data.py` first to generate `smart_meter_data.csv`.")
    st.stop()

# --- ML & Statistical Anomaly Engines ---
# 1. Isolation Forest
features = ['Electricity Consumed (kWh)', 'Temperature (°C)', 'Humidity (%)', 'Is_Peak']
X = df[features]

iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
df['IF_Anomaly'] = iso_forest.fit_predict(X)
# IsolationForest returns -1 for anomaly, 1 for normal
df['Is_IF_Anomaly'] = (df['IF_Anomaly'] == -1).astype(int)

# 2. Rolling Z-Score (Statistical Drift)
# Z_t = (x_t - mean_window) / std_window
rolling_mean = df['Electricity Consumed (kWh)'].rolling(window=z_score_window, min_periods=1).mean()
rolling_std = df['Electricity Consumed (kWh)'].rolling(window=z_score_window, min_periods=1).std()

# Handle very first record where std is NaN
rolling_std = rolling_std.bfill()

# Calculate Z-Score safely
df['Z_Score'] = np.where(
    rolling_std > 0, 
    (df['Electricity Consumed (kWh)'] - rolling_mean) / rolling_std, 
    0
)

# For visualization and KPI, we consider Isolation Forest output as the primary anomaly metric
df['Is_Anomaly'] = df['Is_IF_Anomaly'] 

# --- KPI Calculations ---
total_energy = df['Electricity Consumed (kWh)'].sum()
total_cost = df['Cost (RM)'].sum()
total_anomalies = df['Is_Anomaly'].sum()
anomaly_rate = (total_anomalies / len(df)) * 100

# --- Render KPIs ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Energy (kWh)</div>
        <div class="kpi-value">{total_energy:,.1f}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total RP4 Cost (RM)</div>
        <div class="kpi-value">RM {total_cost:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Malfunctions Flagged</div>
        <div class="kpi-value">{total_anomalies}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Active Anomaly Rate (%)</div>
        <div class="kpi-value">{anomaly_rate:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# --- Visualization: Plotly Chart ---
st.subheader("Telemetry Timeline & Anomaly Detection")

fig = go.Figure()

# Baseline trace (Blue)
fig.add_trace(go.Scatter(
    x=df['Timestamp'],
    y=df['Electricity Consumed (kWh)'],
    mode='lines',
    name='Consumption (kWh)',
    line=dict(color='#3182ce', width=1.5),
    opacity=0.8
))

# Anomaly markers (Red)
anomalies_df = df[df['Is_Anomaly'] == 1]
fig.add_trace(go.Scatter(
    x=anomalies_df['Timestamp'],
    y=anomalies_df['Electricity Consumed (kWh)'],
    mode='markers',
    name='Anomaly Flag',
    marker=dict(color='#e53e3e', size=8, symbol='circle-open', line=dict(width=2))
))

fig.update_layout(
    plot_bgcolor='rgba(15, 17, 26, 1)',
    paper_bgcolor='rgba(15, 17, 26, 1)',
    font=dict(color='#A0AEC0'),
    xaxis=dict(showgrid=True, gridcolor='#2D3748', title='Time'),
    yaxis=dict(showgrid=True, gridcolor='#2D3748', title='Electricity (kWh)'),
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- Incident Log ---
st.subheader("Incident Log")
st.markdown("Detailed log of flagged anomalous events including local meteorological factors.")

# Filter and format for display
incident_df = anomalies_df[[
    'Timestamp', 
    'Electricity Consumed (kWh)', 
    'Temperature (°C)', 
    'Humidity (%)', 
    'Wind Speed (km/h)', 
    'Cost (RM)', 
    'Z_Score'
]].copy()

# Sort by most recent
incident_df = incident_df.sort_values('Timestamp', ascending=False).reset_index(drop=True)

# Format numerical columns
incident_df['Electricity Consumed (kWh)'] = incident_df['Electricity Consumed (kWh)'].round(3)
incident_df['Temperature (°C)'] = incident_df['Temperature (°C)'].round(1)
incident_df['Humidity (%)'] = incident_df['Humidity (%)'].round(1)
incident_df['Wind Speed (km/h)'] = incident_df['Wind Speed (km/h)'].round(1)
incident_df['Cost (RM)'] = incident_df['Cost (RM)'].round(2)
incident_df['Z_Score'] = incident_df['Z_Score'].round(2)

st.dataframe(
    incident_df,
    use_container_width=True,
    height=400
)
