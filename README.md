# GridOptima: Smart Meter Telemetry Analyzer & Anomaly Engine

GridOptima is an industrial-grade portfolio prototype designed to address the inefficiencies of traditional manual utility monitoring systems and improve grid-level demand-side management. The application features a real-time Advanced Metering Infrastructure (AMI) data ingestion simulation, an automated dynamic tariff optimization engine, and unsupervised machine learning algorithms to detect operational anomalies, grid leaks, and physical meter tampering.

## The Real-World Challenge
Modern utility providers face significant challenges in managing peak electricity grid loads and mitigating resource-side infrastructure losses. Legacy manual reading infrastructure lacks the granularity needed to identify localized failures or coordinate load-flattening incentives, such as dynamic billing tariffs. Additionally, standard monitoring pipelines struggle to distinguish natural, weather-induced surges (like air-conditioning usage on hot days) from actual system malfunctions, leading to high false-alarm rates and missed infrastructure faults.

## The GridOptima Solution
GridOptima addresses these structural challenges through a three-layer analytical approach:

* **Dynamic RP4 Tariff Ingestion Engine**: Dynamically segments and parses 30-minute logging intervals into Peak and Off-Peak consumption periods, calculating usage costs according to national regulatory guidelines.
* **Unsupervised Anomaly Isolation**: Uses an Isolation Forest machine learning model to isolate multi-dimensional anomalies (e.g., sudden load drops or sustained energy leaks) relative to external environmental conditions.
* **Statistical Verification Layer**: Mitigates false positives by calculating a continuous rolling Z-score of real-time electricity usage against historical variance.

## System Architecture

```text
                                  [Meteorological Sensors]
                                             │
                                             ▼
[Consumer Smart Meter] ───(30-Min Intervals)───► [GridOptima Ingestion Pipeline]
                                             │
               ┌─────────────────────────────┴──────────────────────────────┐
               ▼                                                            ▼
[Isolation Forest ML Engine]                                  [RP4 Dynamic Tariff Parser]
(Flags Multi-Dimensional Anomalies)                           (Time-of-Use Cost Calculations)
               │                                                            │
               └─────────────────────────────┬──────────────────────────────┘
                                             ▼
                                [Interactive Dashboard UI]
                            (Plotly Time-Series & Incident Log)
```

### 1. Malaysia Regulatory Period 4 (RP4) Tariff Engine
GridOptima calculates electricity billing costs in accordance with the Malaysian Energy Commission's approved Incentive-Based Regulation (IBR) framework for **Regulatory Period 4 (2025–2027)**. Designed for domestic consumers consuming less than 1,500 kWh/month, usage is parsed according to a Time-of-Use (ToU) model:

| Tariff Period | Time Window | Domestic Rate (sen/kWh) |
| :--- | :--- | :--- |
| **Peak Hours** | 08:00 - 22:00 | 28.52 |
| **Off-Peak Hours** | 22:00 - 08:00 | 24.43 |

### 2. Statistical Rolling Z-score Normalization
To prevent false-positive alarms caused by weather-driven consumption shifts, GridOptima calculates a statistical rolling Z-score. This algorithm evaluates real-time telemetry fluctuations against a sliding historical window:

$$Z_t = \frac{x_t - \mu_{(t, w)}}{\sigma_{(t, w)}}$$

Where:
* $x_t$ represents the active consumption telemetry during the 30-minute interval $t$.
* $\mu_{(t, w)}$ represents the rolling mean of consumption over a sliding window $w$.
* $\sigma_{(t, w)}$ represents the rolling standard deviation over the same sliding window $w$.

This mathematical check allows the engine to verify if a telemetry spike is a statistically significant drift (indicative of a leak/tamper) or a standard variation aligned with ambient temperature changes.

---

## Tech Stack & Data Sources

* **Language**: Python 3.11+
* **Framework**: Streamlit (Dashboard UI and Sidebar Controllers)
* **Visualizations**: Plotly (Interactive multi-trace timeline plots)
* **Machine Learning**: Scikit-Learn (Isolation Forest Unsupervised Model)
* **Data Processing**: Pandas & NumPy (Timestamp conversion and time-series aggregation)
* **Reference Dataset**: Kaggle Smart Meter Electricity Consumption Dataset (30-minute resolution records enriched with meteorological parameters)

---

## Directory Structure

```text
GridOptima/
├── .gitignore
├── README.md
├── app.py
├── generate_data.py
├── requirements.txt
└── smart_meter_data.csv
```

## Getting Started: Local Installation

### 1. Clone the Repository & Initialize Environment
```bash
git clone https://github.com/HarizuAru/GridOptima.git
cd GridOptima

# Initialize virtual environment
python -m venv env

# Activate environment (Windows)
env\Scripts\activate

# Activate environment (macOS/Linux)
source env/bin/activate
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Simulated Production Telemetry
Before running the main dashboard application, execute the data generator script to synthesize 90 days of high-fidelity, 30-minute interval smart meter logs (including injected physical faults and power surge scenarios):
```bash
python generate_data.py
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```
Your local server will boot up automatically at `http://localhost:8501`.

## Deploying to Cloud Production (100% Free)
GridOptima is configured to deploy directly to the web using the Streamlit Community Cloud:

**Push to GitHub:**
Ensure you commit all codebase files (`app.py`, `generate_data.py`, `requirements.txt`, and your generated data `smart_meter_data.csv`).

```bash
git add .
git commit -m "feat: complete GridOptima production build"
git push origin main
```

**Launch Streamlit Community Cloud:**
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in using your GitHub account.
2. Click **New App**.
3. Specify your repository path, branch name as `main`, and main file path as `app.py`.
4. Click **Deploy!** Your system will be online at a custom `*.streamlit.app` URL in under 2 minutes.

## License
Distributed under the MIT License. See `LICENSE` for more details.
