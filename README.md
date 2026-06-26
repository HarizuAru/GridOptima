# GridOptima: Smart Meter Telemetry Analyzer & Anomaly Engine

GridOptima is a production-ready, interactive web application built with Streamlit for analyzing smart meter telemetry data and detecting anomalies in power consumption. 

## Features
- **Synthetic Data Generation**: Generates 90 days of high-fidelity synthetic data mimicking realistic power consumption behaviors, cooling load dependencies on temperature, and human day/night cycles, along with injected fault scenarios.
- **RP4 Time-of-Use Billing Module**: Implements standard domestic tariff rules (Regulatory Period 4 for Malaysia):
  - Peak Hours (08:00 - 22:00) at 28.52 sen/kWh
  - Off-Peak Hours (22:00 - 08:00) at 24.43 sen/kWh
- **Machine Learning Anomaly Detection**: Uses `IsolationForest` from Scikit-Learn to detect multivariable outliers based on electricity consumption, temperature, humidity, and time of day.
- **Statistical Drift Detection**: Implements a Rolling Z-score anomaly metric to detect statistical drift:
  $Z_t = \frac{(x_t - \mu_{window})}{\sigma_{window}}$
- **Interactive UI**: Key performance indicators (KPI), interactive Plotly timeline charts, and dynamically updated incident logs.

## Getting Started

### 1. Installation
Install the necessary requirements:
```bash
pip install -r requirements.txt
```

### 2. Generate Data
Run the data generator to create the `smart_meter_data.csv` dataset:
```bash
python generate_data.py
```

### 3. Run the App
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

## Architecture
- `generate_data.py`: Synthesizes data incorporating natural weather, time-of-day baselines, and injects anomalies (0 kWh drops, high sustained spikes).
- `app.py`: The main Streamlit dashboard. Incorporates data ingestion, RP4 cost computation, ML processing, and data visualization via Plotly.
