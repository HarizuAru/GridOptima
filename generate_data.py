import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(start_date, days=90, interval_minutes=30):
    np.random.seed(42)  # For reproducibility
    
    # Calculate number of periods
    periods = (days * 24 * 60) // interval_minutes
    
    # Generate timestamps
    timestamps = pd.date_range(start=start_date, periods=periods, freq=f'{interval_minutes}min')
    
    # Initialize dataframe
    df = pd.DataFrame({'Timestamp': timestamps})
    
    # Generate Meteorological Data
    # Temperature: Daily cycle with random noise (e.g., Malaysia typical 24°C to 34°C)
    hours = df['Timestamp'].dt.hour
    # Sinusoidal temperature curve peaking around 2 PM (14:00)
    df['Temperature (°C)'] = 28 + 5 * np.sin((hours - 8) * (2 * np.pi / 24)) + np.random.normal(0, 0.5, periods)
    
    # Humidity: Inverse relationship with temperature
    df['Humidity (%)'] = 90 - 2 * (df['Temperature (°C)'] - 24) + np.random.normal(0, 2, periods)
    df['Humidity (%)'] = df['Humidity (%)'].clip(40, 100)
    
    # Wind Speed: Random fluctuations (Rayleigh-like distribution)
    df['Wind Speed (km/h)'] = np.abs(np.random.normal(10, 5, periods))
    
    # Baseline Electricity Consumption
    # Day cycle: higher during day/evening (active hours), lower at night
    day_cycle = np.where((hours >= 8) & (hours <= 22), 1.2, 0.4) 
    
    # AC Cooling load: increases when temperature is high (e.g., above 26°C)
    cooling_load = np.maximum(0, df['Temperature (°C)'] - 26) * 0.4
    
    # Add random noise for a realistic baseline
    base_consumption = day_cycle + cooling_load + np.random.normal(0, 0.15, periods)
    df['Electricity Consumed (kWh)'] = np.maximum(0.1, base_consumption)  # Ensure non-negative base
    
    # Inject Anomalies
    num_drops = int(periods * 0.005) # 0.5% drops
    num_spikes = int(periods * 0.002) # 0.2% spikes
    
    # 1. Sudden drops to 0 (meter faults / physical tamper)
    drop_indices = np.random.choice(periods, num_drops, replace=False)
    df.loc[drop_indices, 'Electricity Consumed (kWh)'] = 0.0
    
    # 2. Sustained high-power consumption spikes (localized leakage / malfunction)
    spike_starts = np.random.choice(periods - 6, num_spikes, replace=False) # max 3 hours sustained
    for start in spike_starts:
        duration = np.random.randint(2, 7) # 1 to 3 hours (2 to 6 periods)
        df.loc[start:start+duration, 'Electricity Consumed (kWh)'] += np.random.uniform(4.0, 8.0, size=duration+1)
        
    return df

if __name__ == "__main__":
    print("Generating high-fidelity synthetic smart meter data...")
    # 90 days ending roughly today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Ensure start date is at midnight for clean 90-day intervals
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    df = generate_synthetic_data(start_date=start_date)
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smart_meter_data.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Dataset successfully generated and saved to '{output_file}'.")
    print(f"Total Rows: {len(df)}")
    print(f"Date Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print("\nSample Data:")
    print(df.head())
