"""
Script to generate a standardized, realistic air quality monitoring dataset
based on Central Pollution Control Board (CPCB) and US EPA pollutant concentration
profiles and official sub-index breakpoint calculation rules.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Breakpoints for CPCB sub-index calculation
# (C_low, C_high, I_low, I_high)
BREAKPOINTS = {
    'PM2.5': [
        (0, 30, 0, 50),
        (30, 60, 50, 100),
        (60, 90, 100, 200),
        (90, 120, 200, 300),
        (120, 250, 300, 400),
        (250, 500, 400, 500)
    ],
    'PM10': [
        (0, 50, 0, 50),
        (50, 100, 50, 100),
        (100, 250, 100, 200),
        (250, 350, 200, 300),
        (350, 430, 300, 400),
        (430, 600, 400, 500)
    ],
    'NO2': [
        (0, 40, 0, 50),
        (40, 80, 50, 100),
        (80, 180, 100, 200),
        (180, 280, 200, 300),
        (280, 400, 300, 400),
        (400, 600, 400, 500)
    ],
    'SO2': [
        (0, 40, 0, 50),
        (40, 80, 50, 100),
        (80, 380, 100, 200),
        (380, 800, 200, 300),
        (800, 1600, 300, 400),
        (1600, 2000, 400, 500)
    ],
    'CO': [
        (0.0, 1.0, 0, 50),
        (1.0, 2.0, 50, 100),
        (2.0, 10.0, 100, 200),
        (10.0, 17.0, 200, 300),
        (17.0, 34.0, 300, 400),
        (34.0, 50.0, 400, 500)
    ],
    'O3': [
        (0, 50, 0, 50),
        (50, 100, 50, 100),
        (100, 168, 100, 200),
        (168, 208, 200, 300),
        (208, 748, 300, 400),
        (748, 1000, 400, 500)
    ]
}

def calc_sub_index(conc, pollutant):
    for c_low, c_high, i_low, i_high in BREAKPOINTS[pollutant]:
        if c_low <= conc <= c_high:
            return i_low + ((i_high - i_low) / (c_high - c_low)) * (conc - c_low)
    # Beyond max scale
    return 500.0

def generate_data(n_samples=5000, random_seed=42):
    np.random.seed(random_seed)
    
    # Generate realistic correlated air pollutant concentrations
    # PM2.5 and PM10 have high correlation; NO2 and CO correlate with vehicular traffic
    mean = [55.0, 110.0, 45.0, 22.0, 2.8, 48.0]
    
    # Covariance matrix reflecting real-world atmospheric chemistry
    cov = [
        [900.0,  1250.0,  180.0,   45.0,   12.0,   60.0],   # PM2.5
        [1250.0, 2200.0,  240.0,   70.0,   16.0,   85.0],   # PM10
        [180.0,   240.0,  320.0,   55.0,   14.0,  -40.0],   # NO2
        [45.0,     70.0,   55.0,  120.0,    4.0,   10.0],   # SO2
        [12.0,     16.0,   14.0,    4.0,    2.2,   -5.0],   # CO
        [60.0,     85.0,  -40.0,   10.0,   -5.0,  280.0]    # O3
    ]
    
    raw = np.random.multivariate_normal(mean, cov, size=n_samples)
    
    # Ensure physical lower bounds (>0)
    pm25 = np.clip(raw[:, 0], 2.0, 450.0)
    pm10 = np.clip(np.maximum(raw[:, 1], pm25 * 1.2), 5.0, 580.0)  # PM10 >= PM2.5
    no2  = np.clip(raw[:, 2], 3.0, 380.0)
    so2  = np.clip(raw[:, 3], 1.5, 250.0)
    co   = np.clip(raw[:, 4], 0.2, 35.0)
    o3   = np.clip(raw[:, 5], 4.0, 420.0)
    
    df = pd.DataFrame({
        'PM2.5': np.round(pm25, 2),
        'PM10': np.round(pm10, 2),
        'NO2': np.round(no2, 2),
        'SO2': np.round(so2, 2),
        'CO': np.round(co, 2),
        'O3': np.round(o3, 2)
    })
    
    # Calculate official sub-indices
    aqi_list = []
    for idx, row in df.iterrows():
        sub_indices = [
            calc_sub_index(row['PM2.5'], 'PM2.5'),
            calc_sub_index(row['PM10'], 'PM10'),
            calc_sub_index(row['NO2'], 'NO2'),
            calc_sub_index(row['SO2'], 'SO2'),
            calc_sub_index(row['CO'], 'CO'),
            calc_sub_index(row['O3'], 'O3')
        ]
        # True AQI is determined by dominant pollutant with ambient sensor noise
        base_aqi = max(sub_indices)
        # Add realistic atmospheric noise (sensor variance +/- 2.5)
        sensor_noise = np.random.normal(0, 2.5)
        final_aqi = np.clip(base_aqi + sensor_noise, 5.0, 500.0)
        aqi_list.append(round(final_aqi, 2))
        
    df['AQI'] = aqi_list
    
    # Map to qualitative category
    def to_category(val):
        if val <= 50: return 'Good'
        elif val <= 100: return 'Satisfactory'
        elif val <= 200: return 'Moderate'
        elif val <= 300: return 'Poor'
        elif val <= 400: return 'Very Poor'
        else: return 'Severe'
        
    df['AQI_Category'] = df['AQI'].apply(to_category)
    return df

if __name__ == '__main__':
    out_path = Path(__file__).resolve().parents[1] / 'data' / 'aqi_pollutant_dataset.csv'
    df = generate_data(5000, random_seed=42)
    df.to_csv(out_path, index=False)
    print(f"Dataset successfully created at: {out_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nSummary Statistics:")
    print(df.describe().round(2))
    print("\nCategory Distribution:")
    print(df['AQI_Category'].value_counts())
