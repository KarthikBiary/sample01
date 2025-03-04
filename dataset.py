import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


NUM_DAYS = 30  
SAMPLES_PER_MINUTE = 1  
TOTAL_SAMPLES = NUM_DAYS * 24 * 60 * SAMPLES_PER_MINUTE


start_time = datetime(2023, 1, 1, 0, 0)

def generate_buggy_data(num_samples):
    timestamps = pd.date_range(start=start_time, periods=num_samples, freq='T') 
    df = pd.DataFrame(index=timestamps)


    df['battery_voltage'] = np.sin(np.linspace(0, 20*np.pi, num_samples)) * 0.5 + 12.2  
    df['battery_voltage'] -= np.linspace(0, 0.05, num_samples)  


    df['speed'] = np.abs(np.random.normal(0, 25, num_samples)).cumsum() % 60  
    df['engine_temp'] = 20 + (df['speed'] * 0.7) + np.random.normal(0, 3, num_samples)


    df['tire_pressure'] = 32 - (np.linspace(0, 0.001, num_samples) * np.arange(num_samples))

    df['vibration'] = np.random.exponential(1, num_samples)
    df['vibration'] += np.abs(np.random.normal(0, 0.5, num_samples))

    df['brake_wear'] = np.linspace(0, 100, num_samples) + np.random.normal(0, 2, num_samples)

    df['suspension_shocks'] = np.random.poisson(0.1, num_samples)



    battery_failure_starts = random.sample(range(1000, num_samples-100), 5)
    for start in battery_failure_starts:
        df['battery_voltage'].iloc[start:start+50] *= 0.6  
    overheat_events = random.sample(range(500, num_samples-50), 8)
    for event in overheat_events:
        df['engine_temp'].iloc[event:event+30] += np.linspace(20, 50, 30)


    df.loc[df.index.day.isin([5,15,25]), 'tire_pressure'] *= 0.8 


    df.loc[df['brake_wear'] > 85, 'brake_wear'] += np.random.uniform(0, 5)

    df['vibration'] += np.random.choice([0, 10], num_samples, p=[0.999, 0.001])


    replacement_days = [7, 21]
    for day in replacement_days:
        mask = (df.index.date == (start_time + timedelta(days=day)).date())
        df.loc[mask, 'battery_voltage'] = 13.8  

    df.loc[df.index.hour == 0, 'tire_pressure'] = 32  


    for col in df.columns:
        df[col] += np.random.normal(0, df[col].std()/50, num_samples)


    df['battery_voltage'] = df['battery_voltage'].clip(10.5, 14.5)
    df['engine_temp'] = df['engine_temp'].clip(20, 130)
    df['tire_pressure'] = df['tire_pressure'].clip(25, 35)
    df['vibration'] = df['vibration'].clip(0, 15)


    df['failure'] = 0
    df.loc[(df['battery_voltage'] < 11) |
           (df['engine_temp'] > 120) |
           (df['tire_pressure'] < 26) |
           (df['vibration'] > 12), 'failure'] = 1

    return df.reset_index().rename(columns={'index':'timestamp'})


buggy_data = generate_buggy_data(TOTAL_SAMPLES)

buggy_data.to_csv('buggy_sensor_data.csv', index=False)
print(f"Generated dataset with {len(buggy_data)} records")
print("Columns:", buggy_data.columns.tolist())