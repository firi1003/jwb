import numpy as np
import pandas as pd

# Generate normal RF sensor data
np.random.seed(42)
n_samples = 1000

# RF Generator: forward_power (W), reflected_power (W), frequency (MHz), temperature (°C)
forward_power_normal = np.random.normal(100, 5, n_samples)  # Normal forward power around 100W
reflected_power_normal = np.random.normal(5, 1, n_samples)  # Low reflected power, normal <10W
rf_freq_normal = np.random.normal(13.56, 0.1, n_samples)  # 13.56 MHz for RF
rf_temp_normal = np.random.normal(50, 2, n_samples)  # Normal temp 50°C

# Matching Network: impedance (Ω), voltage (V), current (A), temperature (°C)
imp_normal = np.random.normal(50, 1, n_samples)  # 50 ohms
volt_normal = np.random.normal(200, 10, n_samples)  # 200V
curr_normal = np.random.normal(2, 0.1, n_samples)  # 2A
match_temp_normal = np.random.normal(45, 2, n_samples)  # 45°C

# Combine into dataframe
normal_data = pd.DataFrame({
    'forward_power': forward_power_normal,
    'reflected_power': reflected_power_normal,
    'rf_freq': rf_freq_normal,
    'rf_temp': rf_temp_normal,
    'match_imp': imp_normal,
    'match_volt': volt_normal,
    'match_curr': curr_normal,
    'match_temp': match_temp_normal
})

normal_data.to_csv('rf_sensor_normal_1000.csv', index=False)

# Generate anomaly data (add some anomalies)
anomaly_data = normal_data.copy()
anomaly_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)  # 10% anomalies

# Introduce anomalies
anomaly_data.loc[anomaly_indices, 'forward_power'] += np.random.normal(0, 20, len(anomaly_indices))  # Power spikes
anomaly_data.loc[anomaly_indices, 'reflected_power'] += np.random.normal(0, 10, len(anomaly_indices))  # High reflected power
anomaly_data.loc[anomaly_indices, 'rf_freq'] += np.random.normal(0, 0.5, len(anomaly_indices))  # Freq drift
anomaly_data.loc[anomaly_indices, 'match_imp'] += np.random.normal(0, 5, len(anomaly_indices))  # Impedance mismatch

# Add label column
anomaly_data['label'] = 0
anomaly_data.loc[anomaly_indices, 'label'] = 1

anomaly_data.to_csv('rf_sensor_anomaly_1000.csv', index=False)

print("RF sensor data generated.")