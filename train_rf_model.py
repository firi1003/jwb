import pandas as pd
import numpy as np
import torch
from semiconductor_autoencoder import load_sensor_dataset, make_dataloaders, fit_autoencoder

# Load RF sensor data
data_path = "rf_sensor_normal_1000.csv"
columns = ["forward_power", "reflected_power", "rf_freq", "rf_temp", "match_imp", "match_volt", "match_curr", "match_temp"]
norm_df = load_sensor_dataset(data_path, columns)

# Create dataloaders
scaler, train_loader, val_loader, _ = make_dataloaders(norm_df, test_size=0.2, val_size=0.1, batch_size=128)

# Train the model
input_dim = 8
model, _ = fit_autoencoder(train_loader, val_loader, input_dim, lr=1e-3, epochs=50)

# Save the model
torch.save(model.state_dict(), "models/rf_semiconductor_autoencoder.pth")
print("RF model trained and saved.")