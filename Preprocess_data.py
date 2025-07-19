# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 02:17:07 2025

@author: gib445
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler

# === Configuration ===
data_dir = r'C:\Data\3D\all_data'
traj_dir = r'C:\Data\Trajectories\all_data'
output_directory = r'C:\Data\ProcessedNumpy'
os.makedirs(output_directory, exist_ok=True)

Hz = "Original sampling rate"         # Replace with numeric value
Desired_Hz = "Original sampling rate" # Replace with numeric value

cutoff_frequency = 10
filter_order = 4

# === Utility Functions ===
def extract_participant_id(filename):
    start = filename.rfind("ID") + 2
    end = filename.find("_", start)
    return filename[start:end] if start >= 2 and end > start else "Unknown"

def interp_array(data, length):
    return np.stack([
        interp1d(np.arange(data.shape[0]), data[:, i], kind='linear', fill_value="extrapolate")(np.linspace(0, data.shape[0] - 1, length))
        for i in range(data.shape[1])
    ], axis=-1)

def resample(data, orig_hz, target_hz):
    length = int(data.shape[0] * (target_hz / orig_hz))
    return interp_array(data, length)

def fill_nan_inf(data):
    df = pd.DataFrame(data)
    return df.ffill().bfill().to_numpy()

def butter_filter(data, cutoff, fs, order):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype='low')
    return np.stack([filtfilt(b, a, data[:, i]) for i in range(data.shape[1])], axis=-1)

# === Load & Process All Files by ID ===
start = time.time()

scaler = RobustScaler()
data_all, traj_all, labels_all = [], [], []

for root, _, files in os.walk(data_dir):
    for file in files:
        if not file.endswith(".xlsx"):
            continue

        data_path = os.path.join(root, file)
        traj_path = os.path.join(traj_dir, file)
        if not os.path.exists(traj_path):
            continue

        try:
            data = fill_nan_inf(pd.read_excel(data_path).to_numpy())
            traj = fill_nan_inf(pd.read_excel(traj_path).to_numpy())
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
            continue

        pid = extract_participant_id(file)
        if pid == "Unknown":
            print(f"⚠️ Skipping unrecognized file: {file}")
            continue

        # Resample and filter
        data_resampled = resample(data, Hz, Desired_Hz)
        traj_scaled = scaler.fit_transform(traj)
        traj_resampled = interp_array(traj_scaled, len(data_resampled))

        data_filtered = butter_filter(data_resampled, cutoff_frequency, Desired_Hz, filter_order)
        traj_filtered = butter_filter(traj_resampled, cutoff_frequency, Desired_Hz, filter_order)

        condition = 0 if "C1" in file else 1 if "C2" in file else 2
        speed = 0 if "3.5" in file else 1 if "4.5" in file else 2
        label_block = [(condition, speed, pid)] * len(data_filtered)

        data_all.append(data_filtered)
        traj_all.append(traj_filtered)
        labels_all.extend(label_block)

        print(f"✅ Processed: {file} | PID: {pid} | Length: {len(data_filtered)}")

# === Final Dataset Assembly ===
data_all = np.concatenate(data_all)
traj_all = np.concatenate(traj_all)
labels_all = np.array(labels_all)

print(f"\n✅ Processing complete in {time.time() - start:.2f} seconds.")
print(f"Shape - Data: {data_all.shape}, Traj: {traj_all.shape}, Labels: {labels_all.shape}")

# === Save to .npy ===
np.save(os.path.join(output_directory, 'full_X.npy'), data_all)
np.save(os.path.join(output_directory, 'full_Y.npy'), traj_all)
np.save(os.path.join(output_directory, 'full_labels.npy'), labels_all)

print(f"💾 Saved to: {output_directory}")

# === Sanity Check ===
def check_stats(name, array):
    print(f"{name}: NaN={np.isnan(array).any()}, Inf={np.isinf(array).any()}, Zero={np.any(array == 0)}")

check_stats("X", data_all)
check_stats("Y", traj_all)
