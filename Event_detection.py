# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 02:19:03 2025

@author: gib445
"""
import os
import numpy as np
from scipy.signal import find_peaks

# === Set Path ===
output_directory = r'C:\Data\ProcessedNumpy'

# === Load Full Dataset ===
X = np.load(os.path.join(output_directory, 'full_X.npy'))
Y = np.load(os.path.join(output_directory, 'full_Y.npy'))
labels = np.load(os.path.join(output_directory, 'full_labels.npy'), allow_pickle=True)

datasets = [Y]
dataset_names = ["full"]

# === Column Indices (Update accordingly) ===
column_index_toe_left = "toe off left column"
column_index_heel_left = "heel strike left column"
column_index_toe_right = "toe off right column"
column_index_heel_right = "heel strike right column"

# === Sampling Rate ===
Hz = 100  # Replace with your actual rate

default_params = {
    'Desired_Hz': Hz,
    'window_dimension': int(Hz * 5),
    'min_distance': int(Hz * 0.45),
    'toe_off_percentile': 80,
    'heel_strike_percentile': 70,
    'toe_off_percentage_local_max': 0.3,
    'heel_strike_percentage_local_max': 0.3,
    'toe_off_window_min_percentage': 0,
    'toe_off_window_max_percentage': 0.8,
    'heel_strike_window_min_percentage': 0,
    'heel_strike_window_max_percentage': 0.4
}

# === Detection Functions ===
def calculate_toe_off(data, params):
    cutoff = np.percentile(data, params['toe_off_percentile'])
    peaks, _ = find_peaks(data, height=cutoff, distance=params['min_distance'])

    valid_maxima = []
    for idx in peaks:
        start = max(0, idx - params['window_dimension'] // 2)
        end = min(len(data), idx + params['window_dimension'] // 2)
        if data[idx] >= params['toe_off_percentage_local_max'] * np.max(data[start:end]):
            valid_maxima.append(idx)

    filtered_indices = []
    for i in range(len(valid_maxima) - 1):
        current = valid_maxima[i]
        next_peak = valid_maxima[i + 1]
        win_min = int(params['toe_off_window_min_percentage'] * (next_peak - current))
        win_max = int(params['toe_off_window_max_percentage'] * (next_peak - current))
        start = current
        end = current + win_max
        min_idx = np.argmin(data[start:end]) + start
        if start + win_min <= min_idx < end:
            filtered_indices.append(min_idx)

    labels = np.zeros(len(data), dtype=int)
    labels[filtered_indices] = 1
    return labels

def calculate_heel_strike(data, params):
    cutoff = np.percentile(data, params['heel_strike_percentile'])
    peaks, _ = find_peaks(data, height=cutoff, distance=params['min_distance'])

    valid_maxima = []
    for idx in peaks:
        start = max(0, idx - params['window_dimension'] // 2)
        end = min(len(data), idx + params['window_dimension'] // 2)
        if data[idx] >= params['heel_strike_percentage_local_max'] * np.max(data[start:end]):
            valid_maxima.append(idx)

    filtered_indices = []
    for i in range(len(valid_maxima) - 1):
        current = valid_maxima[i]
        next_peak = valid_maxima[i + 1]
        win_min = int(params['heel_strike_window_min_percentage'] * (next_peak - current))
        win_max = int(params['heel_strike_window_max_percentage'] * (next_peak - current))
        start = current
        end = current + win_max
        min_idx = np.argmin(data[start:end]) + start
        if start + win_min <= min_idx < end:
            filtered_indices.append(min_idx)

    labels = np.zeros(len(data), dtype=int)
    labels[filtered_indices] = 1
    return labels

# === Processing ===
def process_and_save(datasets, names, toe_idx, heel_idx, side):
    for data, name in zip(datasets, names):
        toe_labels = calculate_toe_off(data[:, toe_idx], default_params)
        heel_labels = calculate_heel_strike(data[:, heel_idx], default_params)

        toe_path = os.path.join(output_directory, f"{side}_toe_labels_{name}.npy")
        heel_path = os.path.join(output_directory, f"{side}_heel_labels_{name}.npy")

        np.save(toe_path, toe_labels)
        np.save(heel_path, heel_labels)

        print(f"💾 Saved: {toe_path}, {heel_path}")

# === Run ===
process_and_save(datasets, dataset_names, column_index_toe_left, column_index_heel_left, side="left")
process_and_save(datasets, dataset_names, column_index_toe_right, column_index_heel_right, side="right")

print("✅ Heel strike and toe-off label generation complete.")
