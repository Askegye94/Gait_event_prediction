# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 02:56:28 2025

@author: gib445
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
from scipy.signal import find_peaks

# === Config ===
output_directory = r'C:\Data\ProcessedNumpy'
window_size = 100
overlap = 50
stride = window_size - overlap
smoothing_window = 5
Hz = 100
min_peak_distance = int(0.45 * Hz)

# === Load all folds ===
all_probs, all_labels = [], []
for fold in range(1, 6):  # Assuming 5 folds
    path = os.path.join(output_directory, f"cnn_lstm_probs_fold{fold}.npz")
    with np.load(path) as data:
        all_probs.append(data['probs'][:, 1])  # Take probability of class 1
        all_labels.append(data['labels'][:, 1])  # True label for class 1

probs_windowed = np.concatenate(all_probs)
labels_windowed = np.concatenate(all_labels)

# === Smoothing function ===
def apply_centered_moving_average(data, window_size):
    if window_size <= 0 or window_size >= len(data):
        raise ValueError("Invalid window size")
    smoothed = np.empty_like(data)
    half = window_size // 2
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        smoothed[i] = np.mean(data[start:end])
    return smoothed

# === Smooth predictions ===
probs_smoothed = apply_centered_moving_average(probs_windowed, smoothing_window)

# === Unwindow to frame-level (upsample to match full_Y) ===
def unwindow_predictions(probs, total_frames, window_size, overlap):
    stride = window_size - overlap
    full_probs = np.zeros(total_frames)
    counts = np.zeros(total_frames)
    for i, p in enumerate(probs):
        start = i * stride
        end = start + window_size
        full_probs[start:end] += p
        counts[start:end] += 1
    counts[counts == 0] = 1
    return full_probs / counts

# Load total frame count from original signal
original_length = np.load(os.path.join(output_directory, 'full_Y.npy')).shape[0]
unwindowed_probs = unwindow_predictions(probs_smoothed, original_length, window_size, overlap)

# === Peak detection ===
predicted_peaks, _ = find_peaks(unwindowed_probs, distance=min_peak_distance)

# === Create binary predictions ===
predicted_binary = np.zeros_like(unwindowed_probs, dtype=int)
predicted_binary[predicted_peaks] = 1

# === Load original labels ===
true_labels = np.load(os.path.join(output_directory, 'left_heel_labels_full.npy'))

# === Metrics ===
tp = np.sum((true_labels == 1) & (predicted_binary == 1))
tn = np.sum((true_labels == 0) & (predicted_binary == 0))
fp = np.sum((true_labels == 0) & (predicted_binary == 1))
fn = np.sum((true_labels == 1) & (predicted_binary == 0))

accuracy = accuracy_score(true_labels, predicted_binary)
f1 = f1_score(true_labels, predicted_binary)
sensitivity = recall_score(true_labels, predicted_binary)  # True positive rate
specificity = tn / (tn + fp + 1e-8)
cm = confusion_matrix(true_labels, predicted_binary)

# === Print Results ===
print("\n--- Evaluation Metrics ---")
print(f"Accuracy   : {accuracy:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print("\nConfusion Matrix:")
print(cm)

# === Optional: Plot example ===
plt.figure(figsize=(12, 4))
plt.plot(unwindowed_probs, label='Predicted Probability (Smoothed)', alpha=0.7)
plt.plot(true_labels, label='True Label', alpha=0.4)
plt.plot(predicted_binary, label='Predicted Label (Peaks)', alpha=0.6)
plt.legend()
plt.title("Predicted vs True Labels")
plt.tight_layout()
plt.show()
