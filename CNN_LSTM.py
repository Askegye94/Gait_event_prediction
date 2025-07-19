# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 02:30:41 2025

@author: gib445
"""
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# === Configuration ===
output_directory = r'C:\Data\ProcessedNumpy'
window_size = 100
overlap = 50
dropout_rate = 0.5
l2_regularization = 0.001
base_neurons = 32
num_folds = 5
max_epochs = 200
patience = 5
min_improvement = 0.05  # 5%

# === Load full data and labels ===
def load_data_and_labels(_):
    X = np.load(os.path.join(output_directory, "full_Y.npy"))
    labels = np.load(os.path.join(output_directory, "left_heel_labels_full.npy"))
    return X, labels

X_raw, y_raw = load_data_and_labels('full')

# === Windowing ===
def window_data_and_labels(X, labels, window_size, overlap):
    step = window_size - overlap
    X_windows, y_windows = [], []

    for i in range((len(X) - window_size) // step + 1):
        start = i * step
        end = start + window_size
        center = labels[start + 40:start + 60]
        label = 1 if np.any(center) else 0

        X_windows.append(X[start:end])
        y_windows.append(label)

    return np.array(X_windows), np.array(y_windows).reshape(-1, 1)

X_win, y_bin = window_data_and_labels(X_raw, y_raw, window_size, overlap)

# === One-hot encoding ===
def one_hot_encode(y):
    return OneHotEncoder(sparse=False).fit_transform(y)

y_encoded = one_hot_encode(y_bin)

# === Participant IDs (from full_labels.npy) ===
participant_labels = np.load(os.path.join(output_directory, 'full_labels.npy'), allow_pickle=True)
participant_ids = np.array([int(p[2]) for p in participant_labels])  # p = (condition, speed, pid)

# Get participant ID for the center of each window
step = window_size - overlap
pid_windowed = np.array([participant_ids[i * step + 50] for i in range(len(y_bin))])

# === Class Weights ===
flattened = y_encoded[:, 1]
weights = compute_class_weight('balanced', classes=np.unique(flattened), y=flattened)
class_weight_dict = {i: w for i, w in enumerate(weights)}

# === Cross-validation ===
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
input_shape = (X_win.shape[1], X_win.shape[2])
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(np.unique(pid_windowed)), start=1):
    print(f"\n========== Fold {fold} ==========")

    train_pids = np.unique(pid_windowed)[train_idx]
    val_pids = np.unique(pid_windowed)[val_idx]

    train_mask = np.isin(pid_windowed, train_pids)
    val_mask = np.isin(pid_windowed, val_pids)

    X_train_fold = X_win[train_mask]
    y_train_fold = y_encoded[train_mask]
    X_val_fold = X_win[val_mask]
    y_val_fold = y_encoded[val_mask]

    # === Model ===
    model = Sequential([
        Conv1D(base_neurons, 3, activation='relu', input_shape=input_shape, kernel_regularizer=l2(l2_regularization)),
        MaxPooling1D(2),
        Conv1D(base_neurons * 2, 3, activation='relu', kernel_regularizer=l2(l2_regularization)),
        MaxPooling1D(2),
        LSTM(base_neurons * 2, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(base_neurons * 2),
        Dropout(dropout_rate),
        Dense(base_neurons * 4, activation='relu', kernel_regularizer=l2(l2_regularization)),
        Dropout(dropout_rate),
        Dense(base_neurons * 2, activation='relu', kernel_regularizer=l2(l2_regularization)),
        Dropout(dropout_rate),
        Dense(2, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # === Custom Early Stopping ===
    best_acc = 0
    best_weights = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        print(f"Epoch {epoch}/{max_epochs}")
        history = model.fit(X_train_fold, y_train_fold,
                            validation_data=(X_val_fold, y_val_fold),
                            class_weight=class_weight_dict,
                            verbose=0)

        val_acc = history.history['val_accuracy'][0]
        improvement = (val_acc - best_acc) / (best_acc + 1e-8)
        print(f"Val Acc: {val_acc:.4f} | Improvement: {improvement:.2%}")

        if improvement > min_improvement:
            best_acc = val_acc
            best_weights = model.get_weights()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"⏹️ Early stopping at epoch {epoch}")
            break

        # Save best model and validation predictions
        if best_weights:
            model.set_weights(best_weights)
        
            # Save model
            model_path = os.path.join(output_directory, f"cnn_lstm_fold{fold}.h5")
            model.save(model_path)
            print(f"✅ Saved best model to {model_path}")
        
            # Predict probabilities on validation set
            y_val_probs = model.predict(X_val_fold, verbose=0)
        
            # Save probabilities and true labels
            prob_path = os.path.join(output_directory, f"cnn_lstm_probs_fold{fold}.npz")
            np.savez(prob_path, probs=y_val_probs, labels=y_val_fold)
            print(f"📊 Saved predictions to {prob_path}")

# Save best accuracy summary
fold_results.append(f"Fold {fold}: Best Val Acc = {best_acc:.4f}")
