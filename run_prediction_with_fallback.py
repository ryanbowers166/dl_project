import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# --- CONFIG ---
WINDOW_SIZE = 10
UNSAFE_CLASSES = [1]       # Class 1 = "falling"
SAFE_THRUST = 1.0          # Hover level thrust
MODEL_PATH = "saved_models/lstm_model.h5"  # Adjust if different

# --- Safety controller ---
def safety_controller():
    return np.array([SAFE_THRUST, SAFE_THRUST])

# --- Analyze full trajectory ---
def run_prediction_and_fallback(trajectory, lstm_model):
    """
    Run LSTM prediction on each window in a trajectory.
    If unsafe motion is predicted, switch to safety controller.
    """
    corrected_actions = []
    warnings = []
    T = trajectory.shape[0]
    switched = False

    for t in range(T):
        if not switched and t <= T - WINDOW_SIZE:
            window = trajectory[t:t+WINDOW_SIZE].reshape(1, WINDOW_SIZE, 14)
            pred = lstm_model.predict(window, verbose=0)
            cluster_id = np.argmax(pred[0])
            
            if cluster_id in UNSAFE_CLASSES:
                print(f"⚠️  Unsafe motion detected at step {t} (cluster {cluster_id}). Switching to safety.")
                switched = True
                warnings.append(t)

        # Choose action
        if switched:
            corrected_actions.append(safety_controller())
        else:
            corrected_actions.append(trajectory[t, 12:14])  # original action

    return np.array(corrected_actions), warnings

# def run_prediction_and_fallback(trajectory, lstm_model):
#     corrected_actions = trajectory[:, 12:14].copy()  # Start with original actions
#     warnings = []
#     T = trajectory.shape[0]
    
#     for t in range(T - WINDOW_SIZE):
#         window = trajectory[t:t+WINDOW_SIZE].reshape(1, WINDOW_SIZE, 14)
#         pred = lstm_model.predict(window, verbose=0)
#         cluster_id = np.argmax(pred[0])

#         if cluster_id in UNSAFE_CLASSES:
#             print(f"⚠️  Unsafe motion detected at step {t} (cluster {cluster_id}). Switching to safety.")
#             warnings.append(t)
#             corrected_actions[t:] = np.array([SAFE_THRUST, SAFE_THRUST])  # ← Replace only after trigger
#             break

#     return corrected_actions, warnings

# Load the test trajectory
df = pd.read_csv("state_action_dataset.csv")
trajectory = df.values  # shape (T, 14)

# Load trained LSTM model
from tensorflow.keras.models import load_model
model = load_model("saved_models/lstm_model.h5")

# Run detection + correction
corrected_actions, warnings = run_prediction_and_fallback(trajectory, model)

print("Total warnings:", len(warnings))
print("Corrected actions shape:", corrected_actions.shape)

# Save to CSV for visualization or later evaluation
np.savetxt("corrected_actions.csv", corrected_actions, delimiter=",", header="u1,u2", comments='')


def plot_action_comparison(original_actions, corrected_actions, warning_step=None, zoom_range=None):
    """
    Plot original vs. corrected actions with optional safety trigger line and zoom.

    Args:
        original_actions: np.array of shape (T, 2)
        corrected_actions: np.array of shape (T, 2)
        warning_step: int or None, timestep where safety was triggered
        zoom_range: tuple (start, end) to zoom in, e.g., (550, 650)
    """
    if zoom_range:
        start, end = zoom_range
        timesteps = np.arange(start, end)
        original_actions = original_actions[start:end]
        corrected_actions = corrected_actions[start:end]
        if warning_step is not None and not (start <= warning_step < end):
            warning_step = None
    else:
        timesteps = np.arange(len(original_actions))

    plt.figure(figsize=(12, 5))
    
    plt.plot(timesteps, original_actions[:, 0], 'k--', label="Original Action 0")
    plt.plot(timesteps, original_actions[:, 1], 'gray', linestyle='--', label="Original Action 1")
    plt.plot(timesteps, corrected_actions[:, 0], 'tab:red', label="Corrected Action 0")
    plt.plot(timesteps, corrected_actions[:, 1], 'tab:pink', label="Corrected Action 1")

    if warning_step is not None:
        plt.axvline(warning_step, color='red', linestyle=':', label="Safety Triggered")

    plt.title("Comparison of Actions: Original vs. Corrected (Safety Controller)")
    plt.xlabel("Timestep")
    plt.ylabel("Rotor Thrust")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Visualize (zoomed in around unsafe motion)
original_actions = trajectory[:, 12:14]

if warnings:
    center = warnings[0]
    zoom_range = (max(0, center - 50), center + 50)
else:
    zoom_range = None

plot_action_comparison(
    original_actions, 
    corrected_actions, 
    warning_step=warnings[0] if warnings else None,
    zoom_range=zoom_range
)
