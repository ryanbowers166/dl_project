import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from run_trained_agent_with_LQR_Controller import test_agent_combined

# === CONFIG ===
POLICY_PATHS = [
    # Add your policy .zip files here
    #'./saved_models/0714_2044_PPO_QuadPole2D_lr5e04_env4_7.0M/best/best_model.zip',
    #'./saved_models/0714_2207_PPO_QuadPole2D_lr5e04_env4_7.0M/best/best_model.zip',
    #'./saved_models/0714_2317_PPO_QuadPole2D_lr5e04_env4_7.0M/best/best_model.zip',
    #'./saved_models/0715_0023_PPO_QuadPole2D_lr5e04_env4_7.0M/best/best_model.zip',
    './saved_models/0715_0126_PPO_QuadPole2D_lr5e04_env4_7.0M/best/best_model.zip',
    #'./saved_models/0715_0230_PPO_QuadPole2D_lr5e04_env4_7.0M/best/best_model.zip',
    #'./saved_models/0715_0334_PPO_QuadPole2D_lr5e04_env4_7.0M/best/best_model.zip',
    './saved_models/0715_0437_PPO_QuadPole2D_lr5e04_env4_7.0M/best/best_model.zip',
    './saved_models/0715_0541_PPO_QuadPole2D_lr5e04_env4_7.0M/best/best_model.zip',
    #'./saved_models/0715_0644_PPO_QuadPole2D_lr5e04_env4_7.0M/best/best_model.zip',
]

POLICY_LABELS = [
    #"0.5",
    #"1.0",
    #"2.0",
    #"3.0",
    "4.0",
    #"5.0",
    #"7.0",
    "10.0",
    "15.0",
    #"20.0"
]

CONFIG_PATH = './configs/config_v4.json'
N_EPISODES = 1
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_results = []

# === RUN EVALUATION ===
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
cfg['config_filename'] = CONFIG_PATH

for i, path in enumerate(POLICY_PATHS):
    label = POLICY_LABELS[i]

    for mode in ['monitor', 'intervene']: #
        print(f"\\n=== Evaluating {label} | Mode: {mode} ===")
        metrics = test_agent_combined(
            config=cfg,
            render_mode='human',
            model_path=path,
            mode=mode,
            n_episodes=N_EPISODES,
            manual_goal_position="dynamic-1",
            live_plot = False,
            save_summary_plot=False
        )
        for m in metrics:
            m.update({'policy': label, 'mode': 'yesLQR' if mode == 'intervene' else 'noLQR'})
            all_results.append(m)

# === SAVE RESULTS ===
df = pd.DataFrame(all_results)
csv_path = os.path.join(OUTPUT_DIR, 'evaluation_summary.csv')
df.to_csv(csv_path, index=False)
print(f"Saved results to {{csv_path}}")

# === PLOT RESULTS ===
bar_width = 0.35
index = np.arange(len(POLICY_LABELS))
metrics = ['num_anomalies', 'time_balanced', 'goal1_reach_time', 'goal2_reach_time'] # lqr_duration


fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()
ylim_maxes = [100, 550, 5, 5]

for i, metric in enumerate(metrics):
    
    ax = axes[i]
    #ax.set_ylim(bottom=0)
    noLQR_vals = df[df['mode'] == 'noLQR'].groupby('policy')[metric].mean().reindex(POLICY_LABELS).values
    yesLQR_vals = df[df['mode'] == 'yesLQR'].groupby('policy')[metric].mean().reindex(POLICY_LABELS).values
    
    print(f"{metric} - noLQR: {noLQR_vals}")
    print(f"{metric} - yesLQR: {yesLQR_vals}")
    
    #ax.set_ylim(bottom=0)
    ax.bar(index, noLQR_vals, bar_width, label='noLQR')
    ax.bar(index + bar_width, yesLQR_vals, bar_width, label='yesLQR')
    #ax.set_ylim(bottom=0)
    ax.set_title(metric.replace('_', ' ').title())
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(POLICY_LABELS, rotation=0)
    ax.grid(alpha=0.3)
    
    ax.legend()
    
    ax.set_ylim(0, ylim_maxes[i])



plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'evaluation_metrics_comparison.png')
plt.savefig(plot_path)
print(f"Saved comparison plot to {{plot_path}}")
