import json
import matplotlib.pyplot as plt
import numpy as np

# === Load data ===
with open("evaluation_results_20250725_114903.json", "r") as f:
    data = json.load(f)

# === Plot config ===
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (3.2, 2.6)  # slightly tighter width
})

metrics_to_plot = [
    "time_to_first_waypoint",
    "time_to_second_waypoint",
    "time_to_reach_all_waypoints",
    "time_to_pendulum_upright",
    "time_to_first_goal_state",
    "pendulum_falls_count"
]

labels = list(data.keys())  # e.g., ["P_pos = 15", "P_pos = 20"]
x = np.arange(len(labels))
bar_width = 0.25

for metric in metrics_to_plot:
    means = [data[label][metric]["mean"] for label in labels]
    stds = [data[label][metric]["std"] for label in labels]

    fig, ax = plt.subplots()
    bars = ax.bar(x, means, capsize=4, width=bar_width, color=["#1f77b4", "#ff7f0e"])

    ax.set_ylabel("Time (s)" if "time" in metric else "Count")
    ax.set_title(metric.replace("_", " ").capitalize(), pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0)
    #
    # for bar, mean in zip(bars, means):
    #     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{mean:.2f}",
    #             ha='center', va='bottom', fontsize=8)

    fig.tight_layout(pad=0.2)
    plt.savefig(f"finalplots/{metric}_barplot.png", dpi=300, bbox_inches="tight")  # Save as PNG
    plt.close()
