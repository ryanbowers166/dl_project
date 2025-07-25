import multiprocessing
from datetime import datetime
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from collections import defaultdict

from stable_baselines3 import PPO
from training_pipeline import QuadPole2DWrapper
from quadrotor_env import QuadPole2D


def evaluate_single_model(model_path, config, n_episodes=100):
    """
    Evaluate a single model and return performance metrics

    Args:
        model_path: Path to the saved model
        config: Configuration dictionary
        n_episodes: Number of episodes to evaluate

    Returns:
        dict: Dictionary containing all computed metrics
    """
    print(f"Evaluating model: {model_path}")

    # Load the trained model
    model = PPO.load(model_path)

    # Create test environment with dynamic goal position
    env = QuadPole2DWrapper(config, render_mode='human', manual_goal_position="dynamic-1")

    # Metrics storage
    metrics = {
        'time_to_first_waypoint': [],
        'time_to_pendulum_upright': [],
        'time_to_first_goal_state': [],
        'pendulum_falls_count': [],
        'time_to_second_waypoint': [],
        'total_waypoint_time': []
    }

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_metrics = evaluate_episode(env, model, obs)

        # Append metrics for this episode
        for key in metrics:
            metrics[key].append(episode_metrics[key])

        if (episode + 1) % 20 == 0:
            print(f"  Completed {episode + 1}/{n_episodes} episodes")

    # Compute summary statistics
    summary_metrics = {}
    for key, values in metrics.items():
        valid_values = [v for v in values if v is not None and v != float('inf')]
        if valid_values:
            summary_metrics[key] = {
                'mean': np.mean(valid_values),
                'std': np.std(valid_values),
                'median': np.median(valid_values),
                'success_rate': len(valid_values) / len(values)
            }
        else:
            summary_metrics[key] = {
                'mean': float('inf'),
                'std': 0,
                'median': float('inf'),
                'success_rate': 0
            }

    return summary_metrics


def evaluate_episode(env, model, initial_obs):
    """
    Evaluate a single episode and extract performance metrics

    Args:
        env: Environment instance
        model: Trained model
        initial_obs: Initial observation

    Returns:
        dict: Episode metrics
    """
    obs = initial_obs
    episode_length = 0
    done = False

    # Tracking variables
    first_waypoint_reached = False
    pendulum_upright_achieved = False
    first_goal_state_achieved = False
    second_waypoint_reached = False
    pendulum_falls_count = 0
    last_upright_check = False

    # Time tracking
    time_to_first_waypoint = None
    time_to_pendulum_upright = None
    time_to_first_goal_state = None
    time_to_second_waypoint = None

    # Goal positions (based on dynamic-1 pattern)
    first_goal = np.array([1.0, 1.0])
    second_goal = np.array([0.0, 0.0])
    goal_switch_step = 250

    pendulum_is_currently_upright = False
    while not done:

        # Use the trained policy to select actions
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        episode_length += 1
        current_time = episode_length * env.env.timestep

        # Extract state information
        x, z = obs[0], obs[1]  # Position
        s_phi, c_phi = obs[7], obs[8]  # Pendulum sin/cos
        current_pos = np.array([x, z])

        # Check if pendulum is upright (cos(phi) < -0.92 means phi is close to π)
        is_pendulum_upright = c_phi < -0.92

        # Determine current goal based on episode step
        if episode_length <= goal_switch_step:
            current_goal = first_goal
        else:
            current_goal = second_goal

        # Check if at waypoint (within reasonable distance)
        at_waypoint = np.linalg.norm(current_pos - current_goal) < 0.3

        # Track first waypoint achievement
        if not first_waypoint_reached and at_waypoint and episode_length <= goal_switch_step:
            first_waypoint_reached = True
            time_to_first_waypoint = current_time

        # Track pendulum upright achievement
        if not pendulum_upright_achieved and is_pendulum_upright:
            pendulum_upright_achieved = True
            time_to_pendulum_upright = current_time

        # Track first goal state (at waypoint AND pendulum upright)
        if not first_goal_state_achieved and at_waypoint and is_pendulum_upright and episode_length <= goal_switch_step:
            first_goal_state_achieved = True
            time_to_first_goal_state = current_time

        # Track second waypoint achievement (after goal switch)
        if not second_waypoint_reached and at_waypoint and episode_length > goal_switch_step:
            second_waypoint_reached = True
            time_to_second_waypoint = current_time

        # Track pendulum falls (only after it was initially upright)
        if pendulum_upright_achieved:
            if c_phi < -0.92:
                pendulum_is_currently_upright = True
            elif c_phi > -0.82 and pendulum_is_currently_upright:
                pendulum_falls_count += 1
                pendulum_is_currently_upright = False

        # if pendulum_upright_achieved:
        #     # Check if pendulum fell (more than 20 degrees from upright)
        #     # cos(π ± 20°) ≈ cos(160°) ≈ -0.94, cos(π ± 20°) ≈ cos(200°) ≈ -0.94
        #     pendulum_fell = c_phi > -0.82  # More than ~25 degrees from upright for some margin
        #
        #     if last_upright_check and pendulum_fell:
        #         pendulum_falls_count += 1
        #
        #     last_upright_check = is_pendulum_upright

        done = terminated or truncated

    # Calculate total waypoint time
    total_waypoint_time = None
    if time_to_second_waypoint is not None:
        total_waypoint_time = time_to_second_waypoint
    elif time_to_first_waypoint is not None:
        total_waypoint_time = time_to_first_waypoint

    return {
        'time_to_first_waypoint': time_to_first_waypoint,
        'time_to_pendulum_upright': time_to_pendulum_upright,
        'time_to_first_goal_state': time_to_first_goal_state,
        'pendulum_falls_count': pendulum_falls_count,
        'time_to_second_waypoint': time_to_second_waypoint,
        'total_waypoint_time': total_waypoint_time
    }


def plot_comparison_metrics(results, save_path=None):
    """
    Plot comparison metrics for all evaluated models

    Args:
        results: Dictionary with model names as keys and metrics as values
        save_path: Optional path to save the plots
    """
    model_names = list(results.keys())
    metrics_to_plot = [
        'time_to_first_waypoint',
        'time_to_pendulum_upright',
        'time_to_first_goal_state',
        'pendulum_falls_count',
        'time_to_second_waypoint',
        'total_waypoint_time'
    ]

    metric_labels = {
        'time_to_first_waypoint': 'Time to First Waypoint (s)',
        'time_to_pendulum_upright': 'Time to Pendulum Upright (s)',
        'time_to_first_goal_state': 'Time to First Goal State (s)',
        'pendulum_falls_count': 'Pendulum Falls Count',
        'time_to_second_waypoint': 'Time to Second Waypoint (s)',
        'total_waypoint_time': 'Average Total Waypoint Time (s)'
    }

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]

        means = []
        stds = []
        success_rates = []

        for model_name in model_names:
            metric_data = results[model_name][metric]
            means.append(metric_data['mean'])
            stds.append(metric_data['std'])
            success_rates.append(metric_data['success_rate'])

        # Create bar plot with error bars
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)

        # # Color bars based on success rate
        # for j, (bar, success_rate) in enumerate(zip(bars, success_rates)):
        #     if success_rate > 0.8:
        #         bar.set_color('green')
        #     elif success_rate > 0.5:
        #         bar.set_color('orange')
        #     else:
        #         bar.set_color('red')

        ax.set_xlabel('Model')
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(f'{metric_labels[metric]}')#\n(Color: Green=80%+, Orange=50-80%, Red=<50% success)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # # Add success rate text on bars
        # for j, (bar, success_rate) in enumerate(zip(bars, success_rates)):
        #     height = bar.get_height()
        #     if not np.isinf(height):
        #         ax.text(bar.get_x() + bar.get_width() / 2., height + stds[j],
        #                 f'{success_rate:.1%}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {save_path}")

    plt.show()


def print_comparison_table(results):
    """
    Print a comparison table of all metrics for all models

    Args:
        results: Dictionary with model names as keys and metrics as values
    """
    print("\n" + "=" * 120)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 120)

    # Header
    print(f"{'Model':<25} {'Metric':<25} {'Mean':<12} {'Std':<12} {'Median':<12} {'Success Rate':<12}")
    print("-" * 120)

    metrics_order = [
        'time_to_first_waypoint',
        'time_to_pendulum_upright',
        'time_to_first_goal_state',
        'pendulum_falls_count',
        'time_to_second_waypoint',
        'total_waypoint_time'
    ]

    for model_name in results.keys():
        for i, metric in enumerate(metrics_order):
            metric_data = results[model_name][metric]

            # Format values
            mean_str = f"{metric_data['mean']:.2f}" if not np.isinf(metric_data['mean']) else "∞"
            std_str = f"{metric_data['std']:.2f}"
            median_str = f"{metric_data['median']:.2f}" if not np.isinf(metric_data['median']) else "∞"
            success_str = f"{metric_data['success_rate']:.1%}"

            model_display = model_name if i == 0 else ""
            print(f"{model_display:<25} {metric:<25} {mean_str:<12} {std_str:<12} {median_str:<12} {success_str:<12}")

        print("-" * 120)


def main():
    """
    Main evaluation function
    """
    # Configuration
    config_filename = './configs/config_v5.json'

    # List of model paths to evaluate
    model_paths = [
        #'./saved_models/gamma sweep/0607_1042_poscos20_gamma999',
        #'./saved_models/gamma sweep/0607_1042_poscos20_gamma995',
        #'./saved_models/gamma sweep/0607_1042_poscos20_gamma99',
        './saved_models/pos_cost sweep/0607_0040_PPO_QuadPole2D_lr5e04_env6_7.0M.zip',
        './saved_models/pos_cost sweep/0607_0206_PPO_QuadPole2D_lr5e04_env6_7.0M.zip',
        #'./saved_models/pos_cost sweep/0607_0331_PPO_QuadPole2D_lr5e04_env6_7.0M.zip'
        # Add more model paths as needed
    ]

    num_episodes = 20

    # Load configuration
    with open(config_filename, 'r') as file:
        config = json.load(file)
    config['config_filename'] = config_filename
    config['curriculum_level'] = 2  # Set appropriate curriculum level for testing

    # Evaluate all models
    results = {}

    for model_path in model_paths:
        if not os.path.exists(model_path + '.zip') and not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist, skipping...")
            continue

        # Extract model name from path
        #model_name = Path(model_path).name
        # Custom model names
        custom_names = {
            './saved_models/pos_cost sweep/0607_0040_PPO_QuadPole2D_lr5e04_env6_7.0M.zip': 'P_pos = 15',
            './saved_models/pos_cost sweep/0607_0206_PPO_QuadPole2D_lr5e04_env6_7.0M.zip': 'P_pos = 20',
            './saved_models/pos_cost sweep/0607_0331_PPO_QuadPole2D_lr5e04_env6_7.0M.zip': 'P_pos = 25'
        }

        model_name = custom_names.get(model_path, Path(model_path).name)

        try:
            # Evaluate the model
            model_results = evaluate_single_model(model_path, config, n_episodes=num_episodes)
            results[model_name] = model_results

            print(f"Completed evaluation for {model_name}")

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    if not results:
        print("No models were successfully evaluated!")
        return

    # Print comparison table
    print_comparison_table(results)

    # Create comparison plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_save_path = f"./evaluation_comparison_{timestamp}.png"
    plot_comparison_metrics(results, save_path=plot_save_path)

    # Save results to JSON for future reference
    results_save_path = f"./evaluation_results_{timestamp}.json"

    # Convert numpy types to native Python types for JSON serialization
    serializable_results = {}
    for model_name, metrics in results.items():
        serializable_results[model_name] = {}
        for metric_name, metric_data in metrics.items():
            serializable_results[model_name][metric_name] = {
                'mean': float(metric_data['mean']),
                'std': float(metric_data['std']),
                'median': float(metric_data['median']),
                'success_rate': float(metric_data['success_rate'])
            }

    with open(results_save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {results_save_path}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()