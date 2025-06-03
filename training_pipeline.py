from datetime import datetime
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pygame

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import matplotlib.pyplot as plt
import wandb
from wandb.integration.sb3 import WandbCallback

from quadrotor_env import QuadPole2D

class QuadPole2DWrapper(gym.Env):
    """Gymnasium wrapper for QuadPole2D environment"""

    def __init__(self):
        super().__init__()
        self.env = QuadPole2D()

        # Use the observation and action spaces from the original environment
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation"""
        if seed is not None:
            np.random.seed(seed)
        obs, info = self.env.reset()
        return obs.astype(np.float32), info

    def step(self, action):
        """Take a step in the environment"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the environment (optional)"""
        pass  # Implement if needed for visualization

    def close(self):
        """Close the environment"""
        pass


def make_env():
    """Factory function to create environment instances"""
    return QuadPole2DWrapper()


def generate_run_name(config):
    """
    Generate a unique run name based on timestamp and config parameters

    Args:
        config: Dictionary containing training configuration parameters

    Returns:
        str: Formatted run name string in format MMDD_HHMM_{config_params}
    """
    # Get current timestamp
    now = datetime.now()
    timestamp = now.strftime("%m%d_%H%M")

    # Extract key config parameters for the run name
    algorithm = config.get('algorithm', 'PPO')
    env_name = config.get('env_name', 'QuadPole2D')
    learning_rate = config.get('learning_rate', 3e-4)
    n_envs = config.get('n_envs', 4)
    total_timesteps = config.get('total_timesteps', 100000)

    # Format learning rate for readability (e.g., 3e-4 -> 3e4)
    lr_str = f"{learning_rate:.0e}".replace('-', '')

    # Format timesteps (e.g., 500000 -> 500k)
    if total_timesteps >= 1000000:
        timesteps_str = f"{total_timesteps // 1000000}M"
    elif total_timesteps >= 1000:
        timesteps_str = f"{total_timesteps // 1000}k"
    else:
        timesteps_str = str(total_timesteps)

    # Create run name
    run_name = f"{timestamp}_{algorithm}_{env_name}_lr{lr_str}_env{n_envs}_{timesteps_str}"

    return run_name


class CustomMetricsCallback(BaseCallback):
    """Custom callback to log environment-specific metrics to wandb"""

    def __init__(self, eval_env, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Run a quick evaluation episode to get metrics
            obs, info = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            time_balanced = 0
            max_payload_angle = 0
            done = False

            while not done and episode_length < 500:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

                # Extract payload angle from observation
                s_phi, c_phi = obs[7], obs[8]
                phi = np.arctan2(s_phi, c_phi)
                max_payload_angle = max(max_payload_angle, abs(phi))

                # Track time balanced (if info contains it)
                if hasattr(self.eval_env, '_time_balanced'):
                    time_balanced = self.eval_env._time_balanced

            # Log custom metrics to wandb
            wandb.log({
                "c_eval/episode_reward": episode_reward,
                "c_eval/episode_length": episode_length,
                "c_eval/time_balanced": time_balanced,
                "c_eval/max_payload_angle": max_payload_angle,
                "c_eval/final_position_error": np.linalg.norm(obs[:2]),
            }, step=self.num_timesteps)

        return True

def train_ppo_agent(total_timesteps=100000, n_envs=4, save_path="saved_model"):
    """
    Train a PPO agent on the QuadPole2D environment

    Args:
        total_timesteps: Total number of training timesteps
        n_envs: Number of parallel environments
        save_path: Path to save the trained model
    """

    # Create vectorized environment for parallel training
    env = make_vec_env(make_env, n_envs=n_envs)

    # Create evaluation environment
    eval_env = make_env()

    config = {
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "algorithm": "PPO",
        "env_name": "QuadPole2D",
    }

    run_name = generate_run_name(config)

    run = wandb.init(
        project='dl-project',
        name=run_name,
        config=config,
        sync_tensorboard=True,  # Also sync tensorboard logs
        monitor_gym=True,  # Automatically log gym environments
        save_code=True,  # Save code for reproducibility
    )

    # Initialize PPO agent with reasonable hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        verbose=1,
        tensorboard_log=f"runs/{run.id}",  # Use wandb run ID for tensorboard
    )

    # Setup callbacks
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        #model_save_path=f"models/{run.id}",
        verbose=2,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}_best",
        log_path=f"{save_path}_logs",
        eval_freq=10000,  # Evaluate every 10k steps
        deterministic=True,
        render=False
    )

    custom_callback = CustomMetricsCallback(
        eval_env=eval_env,
        log_freq=5000,  # Log custom metrics every 5k steps
    )

    print("Starting training...")

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[wandb_callback, eval_callback, custom_callback],
        progress_bar=True
    )

    # Save the final model
    model.save(save_path)
    # TODO: Add model saving (currently blocked by GT computer admin privileges
    print(f"Training completed! Model saved to {save_path}")

    return model


def test_trained_agent(model_path="quadpole_ppo", n_episodes=5):
    """
    Test a trained agent

    Args:
        model_path: Path to the saved model
        n_episodes: Number of episodes to test
    """

    # Initialize pygame
    pygame.init()

    # Set up display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("QuadPole2D Environment Test")
    clock = pygame.time.Clock()

    # Load the trained model
    model = PPO.load(model_path)

    # Create test environment
    env = QuadPole2DWrapper()

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Handle pygame events to prevent window from becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return episode_rewards, episode_lengths

            # Use the trained policy to select actions
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Create matplotlib figure for rendering
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_facecolor('lightblue')

            # Render the environment - pass the observation to the render method
            env.env.render(ax, observation=obs)  # Use env.env to access the QuadPole2D instance

            # Add episode info to the plot
            ax.text(0.02, 0.98, f'Episode: {episode + 1}',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(0.02, 0.92, f'Step: {episode_length + 1}',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(0.02, 0.86, f'Reward: {reward:.2f}',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(0.02, 0.80, f'Total Reward: {episode_reward:.2f}',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Add action info
            ax.text(0.98, 0.98, f'Action: [{action[0]:.2f}, {action[1]:.2f}]',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f'QuadPole2D - Episode {episode + 1}')
            ax.grid(True, alpha=0.3)

            # Convert matplotlib figure to pygame surface
            canvas = agg.FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()

            # Get the RGBA buffer and convert to RGB
            buf = renderer.buffer_rgba()
            size = canvas.get_width_height()

            # Create pygame surface from matplotlib data
            surf = pygame.image.frombuffer(buf, size, 'RGBA')

            # Scale to fit screen if necessary
            surf = pygame.transform.scale(surf, (width, height))

            # Blit to screen
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Close matplotlib figure to free memory
            plt.close(fig)

            # Control frame rate
            clock.tick(30)  # 30 FPS

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    pygame.quit()
    print(f"\nAverage reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")

    return episode_rewards, episode_lengths


def visualize_performance(model_path="quadpole_ppo"):
    """
    Visualize the trained agent's performance
    """
    # Load the trained model
    model = PPO.load(model_path)
    env = QuadPole2DWrapper()

    # Run one episode and collect trajectory
    obs, info = env.reset()
    trajectory = [obs.copy()]
    actions = []
    rewards = []
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        trajectory.append(obs.copy())
        actions.append(action)
        rewards.append(reward)
        done = terminated or truncated

    trajectory = np.array(trajectory)

    # Plot trajectory
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Position trajectory
    axes[0, 0].plot(trajectory[:, 0], trajectory[:, 1])
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Z Position')
    axes[0, 0].set_title('Quadrotor Trajectory')
    axes[0, 0].grid(True)

    # Quadrotor angle over time
    theta = np.arctan2(trajectory[:, 4], trajectory[:, 5])  # From sin/cos
    axes[0, 1].plot(theta)
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Quadrotor Angle (rad)')
    axes[0, 1].set_title('Quadrotor Orientation')
    axes[0, 1].grid(True)

    # Payload angle over time
    phi = np.arctan2(trajectory[:, 7], trajectory[:, 8])  # From sin/cos
    axes[1, 0].plot(phi)
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Payload Angle (rad)')
    axes[1, 0].set_title('Payload Swing Angle')
    axes[1, 0].grid(True)

    # Rewards over time
    axes[1, 1].plot(rewards)
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].set_title('Reward per Step')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Train the agent
    #print("Training PPO agent on QuadPole2D environment...")
    #model = train_ppo_agent(total_timesteps=7000000, n_envs=6)

    # Test the trained agent
    print("\nTesting trained agent...")
    test_trained_agent("saved_model.zip", n_episodes=10)

    # Visualize performance
    #print("\nVisualizing performance...")
    #visualize_performance("saved_models.zip")