import multiprocessing
from datetime import datetime
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pygame
import json

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
import matplotlib.pyplot as plt
import wandb
from wandb.integration.sb3 import WandbCallback

from quadrotor_env import QuadPole2D

class QuadPole2DWrapper(gym.Env):
    """Gymnasium wrapper for QuadPole2D environment"""

    def __init__(self, config, render_mode, manual_goal_position):
        super().__init__()
        self.env = QuadPole2D(config, render_mode, manual_goal_position)

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

    def render(self, render_mode='human'):
        """Render the environment (optional)"""
        pass

    def _time_balanced(self):
        """Expose the time_balanced metric from the underlying environment"""
        return getattr(self.env, '_time_balanced', 0)

    def close(self):
        """Close the environment"""
        pass


def make_env(config, render_mode, manual_goal_position=None):
    """Factory function to create environment instances"""
    return Monitor(QuadPole2DWrapper(config, render_mode, manual_goal_position))


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

    def __init__(self, eval_env, log_freq=1000, use_wandb=False, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_freq = log_freq
        self.use_wandb = use_wandb

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
                time_balanced = self.eval_env.env.env.total_time_balanced

            # Log custom metrics to wandb
            if self.use_wandb:
                wandb.log({
                    "c_eval/episode_reward": episode_reward,
                    "c_eval/episode_length": episode_length,
                    "c_eval/time_balanced": time_balanced,
                    "c_eval/max_payload_angle": max_payload_angle,
                    "c_eval/final_position_error": np.linalg.norm(obs[:2]),
                }, step=self.num_timesteps)
            else:
                # Optional: print metrics to console instead
                print(f"Step {self.num_timesteps}: reward={episode_reward:.2f}, "
                      f"time_balanced={time_balanced:.2f}, max_angle={max_payload_angle:.2f}")

            return True

        return True

def train_ppo_agent(config, render_mode, manual_goal_position=None, use_wandb = True):
    """
    Train a PPO agent on the QuadPole2D environment
    Args:
        total_timesteps: Total number of training timesteps
        n_envs: Number of parallel environments
        save_path: Path to save the trained model
    """

    # Generate unique run name
    run_name = generate_run_name(config)

    # Create envs for training and eval
    env = make_vec_env(lambda: make_env(config, render_mode, manual_goal_position), n_envs=config.get('n_envs', 4))
    eval_env = make_env(config,"eval", manual_goal_position)

    if use_wandb:
        run = wandb.init(
            project='dl-project',
            name=run_name,
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    tensorboard_log = f"./logs/runs/{run_name}" if config.get('use_wandb', False) else None

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
        verbose=2,
        tensorboard_log=tensorboard_log,  # Use wandb run ID for tensorboard
    )

    ################################ Setup callbacks ################################
    callbacks = []

    if use_wandb:
        wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)
        callbacks.append(wandb_callback)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./saved_models/{run_name}/best",
        log_path=f"./logs/runs/{run_name}/eval",
        eval_freq=10000,
        deterministic=True, render=False
    )
    callbacks.append(eval_callback)

    custom_callback = CustomMetricsCallback(
        eval_env=eval_env,
        log_freq=5000,
        use_wandb=use_wandb
    )
    callbacks.append(custom_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=f"./saved_models/{run_name}/checkpoints/",
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )
    callbacks.append(checkpoint_callback)


    ################################ Start training ################################
    print("Starting training...")
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callbacks,
        progress_bar=True
    )

    if use_wandb:
        run.finish()

    # Save the final model
    model.save(f'./saved_models/{run_name}')
    print(f"Training completed! Model saved to ./saved_models")

    return model



if __name__ == "__main__":

    # Load config
    config_filename = './configs/config_v4.json'
    use_wandb = False # Set to false if you don't want to log training metrics to wandb

    with open(config_filename, 'r') as file:
        config = json.load(file)
    config['config_filename'] = config_filename
    config['n_envs'] = multiprocessing.cpu_count()

    
    print(f"Training PPO agent on QuadPole2D environment with {config['n_envs']} envs")
    model = train_ppo_agent(config,"train", use_wandb=use_wandb)