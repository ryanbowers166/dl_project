import optuna
from datetime import datetime

import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from wandb.integration.sb3 import WandbCallback
from training_pipeline import QuadPole2DWrapper
import sklearn


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
                time_balanced = self.eval_env.env.env.env.total_time_balanced

            # Log custom metrics to wandb
            wandb.log({
                "c_eval/episode_reward": episode_reward,
                "c_eval/episode_length": episode_length,
                "c_eval/time_balanced": time_balanced,
                "c_eval/max_payload_angle": max_payload_angle,
                "c_eval/final_position_error": np.linalg.norm(obs[:2]),
            }, step=self.num_timesteps)

        return True


def make_env(config):
    return Monitor(QuadPole2DWrapper(config))


def objective(trial):
    # Suggest hyperparameters
    #vf_coef = trial.suggest_float('vf_coef', 0.2, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0001, 0.0005, 0.00005])
    n_epochs = trial.suggest_categorical('n_epochs', [10, 20, 25])
    #n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
    #n_epochs = trial.suggest_int('n_epochs', 5, 20)
    #gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)

    # Fixed hyperparameters
    config = {
          "n_envs": 6,
          "total_timesteps": 20e6,
          "learning_rate": learning_rate,
          "n_steps": 2048,
          "batch_size": batch_size,
          "n_epochs": n_epochs,
          "gamma": 0.99,
          "gae_lambda": 0.95,
          "clip_range": 0.2,
          "ent_coef": 0.01,
          "vf_coef": 0.45,
          "curriculum_level": 0
        }

    now = datetime.now()
    timestamp = now.strftime("%m%d_%H%M")

    # Initialize wandb
    run = wandb.init(
        project='dl-project',
        name=f"{timestamp}_trial_{trial.number}",
        config=config,
        sync_tensorboard=True,  # Add this
        monitor_gym=True,  # Add this
        reinit=True
    )

    # Create environments
    env = make_vec_env(lambda: make_env(config), n_envs=config['n_envs'])
    eval_env = make_env(config)
    eval_env = Monitor(eval_env)

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        verbose=2,
        tensorboard_log=f"./logs/optuna/{run.id}"
    )

    # Setup callbacks
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        # model_save_path=f"other/{run.id}",
        verbose=2,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"../saved_models/best",
        log_path=f"../logs",
        eval_freq=10000,  # Evaluate every 10k steps
        deterministic=True,
        render=False
    )

    custom_callback = CustomMetricsCallback(
        eval_env=eval_env,
        log_freq=5000,  # Log custom metrics every 5k steps
    )

    # Train
    model.learn(total_timesteps=int(5e6), callback=[wandb_callback, eval_callback, custom_callback])

    # Log final result to wandb
    wandb.log({"final_mean_reward": eval_callback.last_mean_reward})
    wandb.finish()

    # Return mean reward from evaluation
    return eval_callback.last_mean_reward


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50)

    print("Best parameters:")
    print(study.best_params)
    print(f"Best value: {study.best_value}")

    # Visualization
    import matplotlib.pyplot as plt

    # Plot optimization history
    fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization History")
    plt.show()

    # Plot parameter importances
    fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Parameter Importances")
    plt.show()

    # Plot slice plot
    fig3 = optuna.visualization.matplotlib.plot_slice(study)
    plt.title("Slice Plot")
    plt.show()