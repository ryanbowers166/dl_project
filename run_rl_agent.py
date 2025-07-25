import multiprocessing
from datetime import datetime
import numpy as np
import gymnasium as gym
import matplotlib.backends.backend_agg as agg
import pygame
import json

from stable_baselines3 import PPO
from train import QuadPole2DWrapper
import matplotlib.pyplot as plt

from quadrotor_env import QuadPole2D

def test_trained_agent(config, render_mode, model_path, n_episodes=5, manual_goal_position=None):
    """
    Test a trained agent

    Args:
        model_path: Path to the saved model
        n_episodes: Number of episodes to test
    """

    config['curriculum_level'] = 2

    # Initialize pygame
    #pygame.init()
    pygame.display.init()
    pygame.mixer.pre_init()#frequency=44100, size=-16, channels=2, buffersize=512, allowedchanges=pygame.AUDIO_ALLOW_ANY_CHANGE)
    pygame.mixer.init()
    #pygame.joystick.init()

    # Set up display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("QuadPole2D Environment Test")
    clock = pygame.time.Clock()

    # Load the trained model
    model = PPO.load(model_path)

    # Create test environment
    env = QuadPole2DWrapper(config, render_mode, manual_goal_position)

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


if __name__ == "__main__":

    config_filename = './configs/config_v4.json' # Choose which config file to use
    model_path = './saved_models/gamma sweep/0607_1042_poscos20_gamma999' # Choose which model to load (should be a zipped file or compressed folder)

    with open(config_filename, 'r') as file:
        config = json.load(file)
    config['config_filename'] = config_filename

    print("\nTesting trained agent...")
    test_trained_agent(
        config,
        render_mode='human',
        model_path=model_path,
        n_episodes=5,
        manual_goal_position="dynamic-1")