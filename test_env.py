import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from quadrotor_env import QuadPole2D
import time


def test_quadrotor_env():
    """Test the QuadPole2D environment with random actions and pygame rendering"""

    # Initialize pygame
    pygame.init()

    # Set up display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("QuadPole2D Environment Test")
    clock = pygame.time.Clock()

    # Initialize environment
    env = QuadPole2D(max_steps=200, timestep=0.02)

    # Test for 5 episodes
    for episode in range(5):
        print(f"\n=== Episode {episode + 1} ===")

        # Reset environment
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0

        running = True
        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        pygame.quit()
                        return

            # Generate random action
            action = env.action_space.sample()

            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

            # Create matplotlib figure for rendering
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_facecolor('lightblue')

            # Render the environment
            env.render(ax, observation=obs)

            # Add episode info to the plot
            ax.text(0.02, 0.98, f'Episode: {episode + 1}',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(0.02, 0.92, f'Step: {episode_steps}',
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

            # Check if episode is done
            if terminated or truncated:
                print(f"Episode {episode + 1} finished!")
                print(f"Steps: {episode_steps}")
                print(f"Total Reward: {episode_reward:.2f}")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                print(f"Out of Bounds: {env.out_of_bounds()}")
                print(f"Time Balanced: {info.get('time_balanced', 0):.2f}s")

                # Wait a moment before next episode
                time.sleep(1)
                break

    # Show completion message
    print("\n=== All episodes completed! ===")
    print("Press ESC or close window to exit...")

    # Keep window open until user closes it
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    waiting = False

        # Show completion screen
        screen.fill((50, 50, 50))
        font = pygame.font.Font(None, 74)
        text = font.render("Test Complete!", True, (255, 255, 255))
        text_rect = text.get_rect(center=(width // 2, height // 2))
        screen.blit(text, text_rect)

        font_small = pygame.font.Font(None, 36)
        text_small = font_small.render("Press ESC or close window to exit", True, (200, 200, 200))
        text_small_rect = text_small.get_rect(center=(width // 2, height // 2 + 50))
        screen.blit(text_small, text_small_rect)

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    print("Starting QuadPole2D Environment Test")
    print("Controls: ESC to exit, Close window to quit")
    print("Running 5 episodes with random actions...")

    try:
        test_quadrotor_env()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        pygame.quit()
    except Exception as e:
        print(f"Error occurred: {e}")
        pygame.quit()
        raise