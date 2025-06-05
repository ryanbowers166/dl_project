import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from quadrotor_env_new import QuadPole2D
import time
import json


def test_quadrotor_env_human(config_filename):
    """Test the QuadPole2D environment with human control using pygame"""

    # Initialize pygame
    pygame.init()

    # Set up display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("QuadPole2D - Human Control")
    clock = pygame.time.Clock()

    with open(config_filename, 'r') as file:
        config = json.load(file)

    # Initialize environment
    env = QuadPole2D(config, max_steps=1000, timestep=0.02)  # Longer episodes for human play

    # Control variables
    left_thrust = 0.0  # Thrust adjustment for left rotor
    right_thrust = 0.0  # Thrust adjustment for right rotor
    thrust_increment = 0.1  # How much thrust changes per key press

    print("\n=== CONTROLS ===")
    print("LEFT ARROW:  Decrease left rotor thrust")
    print("RIGHT ARROW: Decrease right rotor thrust")
    print("UP ARROW:    Increase both rotors thrust")
    print("DOWN ARROW:  Decrease both rotors thrust")
    print("Q:           Increase left rotor thrust")
    print("E:           Increase right rotor thrust")
    print("SPACE:       Reset thrust to hover")
    print("R:           Reset environment")
    print("ESC:         Exit")
    print("================\n")

    # Test for multiple episodes
    episode = 1
    obs, info = env.reset()
    episode_reward = 0
    episode_steps = 0
    paused = False

    running = True
    while running:
        # Handle pygame events
        keys_pressed = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset environment
                    print(f"\nResetting environment (Episode {episode} -> {episode + 1})")
                    print(f"Previous episode - Steps: {episode_steps}, Total Reward: {episode_reward:.2f}")
                    episode += 1
                    obs, info = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    left_thrust = 0.0
                    right_thrust = 0.0
                elif event.key == pygame.K_SPACE:
                    # Reset thrust to hover
                    left_thrust = 0.0
                    right_thrust = 0.0
                    print("Thrust reset to hover")
                elif event.key == pygame.K_p:
                    # Pause/unpause
                    paused = not paused
                    print(f"{'Paused' if paused else 'Unpaused'}")

        if not paused:
            # Handle continuous key presses for thrust control
            if keys_pressed[pygame.K_LEFT]:
                left_thrust -= thrust_increment * 0.5
            if keys_pressed[pygame.K_RIGHT]:
                right_thrust -= thrust_increment * 0.5
            if keys_pressed[pygame.K_UP]:
                left_thrust += thrust_increment * 0.5
                right_thrust += thrust_increment * 0.5
            if keys_pressed[pygame.K_DOWN]:
                left_thrust -= thrust_increment * 0.5
                right_thrust -= thrust_increment * 0.5
            if keys_pressed[pygame.K_q]:
                left_thrust += thrust_increment * 0.5
            if keys_pressed[pygame.K_e]:
                right_thrust += thrust_increment * 0.5

            # Clamp thrust values to reasonable range
            left_thrust = np.clip(left_thrust, -1.0, 1.0)
            right_thrust = np.clip(right_thrust, -1.0, 1.0)

            # Create action from thrust values
            action = np.array([left_thrust, right_thrust])

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
        info_text = [
            f'Episode: {episode}',
            f'Step: {episode_steps}',
            f'Reward: {reward:.2f}' if not paused else 'Reward: PAUSED',
            f'Total: {episode_reward:.2f}',
            f'Phi / phi_dot {round(env.state[8],2)} / {round(env.state[9],2)}',

            f'Balanced Time: {env.total_time_balanced}'
        ]

        for i, text in enumerate(info_text):
            ax.text(0.02, 0.98 - i * 0.06, text,
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add control info
        control_text = [
            f'Left Thrust: {left_thrust:.2f}',
            f'Right Thrust: {right_thrust:.2f}',
            f'Total Thrust: {left_thrust + right_thrust:.2f}',
            'PAUSED' if paused else 'RUNNING'
        ]

        for i, text in enumerate(control_text):
            ax.text(0.98, 0.98 - i * 0.06, text,
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round',
                              facecolor='yellow' if paused and i == 3 else 'white',
                              alpha=0.8))

        ax.set_title(f'QuadPole2D - Manual Control (Episode {episode})')
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

        # Add control instructions overlay
        font = pygame.font.Font(None, 24)
        instructions = [
            "Controls: ↑↓←→ for thrust, Q/E for individual rotors",
            "SPACE: Reset thrust, R: Reset env, P: Pause, ESC: Exit"
        ]

        for i, instruction in enumerate(instructions):
            text_surface = font.render(instruction, True, (255, 255, 255))
            text_rect = text_surface.get_rect()
            text_rect.x = 10
            text_rect.y = height - 60 + i * 25

            # Black background for text
            pygame.draw.rect(screen, (0, 0, 0), text_rect.inflate(10, 5))
            screen.blit(text_surface, text_rect)

        pygame.display.flip()

        # Close matplotlib figure to free memory
        plt.close(fig)

        # Control frame rate
        clock.tick(50)  # 50 FPS for responsive controls

        # Check if episode is done
        if not paused and (terminated or truncated):
            print(f"\nEpisode {episode} finished!")
            print(f"Steps: {episode_steps}")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            print(f"Out of Bounds: {env.out_of_bounds()}")
            print(f"Time Balanced: {info.get('time_balanced', 0):.2f}s")
            print("Press 'R' to start new episode or ESC to exit")

    pygame.quit()
    print(f"\nFinal Stats:")
    print(f"Episodes completed: {episode}")
    print(f"Total balanced time: {env.total_time_balanced}")


def test_quadrotor_env_auto():
    """Original automatic test with random actions"""

    # Initialize pygame
    pygame.init()

    # Set up display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("QuadPole2D Environment Test - Auto Mode")
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

            ax.set_title(f'QuadPole2D - Episode {episode + 1} (Auto)')
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

    pygame.quit()


if __name__ == "__main__":

    config_filename = './configs/config.json'

    print("QuadPole2D Environment Test")
    print("Choose mode:")
    print("1. Human Control (default)")
    print("2. Automatic (random actions)")

    choice = 1#input("Enter choice (1 or 2): ").strip()

    if choice == "2":
        print("Starting automatic test with random actions...")
        try:
            test_quadrotor_env_auto()
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
            pygame.quit()
        except Exception as e:
            print(f"Error occurred: {e}")
            pygame.quit()
            raise
    else:
        print("Starting human control test...")
        print("Get ready to control the quadrotor!")
        try:
            test_quadrotor_env_human(config_filename)
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
            pygame.quit()
        except Exception as e:
            print(f"Error occurred: {e}")
            pygame.quit()
            raise