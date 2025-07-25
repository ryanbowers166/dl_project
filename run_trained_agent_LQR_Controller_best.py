
import os
import json
import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pandas as pd

from stable_baselines3 import PPO
from scipy.linalg import solve_continuous_are, solve_discrete_are
from numpy.linalg import LinAlgError

from train import QuadPole2DWrapper, QuadPole2D


# === Utility functions ===
def parse_obs(obs):
    return obs[0:3], obs[3:6], obs[6:9], obs[9:12]

MAX_TILT = np.radians(30)
MAX_VEL = 5.0
MAX_ANG_VEL = np.radians(300)
DISTURB_THRESH = 1.5

def anomaly_detector(obs, prev_obs):
    if prev_obs is None:
        return None
    pos, vel, orient, ang = parse_obs(obs)
    _, prev_vel, _, prev_ang = parse_obs(prev_obs)
    if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
        return "severe"
    if np.any(np.abs(orient[:2]) > MAX_TILT):
        return "severe"
    if np.linalg.norm(vel) > MAX_VEL or np.linalg.norm(ang) > MAX_ANG_VEL:
        return "mild"
    if np.linalg.norm(vel - prev_vel) > DISTURB_THRESH or np.linalg.norm(ang - prev_ang) > DISTURB_THRESH:
        return "mild"
    return None

def obs_to_state(obs):
    pos, vel, orient, ang = parse_obs(obs)
    x, z = pos[0], pos[2]
    vx, vz = vel[0], vel[2]
    s_th, c_th = orient[0], orient[1]
    th_dot = ang[0]
    s_ph = orient[2]
    c_ph = np.sqrt(max(0, 1 - s_ph**2))
    ph_dot = ang[2]
    return np.array([x, z, vx, vz, s_th, c_th, th_dot, s_ph, c_ph, ph_dot])

def dyn_fn(x, u):
    qp = QuadPole2D(None, None)
    return np.array(qp._dynamics(x.tolist(), u.tolist()), dtype=np.float64)

def linearize_fd(f, x0, u0, eps=1e-5):
    n, m = x0.size, u0.size
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    f0 = f(x0, u0)
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        A[:, i] = (f(x0 + dx, u0) - f0) / eps
    for j in range(m):
        du = np.zeros(m); du[j] = eps
        B[:, j] = (f(x0, u0 + du) - f0) / eps
    return A, B

def safety_policy(obs, K, x0):
    x = obs_to_state(obs)
    x_err = x - x0
    u = -K.dot(x_err)
    u = np.clip(u, 0, 20)
    return u.astype(np.float32)


# === Main testing pipeline ===
def test_agent_combined(config, render_mode, model_path,
                        mode="intervene", n_episodes=3,
                        manual_goal_position=None, live_plot=False,save_summary_plot=False):
    pygame.init()
    width, height = 800, 1000
    screen = pygame.display.set_mode((width, height))
    model = PPO.load(model_path)
    env = QuadPole2DWrapper(config, render_mode, manual_goal_position)

    K = None
    if mode == "intervene":
        state_dim, act_dim = 10, 2
        x0 = np.zeros(state_dim)
        x0[5] = -1.0 / 10
        x0[8] = 1.0 / 30
        qp = QuadPole2D(None, None)
        u0 = np.array([9.81 * 0.5 * (qp.mq + qp.mp)] * act_dim)

        print("Linearizing dynamics at nominal upright position...")
        A, B = linearize_fd(dyn_fn, x0, u0)
        Q = np.diag([10.0] * state_dim)
        R = np.eye(act_dim) * 0.1

        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            print("Continuous-time LQR succeeded.")
        except LinAlgError:
            print("Continuous LQR failed, trying discrete-time...")
            try:
                dt = qp.timestep
                Ad = np.eye(state_dim) + A * dt
                Bd = B * dt
                P = solve_discrete_are(Ad, Bd, Q, R)
                K = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
                print("Discrete-time LQR succeeded.")
            except LinAlgError:
                print("LQR failed; safety policy disabled.")
                K = None

    all_episode_metrics = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        prev_obs = None
        done = False
        step = 0
        in_safety = False
        lqr_start_step = None
        total_reward = 0

        save_dir = f"saved_plots/episode_{ep+1}"
        os.makedirs(save_dir, exist_ok=True)

        logs = {k: [] for k in [
            "tilt_deg", "lin_vel", "ang_vel",
            "pos_x", "pos_y", "pos_z",
            "anomaly", "control_mode"
        ]}

        while not done:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_q):
                    pygame.quit()
                    return

            anomaly = anomaly_detector(obs, prev_obs)
            if not in_safety and anomaly:
                in_safety = True
                lqr_start_step = step
                print(f"[Episode {ep+1}, Step {step}] → LQR takeover due to {anomaly}")

            if mode == "intervene" and K is not None and in_safety:
                action = safety_policy(obs, K, x0)
            else:
                action = model.predict(obs, deterministic=True)[0]
                if anomaly and mode != "intervene":
                    print(f"[Monitor] Anomaly: {anomaly}")

            logs["control_mode"].append("LQR" if in_safety else "Agent")

            pos, vel, orient, ang = parse_obs(obs)
            logs["tilt_deg"].append(np.degrees(np.linalg.norm(orient[:2])))
            logs["lin_vel"].append(np.linalg.norm(vel))
            logs["ang_vel"].append(np.linalg.norm(ang))
            logs["pos_x"].append(pos[0])
            logs["pos_y"].append(pos[1])
            logs["pos_z"].append(pos[2])
            logs["anomaly"].append(anomaly)

            prev_obs = obs.copy()
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            done = term or trunc

            # Live pygame rendering
            if live_plot:
                fig, axs = plt.subplots(4, 1, figsize=(8, 14),
                                        gridspec_kw={'height_ratios': [2, 1, 1, 0.5]})
                ax1, ax2, ax3, ax4 = axs
                ax1.set_facecolor('lightblue')
                #env.env.render(ax1, observation=obs)
                env.env.render(screen)
                label = f"Anomaly: {anomaly}" if anomaly else "Anomaly: None"
                color = 'red' if anomaly == "severe" else 'orange' if anomaly == "mild" else 'green'
                ax1.text(0.02, 0.95, label, color=color, transform=ax1.transAxes,
                         bbox=dict(facecolor='white', edgecolor=color))
                ax1.text(0.02, 0.90, f"Ep {ep+1}, Step {step+1}", transform=ax1.transAxes)
                ax1.set_title("Simulation")

                t = np.arange(len(logs["tilt_deg"]))
                ax2.plot(t, logs["tilt_deg"], label='Tilt (°)', color='purple')
                ax2.plot(t, logs["lin_vel"], label='Lin Vel', color='blue')
                ax2.plot(t, logs["ang_vel"], label='Ang Vel', color='green')
                for i, a in enumerate(logs["anomaly"]):
                    if a == "severe":
                        ax2.axvline(i, color='red', linestyle='--', alpha=0.3)
                    elif a == "mild":
                        ax2.axvline(i, color='orange', linestyle='--', alpha=0.2)
                ax2.set_title("State Metrics"); ax2.legend(); ax2.grid(alpha=0.4)

                ax3.plot(t, logs["pos_x"], label='X', color='r')
                ax3.plot(t, logs["pos_y"], label='Y', color='g')
                ax3.plot(t, logs["pos_z"], label='Z', color='b')
                ax3.set_title("Position")
                ax3.legend()
                ax3.grid(alpha=0.4)

                mode_numeric = [1 if m == "LQR" else 0 for m in logs["control_mode"]]
                ax4.step(t, mode_numeric, where='post', color='black', linewidth=2)
                ax4.set_yticks([0, 1])
                ax4.set_yticklabels(["Agent", "LQR"])
                ax4.set_title("Control Mode")
                ax4.set_xlabel("Timestep")
                ax4.grid(alpha=0.4)

                # Draw on pygame screen
                canvas = agg.FigureCanvasAgg(fig)
                canvas.draw()
                surf = pygame.image.frombuffer(canvas.buffer_rgba(),
                                               canvas.get_width_height(), 'RGBA')
                screen.blit(pygame.transform.scale(surf, (width, height)), (0, 0))
                plt.close(fig)
            pygame.display.flip()


            step += 1
        reached_times = [t for g, t in env.env.goal_reach_times if t != -1]
        goal_1_time = reached_times[0] if len(reached_times) > 0 else -1
        goal_2_time = reached_times[1] if len(reached_times) > 1 else -1
        goal_3_time = reached_times[2] if len(reached_times) > 2 else -1

        # Episode finished — print summary
        lqr_duration = sum(1 for m in logs["control_mode"] if m == "LQR")
        num_anomalies = sum(1 for a in logs["anomaly"] if a)
        first_anomaly_step = next((i for i, a in enumerate(logs["anomaly"]) if a), None)

        print(f"\n--- Episode {ep+1} Summary ---")
        print(f"Total Steps: {step}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Anomalies Detected: {num_anomalies}")
        print(f"First Anomaly at Step: {first_anomaly_step}")
        print(f"LQR Started at Step: {lqr_start_step}")
        print(f"LQR Active for: {lqr_duration} steps")
        print(f"Max Tilt: {max(logs['tilt_deg']):.2f}°")
        print(f"Mean Linear Velocity: {np.mean(logs['lin_vel']):.2f}")
        print(f"Mean Angular Velocity: {np.mean(logs['ang_vel']):.2f}\n")

        all_episode_metrics.append({
            "episode": ep + 1,
            "total_steps": step,
            "total_reward": total_reward,
            "num_anomalies": num_anomalies,
            "first_anomaly_step": first_anomaly_step,
            "lqr_start_step": lqr_start_step,
            "lqr_duration": lqr_duration,
            "max_tilt_deg": max(logs["tilt_deg"]),
            "mean_lin_vel": np.mean(logs["lin_vel"]),
            "mean_ang_vel": np.mean(logs["ang_vel"]),
            "time_balanced": env.env.total_time_balanced,
            "goal1_reach_time": goal_1_time,
            "goal2_reach_time": goal_2_time,
            #"goal3_reach_time": goal_3_time

        })
        # === Save summary plot of entire episode ===
        if save_summary_plot:
            fig, axs = plt.subplots(4, 1, figsize=(8, 14),
                                    gridspec_kw={'height_ratios': [2, 1, 1, 0.5]})
            ax1, ax2, ax3, ax4 = axs
            ax1.set_facecolor('lightblue')
            # Render last observation as a snapshot
            env.env.render(ax1, observation=obs)
            label = f"Anomaly: {logs['anomaly'][-1]}" if logs['anomaly'][-1] else "Anomaly: None"
            color = 'red' if logs['anomaly'][-1] == "severe" else 'orange' if logs['anomaly'][-1] == "mild" else 'green'
            ax1.text(0.02, 0.95, label, color=color, transform=ax1.transAxes,
                     bbox=dict(facecolor='white', edgecolor=color))
            ax1.text(0.02, 0.90, f"Episode {ep+1} Summary", transform=ax1.transAxes)
            ax1.set_title("Simulation Snapshot")

            t = np.arange(step)
            ax2.plot(t, logs["tilt_deg"], label='Tilt (°)', color='purple')
            ax2.plot(t, logs["lin_vel"], label='Lin Vel', color='blue')
            ax2.plot(t, logs["ang_vel"], label='Ang Vel', color='green')
            for i, a in enumerate(logs["anomaly"]):
                if a == "severe":
                    ax2.axvline(i, color='red', linestyle='--', alpha=0.3)
                elif a == "mild":
                    ax2.axvline(i, color='orange', linestyle='--', alpha=0.2)
            ax2.set_title("State Metrics")
            ax2.legend()
            ax2.grid(alpha=0.4)

            ax3.plot(t, logs["pos_x"], label='X', color='r')
            ax3.plot(t, logs["pos_y"], label='Y', color='g')
            ax3.plot(t, logs["pos_z"], label='Z', color='b')
            ax3.set_title("Position")
            ax3.legend()
            ax3.grid(alpha=0.4)

            mode_numeric = [1 if m == "LQR" else 0 for m in logs["control_mode"]]
            ax4.step(t, mode_numeric, where='post', color='black', linewidth=2)
            ax4.set_yticks([0, 1])
            ax4.set_yticklabels(["Agent", "LQR"])
            ax4.set_title("Control Mode")
            ax4.set_xlabel("Timestep")
            ax4.grid(alpha=0.4)

            summary_plot_path = os.path.join(save_dir, f"episode_{ep+1}_summary.png")
            plt.tight_layout()
            fig.savefig(summary_plot_path)
            plt.close(fig)
            print(f"Saved episode summary plot to {summary_plot_path}")

    pygame.quit()

    # Save all metrics to CSV
    df_metrics = pd.DataFrame(all_episode_metrics)
    df_metrics.to_csv("evaluation_metrics.csv", index=False)
    print("Saved evaluation metrics to evaluation_metrics.csv")
    return all_episode_metrics


# === Script entry point ===
if __name__ == "__main__":
    with open('./configs/config_v4.json') as f:
        cfg = json.load(f)
    cfg['config_filename'] = './configs/config_v4.json'

    test_agent_combined(
        config=cfg,
        render_mode='human',
        model_path='./saved_models/gamma sweep/0607_1042_poscos20_gamma999',
        mode="intervene",         # or "monitor"
        n_episodes=3,
        manual_goal_position="dynamic-1"
    )

