# Toward Safe Quadrotor Navigation: Hybrid Learning-Based Control with Predictive Safety Mechanisms

This repository accompanies our project on safe and reliable quadrotor navigation with suspended payloads. Our system integrates deep reinforcement learning, trajectory generation, predictive modeling, and safety-triggered fallback control to ensure robust performance in the face of dynamic disturbances and potential failures.

We combine learning-based and classical control methods to design a modular and interpretable control framework for aerial robots operating in uncertain environments.

---

## 🧠 Project Overview

We present a hybrid control pipeline composed of four main components:

1. **Goal-Conditioned PPO Controller**
   A reinforcement learning agent trained using Proximal Policy Optimization (PPO) to navigate 2D waypoints while balancing a suspended pendulum payload.

2. **Trajectory Generator**
   A supervised model trained on optimal trajectories from a classical planner, generating full state-action paths between arbitrary start and goal states.

3. **Predictive Stability Forecasting**
   An LSTM-based classifier that anticipates unsafe behavior by predicting motion patterns from recent state-action history.

4. **Anomaly-Triggered Control Mode Switching (ATCMS)**
   A rule-based safety module that activates fallback control (e.g. LQR or constant thrust) when instability is detected or predicted.

---

## 🚁 Environment

All components are developed around a 2D Gymnasium environment (`QuadPole2D`) that simulates a quadrotor carrying a pendulum payload. The agent must reach goal waypoints while keeping the pendulum upright and stable.

Key environment features:

* 12D observation space: quadrotor position, velocity, pitch, pendulum angle and angular velocity, and waypoint goal position.
* 2D action space: rotor thrusts
* Configurable reward shaping and curriculum learning
* Compatible with Stable-Baselines3

---

## 📁 Repository Structure

```
├── results/                    # Plots and other results from experiments
├── configs/                    # .json config files for training
├── saved_models/               # Pretrained RL models that can be loaded for evaluation
├── evaluate_agents.py          # Load a group of trained agents to evaluate and compare them
├── quadrotor_env.py            # Contians the QuadPole2D environment class
├── run_hybrid_agent.py         # Run and evaluate a hybrid RL + LQR agent
├── run_rl_agent.py             # Run and evaluate a RL agent
├── test_env_human.py           # Allows the user to play the environment using keyboard inputs
├── train.py                    # Main training script for the RL policies
```

## Acknowledgments

This work was developed as part of the CS 7643: Deep Learning course at Georgia Tech. We thank Dr. Zzolt Kira and the TAs for their guidance.