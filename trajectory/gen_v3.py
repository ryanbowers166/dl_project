import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import List, Tuple, Dict

# =============================================================================
# MPC EXPERT TRAJECTORY GENERATOR FOR PPO FINE-TUNING
# =============================================================================

def generate_expert_trajectories(num_trajectories: int = 50, 
                                save_path: str = 'expert_trajectories.npz') -> Dict:
    """
    Generate multiple expert trajectories with different initial conditions and goals
    for PPO fine-tuning data.
    """
    
    # Base parameters
    T = 40
    dt = 0.1
    nx = 10
    nu = 2
    
    # Constraints (slightly relaxed for diverse trajectories)
    POSITION_LIMIT = 4.0
    VELOCITY_LIMIT = 4.0
    ANGLE_LIMIT = 60.0
    PENDULUM_LIMIT = 90.0
    ANGULAR_VEL_LIMIT = 8.0
    
    # Cost weights
    w_goal = 100.0
    w_upright = 150.0
    w_effort = 1.0
    w_velocity = 8.0
    w_smooth = 3.0
    w_terminal = 15.0
    
    # Physical parameters
    params = dict(dt=dt, mq=1.0, mp=0.3, Lq=0.1, Lp=0.5, I=0.02, g=9.81)
    u_max = 20.0
    u_min = 0.1
    
    def dynamics(dt, mq, mp, Lq, Lp, I, g):
        x = ca.MX.sym('x', nx)
        u = ca.MX.sym('u', nu)

        x_pos, z_pos, vx, vz = x[0], x[1], x[2], x[3]
        s_theta, c_theta, theta_dot = x[4], x[5], x[6]
        s_phi, c_phi, phi_dot = x[7], x[8], x[9]
        u1, u2 = u[0], u[1]

        eps = 1e-8
        F = u1 + u2
        M = mq + mp

        # Drone dynamics
        ddtheta = (Lq / I) * (u2 - u1)
        
        # Pendulum dynamics (upright equilibrium)
        phi_angle = ca.atan2(s_phi, c_phi + eps)
        phi_dev = phi_angle - np.pi
        theta_angle = ca.atan2(s_theta, c_theta + eps)
        ddphi = (g / Lp) * phi_dev + (F / (M * Lp)) * theta_angle
        
        # Translational dynamics
        ddx = -theta_angle * F / M
        ddz = F / M - g

        # Integration with damping
        damping = 0.92
        vx_new = damping * (vx + ddx * dt)
        vz_new = damping * (vz + ddz * dt)
        theta_dot_new = damping * (theta_dot + ddtheta * dt)
        phi_dot_new = damping * (phi_dot + ddphi * dt)

        x_new = x_pos + vx_new * dt
        z_new = z_pos + vz_new * dt

        theta_new = ca.atan2(s_theta, c_theta + eps) + theta_dot_new * dt
        phi_new = ca.atan2(s_phi, c_phi + eps) + phi_dot_new * dt
        
        s_theta_new = ca.sin(theta_new)
        c_theta_new = ca.cos(theta_new)
        s_phi_new = ca.sin(phi_new)
        c_phi_new = ca.cos(phi_new)

        x_next = ca.vertcat(
            x_new, z_new, vx_new, vz_new,
            s_theta_new, c_theta_new, theta_dot_new,
            s_phi_new, c_phi_new, phi_dot_new
        )

        return ca.Function("f", [x, u], [x_next])

    f_dyn = dynamics(**params)
    
    # Setup optimization problem once
    X = ca.MX.sym('X', nx, T+1)
    U = ca.MX.sym('U', nu, T)
    x0_param = ca.MX.sym('x0', nx)
    xg_param = ca.MX.sym('xg', 2)

    cost = 0
    g = []

    # Initial condition (parameterized)
    g.append(X[:, 0] - x0_param)

    # Stage costs
    for t in range(T):
        xt = X[:, t]
        ut = U[:, t]
        xt_next = X[:, t+1]

        g.append(f_dyn(xt, ut) - xt_next)

        pos_error = xt[0:2] - xg_param
        vel = xt[2:4]
        pend_upright_error = 1.0 + xt[8]
        
        hover_thrust = params['g'] * (params['mq'] + params['mp']) / 2
        control_deviation = ut - hover_thrust
        
        cost += w_goal * ca.sumsqr(pos_error)
        cost += w_upright * pend_upright_error**2
        cost += w_effort * ca.sumsqr(control_deviation)
        cost += w_velocity * ca.sumsqr(vel)
        
        if t > 0:
            control_change = ut - U[:, t-1]
            cost += w_smooth * ca.sumsqr(control_change)

    # Terminal cost
    final_pos_error = X[0:2, -1] - xg_param
    final_pend_error = 1.0 + X[8, -1]
    final_vel = X[2:4, -1]

    cost += w_terminal * w_goal * ca.sumsqr(final_pos_error)
    cost += w_terminal * w_upright * final_pend_error**2
    cost += w_terminal * w_velocity * ca.sumsqr(final_vel)

    # Setup NLP with parameters
    g_flat = ca.vertcat(*g)
    opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    params_vec = ca.vertcat(x0_param, xg_param)

    nlp = {'x': opt_vars, 'f': cost, 'g': g_flat, 'p': params_vec}

    opts = {
        'ipopt.print_level': 0,
        'ipopt.max_iter': 800,
        'ipopt.tol': 1e-5,
        'ipopt.acceptable_tol': 1e-3,
        'print_time': 0
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    # Setup bounds (same for all problems)
    hover_thrust = (params['mq'] + params['mp']) * params['g'] / 2
    x0_guess = np.zeros((nx, T+1))
    u0_guess = np.ones((nu, T)) * hover_thrust
    
    # Initialize guess arrays
    for k in range(T+1):
        x0_guess[4, k] = 0.0    # s_theta
        x0_guess[5, k] = 1.0    # c_theta
        x0_guess[7, k] = 0.0    # s_phi (upright)
        x0_guess[8, k] = -0.95  # c_phi (upright)

    initial_guess = np.concatenate([x0_guess.flatten(), u0_guess.flatten()])
    
    # Bounds
    lbx = -np.inf * np.ones_like(initial_guess)
    ubx = np.inf * np.ones_like(initial_guess)
    
    for i in range(T+1):
        idx = i * nx
        lbx[idx:idx+2] = -POSITION_LIMIT
        ubx[idx:idx+2] = POSITION_LIMIT
        lbx[idx+2:idx+4] = -VELOCITY_LIMIT
        ubx[idx+2:idx+4] = VELOCITY_LIMIT
        
        # Drone angle bounds
        max_sin_theta = np.sin(np.radians(ANGLE_LIMIT))
        min_cos_theta = np.cos(np.radians(ANGLE_LIMIT))
        lbx[idx+4] = -max_sin_theta
        ubx[idx+4] = max_sin_theta
        lbx[idx+5] = min_cos_theta
        ubx[idx+5] = 1.0
        
        # Pendulum bounds
        upright_angle = np.pi
        max_dev = np.radians(PENDULUM_LIMIT)
        min_phi = upright_angle - max_dev
        max_phi = upright_angle + max_dev
        
        lbx[idx+7] = min(np.sin(min_phi), np.sin(max_phi))
        ubx[idx+7] = max(np.sin(min_phi), np.sin(max_phi))
        lbx[idx+8] = min(np.cos(min_phi), np.cos(max_phi))
        ubx[idx+8] = max(np.cos(min_phi), np.cos(max_phi))
        
        lbx[idx+6] = -ANGULAR_VEL_LIMIT
        ubx[idx+6] = ANGULAR_VEL_LIMIT
        lbx[idx+9] = -ANGULAR_VEL_LIMIT
        ubx[idx+9] = ANGULAR_VEL_LIMIT

    lbx[nx*(T+1):] = u_min
    ubx[nx*(T+1):] = u_max
    lbg = ubg = np.zeros(g_flat.shape)
    
    # Generate diverse scenarios
    print(f"Generating {num_trajectories} expert trajectories...")
    
    expert_data = []
    successful_trajectories = 0
    failed_optimizations = 0
    
    for traj_idx in range(num_trajectories):
        # Generate random initial condition (near upright)
        x0_np = np.zeros(nx)
        
        # Random initial position
        x0_np[0] = np.random.uniform(-2.0, 2.0)  # x position
        x0_np[1] = np.random.uniform(0.5, 2.5)   # z position
        
        # Small initial velocities
        x0_np[2] = np.random.uniform(-0.5, 0.5)  # vx
        x0_np[3] = np.random.uniform(-0.5, 0.5)  # vz
        
        # Drone nearly level with small perturbation
        small_theta = np.random.uniform(-0.2, 0.2)
        x0_np[4] = np.sin(small_theta)  # s_theta
        x0_np[5] = np.cos(small_theta)  # c_theta
        x0_np[6] = np.random.uniform(-0.5, 0.5)  # theta_dot
        
        # Pendulum near upright with perturbation
        phi_perturbation = np.random.uniform(-0.3, 0.3)  # ±17 degrees
        phi_initial = np.pi + phi_perturbation
        x0_np[7] = np.sin(phi_initial)  # s_phi
        x0_np[8] = np.cos(phi_initial)  # c_phi
        x0_np[9] = np.random.uniform(-1.0, 1.0)  # phi_dot
        
        # Random goal position
        xg_np = np.array([
            np.random.uniform(-2.5, 2.5),  # goal x
            np.random.uniform(0.5, 3.0)    # goal z
        ])
        
        # Update initial guess based on scenario
        for k in range(T+1):
            alpha = k / T
            x0_guess[0, k] = x0_np[0] + alpha * (xg_np[0] - x0_np[0])
            x0_guess[1, k] = x0_np[1] + alpha * (xg_np[1] - x0_np[1])
            x0_guess[2, k] = (xg_np[0] - x0_np[0]) / (T * dt) * 0.2
            x0_guess[3, k] = (xg_np[1] - x0_np[1]) / (T * dt) * 0.2
        
        initial_guess = np.concatenate([x0_guess.flatten(), u0_guess.flatten()])
        
        # Set parameters
        params_val = np.concatenate([x0_np, xg_np])
        
        # Solve
        try:
            solution = solver(x0=initial_guess, lbx=lbx, ubx=ubx, 
                            lbg=lbg, ubg=ubg, p=params_val)
            
            if solver.stats()['success']:
                # Extract solution
                sol = solution['x'].full().flatten()
                X_sol = sol[:nx*(T+1)].reshape((nx, T+1))
                U_sol = sol[nx*(T+1):].reshape((nu, T))
                
                # Check trajectory quality
                final_pos_error = np.linalg.norm(X_sol[0:2, -1] - xg_np)
                phi_traj = np.arctan2(X_sol[7, :], X_sol[8, :])
                max_pend_dev = np.max(np.abs(phi_traj - np.pi))
                
                # Only keep good trajectories
                if final_pos_error < 0.5 and max_pend_dev < np.radians(120):
                    # Convert to state-action pairs
                    for t in range(T):
                        state = X_sol[:, t].copy()
                        action = U_sol[:, t].copy()
                        next_state = X_sol[:, t+1].copy()
                        
                        # Simple reward function
                        pos_error_t = np.linalg.norm(state[0:2] - xg_np)
                        pend_upright_error = 1.0 + state[8]  # cos(phi) penalty
                        reward = -pos_error_t - 10.0 * pend_upright_error**2
                        
                        expert_data.append({
                            'state': state,
                            'action': action,
                            'next_state': next_state,
                            'reward': reward,
                            'trajectory_id': successful_trajectories,
                            'timestep': t
                        })
                    
                    successful_trajectories += 1
                    if successful_trajectories % 10 == 0:
                        print(f"  Generated {successful_trajectories} successful trajectories...")
                        
            else:
                failed_optimizations += 1
                
        except Exception as e:
            failed_optimizations += 1
            continue
    
    print(f"✅ Successfully generated {successful_trajectories} trajectories")
    print(f"❌ Failed optimizations: {failed_optimizations}")
    print(f"📊 Total state-action pairs: {len(expert_data)}")
    
    # Convert to arrays for easier handling
    states = np.array([d['state'] for d in expert_data])
    actions = np.array([d['action'] for d in expert_data])
    next_states = np.array([d['next_state'] for d in expert_data])
    rewards = np.array([d['reward'] for d in expert_data])
    trajectory_ids = np.array([d['trajectory_id'] for d in expert_data])
    timesteps = np.array([d['timestep'] for d in expert_data])
    
    # Save data
    np.savez_compressed(save_path,
                       states=states,
                       actions=actions,
                       next_states=next_states,
                       rewards=rewards,
                       trajectory_ids=trajectory_ids,
                       timesteps=timesteps,
                       num_trajectories=successful_trajectories,
                       sequence_length=T)
    
    print(f"💾 Expert data saved to {save_path}")
    
    # Basic statistics
    print(f"\n=== DATASET STATISTICS ===")
    print(f"State dimensionality: {states.shape[1]}")
    print(f"Action dimensionality: {actions.shape[1]}")
    print(f"Total samples: {states.shape[0]}")
    print(f"Action range: [{actions.min():.2f}, {actions.max():.2f}]")
    print(f"Average reward: {rewards.mean():.3f} ± {rewards.std():.3f}")
    
    return {
        'states': states,
        'actions': actions,
        'next_states': next_states,
        'rewards': rewards,
        'trajectory_ids': trajectory_ids,
        'timesteps': timesteps,
        'metadata': {
            'num_trajectories': successful_trajectories,
            'sequence_length': T,
            'state_dim': nx,
            'action_dim': nu,
            'dt': dt
        }
    }

def visualize_expert_trajectories(data_path: str = 'expert_trajectories.npz', num_to_plot: int = 5):
    """Visualize a few expert trajectories for verification"""
    
    data = np.load(data_path)
    states = data['states']
    actions = data['actions']
    trajectory_ids = data['trajectory_ids']
    
    unique_trajs = np.unique(trajectory_ids)[:num_to_plot]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Expert Trajectory Samples', fontsize=14)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_trajs)))
    
    for i, traj_id in enumerate(unique_trajs):
        mask = trajectory_ids == traj_id
        traj_states = states[mask]
        traj_actions = actions[mask]
        
        # Position trajectory
        axes[0,0].plot(traj_states[:, 0], traj_states[:, 1], 
                      color=colors[i], alpha=0.7, label=f'Traj {traj_id}')
        axes[0,0].scatter(traj_states[0, 0], traj_states[0, 1], 
                         color=colors[i], s=50, marker='o')
        axes[0,0].scatter(traj_states[-1, 0], traj_states[-1, 1], 
                         color=colors[i], s=50, marker='x')
    
    axes[0,0].set_xlabel('X Position [m]')
    axes[0,0].set_ylabel('Z Position [m]')
    axes[0,0].set_title('Position Trajectories')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Pendulum angles
    for i, traj_id in enumerate(unique_trajs):
        mask = trajectory_ids == traj_id
        traj_states = states[mask]
        phi_traj = np.degrees(np.arctan2(traj_states[:, 7], traj_states[:, 8]))
        time_vec = np.arange(len(traj_states)) * 0.1
        
        axes[0,1].plot(time_vec, phi_traj, color=colors[i], alpha=0.7)
    
    axes[0,1].axhline(y=180, color='black', linestyle='--', alpha=0.5, label='Upright')
    axes[0,1].set_xlabel('Time [s]')
    axes[0,1].set_ylabel('Pendulum Angle [deg]')
    axes[0,1].set_title('Pendulum Angles')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Control inputs
    for i, traj_id in enumerate(unique_trajs):
        mask = trajectory_ids == traj_id
        traj_actions = actions[mask]
        time_vec = np.arange(len(traj_actions)) * 0.1
        
        axes[1,0].plot(time_vec, traj_actions[:, 0], color=colors[i], alpha=0.7, linestyle='-')
        axes[1,1].plot(time_vec, traj_actions[:, 1], color=colors[i], alpha=0.7, linestyle='-')
    
    axes[1,0].set_xlabel('Time [s]')
    axes[1,0].set_ylabel('Thrust [N]')
    axes[1,0].set_title('Left Thruster (u1)')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].set_xlabel('Time [s]')
    axes[1,1].set_ylabel('Thrust [N]')
    axes[1,1].set_title('Right Thruster (u2)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate expert trajectories
    expert_data = generate_expert_trajectories(
        num_trajectories=100,  # Generate 100 diverse trajectories
        save_path='expert_trajectories.npz'
    )
    
    # Visualize some trajectories
    visualize_expert_trajectories('expert_trajectories.npz', num_to_plot=5)
    
    print("\n=== READY FOR PPO FINE-TUNING ===")
    print("Use the generated 'expert_trajectories.npz' file for:")
    print("1. Behavior cloning pretraining")
    print("2. PPO fine-tuning with expert demonstrations")
    print("3. Imitation learning evaluation")
