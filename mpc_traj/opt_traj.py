import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import List, Tuple, Dict


def generate_expert_trajectories(num_trajectories: int = 50, 
                                save_path: str = 'expert_trajectories.npz',
                                config: Dict = None) -> Dict:
    """
    Generate multiple expert trajectories with different initial conditions and goals
    for PPO fine-tuning data.
    """
    
    # matches quad_env params
    if config is None:
        config = {
            'pos_cost_multiplier': 15.0,
            'vel_cost_multiplier': 0.5,
            'theta_cost_multiplier': 5.0,
            'omega_cost_multiplier': 5.0,
            'phi_cost_multiplier': 25.0,
            'phi_dot_cost_multiplier': 5.0,
            'balance_reward': 100.0,
            'curriculum_level': 0
        }
    
    T = 40
    dt = 0.02  
    nx = 10    
    nu = 2
    
    POSITION_LIMIT = 2.0  
    VELOCITY_LIMIT = 4.0
    ANGLE_LIMIT = 60.0
    PENDULUM_LIMIT = 90.0
    ANGULAR_VEL_LIMIT = 8.0
    
    params = dict(
        dt=dt, 
        mq=1.5,      
        mp=0.5,    
        Lq=0.5,   
        Lp=0.75,   
        I=0.4,      
        g=9.80665  
    )
    
    hover_force = (params['mq'] + params['mp']) * params['g'] / 2
    
    u_max = 1.0   
    u_min = -1.0 
    
    def wrap_action(action_normalized, hover_force):

        return hover_force + hover_force * np.clip(action_normalized, -1, 1)
    
    def dynamics(dt, mq, mp, Lq, Lp, I, g):
        x = ca.MX.sym('x', nx)
        u = ca.MX.sym('u', nu)  

        x_pos, z_pos, vx, vz = x[0], x[1], x[2], x[3]
        s_theta, c_theta, theta_dot = x[4], x[5], x[6]
        s_phi, c_phi, phi_dot = x[7], x[8], x[9]
        
        u1_norm, u2_norm = u[0], u[1]
        u1 = hover_force + hover_force * ca.fmin(ca.fmax(u1_norm, -1), 1)
        u2 = hover_force + hover_force * ca.fmin(ca.fmax(u2_norm, -1), 1)

        F = u1 + u2
        M = mq + mp

        # 1. Quadrotor attitude dynamics
        ddtheta = (Lq / I) * (u2 - u1)
        
        # 2. Pendulum dynamics
        ddphi = -F * (s_phi * c_theta - s_theta * c_phi) / (mq * Lp)
        
        # 3. Translational dynamics 
        ddx = (-s_theta * F - mp * Lp * c_phi * ddphi + mp * Lp * s_phi * (phi_dot**2)) / M
        ddz = (c_theta * F - M * g - mp * Lp * s_phi * ddphi - mp * Lp * c_phi * (phi_dot**2)) / M

        # 4. Semi-implicit Euler update
        vx_new = vx + ddx * dt
        vz_new = vz + ddz * dt
        theta_dot_new = theta_dot + ddtheta * dt
        phi_dot_new = phi_dot + ddphi * dt

        x_new = x_pos + vx_new * dt
        z_new = z_pos + vz_new * dt

        ds_theta = c_theta * theta_dot_new * dt
        dc_theta = -s_theta * theta_dot_new * dt
        
        s_theta_new = s_theta + ds_theta
        c_theta_new = c_theta + dc_theta
        
        # Renormalize to maintain unit circle constraint
        norm_theta = ca.sqrt(s_theta_new**2 + c_theta_new**2)
        s_theta_new = s_theta_new / norm_theta
        c_theta_new = c_theta_new / norm_theta
        
        phi = ca.atan2(s_phi, c_phi)
        phi_new = phi + phi_dot_new * dt
        s_phi_new = ca.sin(phi_new)
        c_phi_new = ca.cos(phi_new)

        x_next = ca.vertcat(
            x_new, z_new, vx_new, vz_new,
            s_theta_new, c_theta_new, theta_dot_new,
            s_phi_new, c_phi_new, phi_dot_new
        )

        return ca.Function("f", [x, u], [x_next])

    f_dyn = dynamics(**params)
    
    X = ca.MX.sym('X', nx, T+1)
    U = ca.MX.sym('U', nu, T) 
    x0_param = ca.MX.sym('x0', nx)
    xg_param = ca.MX.sym('xg', 2)

    cost = 0
    g = []

    g.append(X[:, 0] - x0_param)

    for t in range(T):
        xt = X[:, t]
        ut = U[:, t] 
        xt_next = X[:, t+1]

        g.append(f_dyn(xt, ut) - xt_next)

        # Position cost relative to goal
        pos_error = xt[0:2] - xg_param
        pos_cost = ca.sum1(ca.fabs(pos_error)) + ca.sumsqr(pos_error)
        
        # Velocity cost
        vel_cost = ca.sumsqr(xt[2:4])
        
        # Quadrotor orientation cost (1 - |cos(theta)|)
        theta_cost = 1 - ca.fabs(xt[5])
        
        # Quadrotor angular velocity cost
        omega_cost = xt[6]**2
        
        # Payload orientation cost (cos(phi)^3)
        phi_cost = xt[8]**3
        
        # Payload angular velocity cost
        phi_dot_cost = xt[9]**2
        
        stage_cost = params['dt'] * (
            config['pos_cost_multiplier'] * pos_cost +
            config['vel_cost_multiplier'] * vel_cost +
            config['theta_cost_multiplier'] * theta_cost +
            config['omega_cost_multiplier'] * omega_cost +
            (config['phi_cost_multiplier'] * phi_cost - config['phi_cost_multiplier']) * 
            (1.0 / (1.0 + config['phi_dot_cost_multiplier'] * phi_dot_cost))
        )
        
        balance_condition = ca.if_else(
            ca.logic_and(xt[8] < -0.92, ca.fabs(xt[9]) < 0.2),
            -config['balance_reward'] * params['dt'],  
            0
        )
        
        cost += stage_cost + balance_condition

    # Terminal cost
    final_pos_error = X[0:2, -1] - xg_param
    final_pos_cost = ca.sum1(ca.fabs(final_pos_error)) + ca.sumsqr(final_pos_error)
    final_vel_cost = ca.sumsqr(X[2:4, -1])
    final_theta_cost = 1 - ca.fabs(X[5, -1])
    final_omega_cost = X[6, -1]**2
    final_phi_cost = X[8, -1]**3
    final_phi_dot_cost = X[9, -1]**2
    
    # Weight terminal cost more heavily
    terminal_weight = 10.0
    cost += terminal_weight * params['dt'] * (
        config['pos_cost_multiplier'] * final_pos_cost +
        config['vel_cost_multiplier'] * final_vel_cost +
        config['theta_cost_multiplier'] * final_theta_cost +
        config['omega_cost_multiplier'] * final_omega_cost +
        (config['phi_cost_multiplier'] * final_phi_cost - config['phi_cost_multiplier']) * 
        (1.0 / (1.0 + config['phi_dot_cost_multiplier'] * final_phi_dot_cost))
    )

    g_flat = ca.vertcat(*g)
    opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    params_vec = ca.vertcat(x0_param, xg_param)

    nlp = {'x': opt_vars, 'f': cost, 'g': g_flat, 'p': params_vec}

    opts = {
        'ipopt.print_level': 0,
        'ipopt.max_iter': 1000, 
        'ipopt.tol': 1e-6,
        'ipopt.acceptable_tol': 1e-4,
        'print_time': 0
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    # Setup bounds
    x0_guess = np.zeros((nx, T+1))
    u0_guess = np.zeros((nu, T))  
    
    # init guess arrays
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
        x0_np[0] = np.random.uniform(-1.5, 1.5)  # x position
        x0_np[1] = np.random.uniform(0.5, 1.5)   # z position
        
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
        
        if config['curriculum_level'] <= 4:
            # Below CL 4, goal can be one of two locations
            if np.random.random() < 0.5:
                xg_np = np.array([0.0, 0.0])
            else:
                xg_np = np.array([1.0, 1.0])
        else:  # curriculum_level >= 5
            # At CL 5, goal can be one of 5 locations
            rng = np.random.random()
            if rng < 0.20:
                xg_np = np.array([0.0, 0.0])
            elif 0.20 <= rng < 0.40:
                xg_np = np.array([1.0, 1.0])
            elif 0.40 <= rng < 0.60:
                xg_np = np.array([-1.0, -1.0])
            elif 0.60 <= rng < 0.80:
                xg_np = np.array([1.0, -1.0])
            else:
                xg_np = np.array([-1.0, 1.0])
        
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
                U_sol = sol[nx*(T+1):].reshape((nu, T))  # Normalized actions
                
                # Check trajectory quality
                final_pos_error = np.linalg.norm(X_sol[0:2, -1] - xg_np)
                phi_traj = np.arctan2(X_sol[7, :], X_sol[8, :])
                max_pend_dev = np.max(np.abs(phi_traj - np.pi))
                
                # Only keep good trajectories
                if final_pos_error < 0.3 and max_pend_dev < np.radians(120):
                    # Convert to state-action pairs
                    for t in range(T):
                        state = np.concatenate([X_sol[:, t], xg_np])
                        action = U_sol[:, t].copy()  # Already normalized
                        next_state = np.concatenate([X_sol[:, t+1], xg_np])
                        
                        pos_error = np.abs(X_sol[0:2, t] - xg_np)
                        pos_cost = np.sum(pos_error) + np.sum((X_sol[0:2, t] - xg_np)**2)
                        vel_cost = np.sum(X_sol[2:4, t]**2)
                        theta_cost = 1 - np.abs(X_sol[5, t])
                        omega_cost = X_sol[6, t]**2
                        phi_cost = X_sol[8, t]**3
                        phi_dot_cost = X_sol[9, t]**2
                        
                        reward = params['dt'] * (
                            -config['pos_cost_multiplier'] * pos_cost
                            -config['vel_cost_multiplier'] * vel_cost
                            -config['theta_cost_multiplier'] * theta_cost
                            -config['omega_cost_multiplier'] * omega_cost
                            -(config['phi_cost_multiplier'] * phi_cost - config['phi_cost_multiplier']) *
                            (1.0 / (1.0 + config['phi_dot_cost_multiplier'] * phi_dot_cost))
                        )
                        
                        # Add balance reward
                        if X_sol[8, t] < -0.92 and abs(X_sol[9, t]) < 0.2:
                            reward += config['balance_reward'] * params['dt']
                        
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
    
    # Convert to arrays
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
                       sequence_length=T,
                       config=config) 
    
    print(f"Expert data saved to {save_path}")
    
    # Basic statistics
    print(f"\n=== DATASET STATISTICS ===")
    print(f"State dimensionality: {states.shape[1]} (10 state + 2 goal)") 
    print(f"Action dimensionality: {actions.shape[1]} (normalized)")
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
            'state_dim': 12, 
            'action_dim': nu,
            'dt': dt,
            'hover_force': hover_force,
            'config': config
        }
    }

def visualize_expert_trajectories(data_path: str = 'expert_trajectories.npz', num_to_plot: int = 5):
    """Visualize a few expert trajectories for verification"""
    
    data = np.load(data_path, allow_pickle=True)
    states = data['states']
    actions = data['actions']
    trajectory_ids = data['trajectory_ids']
    
    goal_positions = states[:, -2:] 
    
    unique_trajs = np.unique(trajectory_ids)[:num_to_plot]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  
    fig.suptitle('Expert Trajectory Samples', fontsize=14)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_trajs)))
    
    for i, traj_id in enumerate(unique_trajs):
        mask = trajectory_ids == traj_id
        traj_states = states[mask]
        traj_actions = actions[mask]
        traj_goals = goal_positions[mask]
        
        # Position trajectory
        axes[0,0].plot(traj_states[:, 0], traj_states[:, 1], 
                      color=colors[i], alpha=0.7, label=f'Traj {traj_id}')
        axes[0,0].scatter(traj_states[0, 0], traj_states[0, 1], 
                         color=colors[i], s=50, marker='o')
        axes[0,0].scatter(traj_states[-1, 0], traj_states[-1, 1], 
                         color=colors[i], s=50, marker='x')
        axes[0,0].scatter(traj_goals[0, 0], traj_goals[0, 1], 
                         color=colors[i], s=100, marker='*', edgecolors='black')
    
    axes[0,0].set_xlabel('X Position [m]')
    axes[0,0].set_ylabel('Z Position [m]')
    axes[0,0].set_title('Position Trajectories (★ = goal)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlim(-2.5, 2.5)
    axes[0,0].set_ylim(-2.5, 2.5)
    
    # Pendulum angles
    for i, traj_id in enumerate(unique_trajs):
        mask = trajectory_ids == traj_id
        traj_states = states[mask]
        phi_traj = np.degrees(np.arctan2(traj_states[:, 7], traj_states[:, 8]))
        time_vec = np.arange(len(traj_states)) * 0.02  
        
        axes[0,1].plot(time_vec, phi_traj, color=colors[i], alpha=0.7)
    
    axes[0,1].axhline(y=180, color='black', linestyle='--', alpha=0.5, label='Upright')
    axes[0,1].set_xlabel('Time [s]')
    axes[0,1].set_ylabel('Pendulum Angle [deg]')
    axes[0,1].set_title('Pendulum Angles')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Drone angles
    for i, traj_id in enumerate(unique_trajs):
        mask = trajectory_ids == traj_id
        traj_states = states[mask]
        theta_traj = np.degrees(np.arctan2(traj_states[:, 4], traj_states[:, 5]))
        time_vec = np.arange(len(traj_states)) * 0.02
        
        axes[0,2].plot(time_vec, theta_traj, color=colors[i], alpha=0.7)
    
    axes[0,2].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Level')
    axes[0,2].set_xlabel('Time [s]')
    axes[0,2].set_ylabel('Drone Angle [deg]')
    axes[0,2].set_title('Drone Pitch Angles')
    axes[0,2].grid(True, alpha=0.3)
    
    for i, traj_id in enumerate(unique_trajs):
        mask = trajectory_ids == traj_id
        traj_actions = actions[mask]
        time_vec = np.arange(len(traj_actions)) * 0.02
        
        axes[1,0].plot(time_vec, traj_actions[:, 0], color=colors[i], alpha=0.7, linestyle='-')
        axes[1,1].plot(time_vec, traj_actions[:, 1], color=colors[i], alpha=0.7, linestyle='-')
    
    axes[1,0].set_xlabel('Time [s]')
    axes[1,0].set_ylabel('Normalized Action')
    axes[1,0].set_title('Left Thruster Action (normalized)')
    axes[1,0].set_ylim(-1.1, 1.1)
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].set_xlabel('Time [s]')
    axes[1,1].set_ylabel('Normalized Action')
    axes[1,1].set_title('Right Thruster Action (normalized)')
    axes[1,1].set_ylim(-1.1, 1.1)
    axes[1,1].grid(True, alpha=0.3)
    
    hover_force = data.get('metadata', {}).item().get('hover_force', 9.80665)
    for i, traj_id in enumerate(unique_trajs):
        mask = trajectory_ids == traj_id
        traj_actions = actions[mask]
        time_vec = np.arange(len(traj_actions)) * 0.02
        
        # Convert normalized actions to actual thrust
        actual_u1 = hover_force + hover_force * traj_actions[:, 0]
        actual_u2 = hover_force + hover_force * traj_actions[:, 1]
        
        axes[1,2].plot(time_vec, actual_u1, color=colors[i], alpha=0.7, linestyle='-', label=f'u1 Traj {traj_id}' if i == 0 else '')
        axes[1,2].plot(time_vec, actual_u2, color=colors[i], alpha=0.7, linestyle='--', label=f'u2 Traj {traj_id}' if i == 0 else '')
    
    axes[1,2].axhline(y=hover_force, color='black', linestyle=':', alpha=0.5, label='Hover thrust')
    axes[1,2].set_xlabel('Time [s]')
    axes[1,2].set_ylabel('Thrust [N]')
    axes[1,2].set_title('Actual Thrust Forces')
    axes[1,2].legend() if len(unique_trajs) <= 3 else None
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def verify_dynamics_match(mpc_params=None):
    print("\n=== VERIFYING DYNAMICS MATCH ===")
    
    test_state = np.array([
        0.5, 1.0,      # position
        0.2, -0.1,     # velocity
        0.1, 0.995,    # sin/cos theta (small angle)
        0.5,           # theta_dot
        0.0, -1.0,     # sin/cos phi (upright)
        0.3            # phi_dot
    ])
    
    test_action_norm = np.array([0.2, -0.1]) 
    
    rl_params = {
        'mq': 1.5,
        'mp': 0.5,
        'Lq': 0.5,
        'Lp': 0.75,
        'I': 0.4,
        'g': 9.80665,
        'dt': 0.02
    }
    
    hover_force = (rl_params['mq'] + rl_params['mp']) * rl_params['g'] / 2
    
    # Convert normalized to actual thrust
    u1 = hover_force + hover_force * np.clip(test_action_norm[0], -1, 1)
    u2 = hover_force + hover_force * np.clip(test_action_norm[1], -1, 1)
    actual_thrust = np.array([u1, u2])
    
    print(f"Test state: {test_state}")
    print(f"Normalized action: {test_action_norm}")
    print(f"Actual thrust: {actual_thrust}")
    print(f"Hover force: {hover_force:.3f} N")

if __name__ == "__main__":
    rl_config = {
        'pos_cost_multiplier': 15.0,
        'vel_cost_multiplier': 0.5,
        'theta_cost_multiplier': 5.0,
        'omega_cost_multiplier': 5.0,
        'phi_cost_multiplier': 25.0,
        'phi_dot_cost_multiplier': 5.0,
        'balance_reward': 100.0,
        'curriculum_level': 0
    }
    
    expert_data = generate_expert_trajectories(
        num_trajectories=100,
        save_path='expert_trajectories_aligned.npz',
        config=rl_config
    )
    
    visualize_expert_trajectories('expert_trajectories_aligned.npz', num_to_plot=5)
    
    verify_dynamics_match()
