import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONSTRAINED MPC WITH PENDULUM STARTING UPRIGHT
# =============================================================================

# Horizon and timing
T = 50                # Horizon length
dt = 0.1              # Time step

# Tighter constraints for realistic behavior
POSITION_LIMIT = 3.0  # Position bounds: [-3, 3]
VELOCITY_LIMIT = 3.0  # Velocity bounds: [-3, 3] m/s
ANGLE_LIMIT = 45.0    # Max drone tilt: ±45 degrees
PENDULUM_LIMIT = 60.0 # Max pendulum swing from upright: ±60 degrees
ANGULAR_VEL_LIMIT = 5.0 # Angular velocity limits

# Cost weights - rebalanced for upright pendulum
w_goal = 200.0        # Goal reaching
w_upright = 200.0     # Pendulum upright (increased weight)
w_effort = 0.5        # Control effort
w_velocity = 5.0      # Velocity damping
w_smooth = 2.0        # Control smoothness
w_terminal = 10.0     # Terminal cost multiplier

# Physical parameters
params = dict(
    dt=dt, mq=1.0, mp=0.3, Lq=0.1, Lp=0.5, I=0.02, g=9.81
)

# Control limits
u_max = 15.0          
u_min = 0.5           

nx = 10
nu = 2

# CORRECTED: Start with pendulum upright (phi = 180° = π radians)
x0_np = np.zeros(nx)
x0_np[4] = 0.0   # s_theta (drone level)
x0_np[5] = 1.0   # c_theta  
x0_np[6] = 0.0   # theta_dot
x0_np[7] = 0.0   # s_phi = sin(π) = 0 (pendulum upright)
x0_np[8] = -1.0  # c_phi = cos(π) = -1 (pendulum upright)
x0_np[9] = 0.0   # phi_dot

# Goal position
xg_np = np.array([0.8, 1.5])

print(f"=== PENDULUM UPRIGHT SETUP ===")
print(f"Initial pendulum angle: {np.degrees(np.arctan2(x0_np[7], x0_np[8])):.1f}° (should be 180° for upright)")
print(f"cos(phi) initial: {x0_np[8]:.1f} (should be -1 for upright)")
print(f"Position bounds: ±{POSITION_LIMIT}m")
print(f"Velocity bounds: ±{VELOCITY_LIMIT}m/s") 
print(f"Drone angle limit: ±{ANGLE_LIMIT}°")
print(f"Pendulum limit: ±{PENDULUM_LIMIT}° from upright")
print(f"Goal: x={xg_np[0]:.1f}, z={xg_np[1]:.1f}")

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
    
    # Pendulum dynamics - corrected for upright equilibrium
    # When pendulum is upright, phi = π, so we want to linearize around phi = π
    phi_angle = ca.atan2(s_phi, c_phi + eps)
    # Deviation from upright (π radians)
    phi_dev = phi_angle - np.pi
    
    # Linearized pendulum dynamics around upright position
    # For inverted pendulum: ddphi = (g/L)*phi_dev + (F/(M*L))*sin(theta)
    theta_angle = ca.atan2(s_theta, c_theta + eps)
    ddphi = (g / Lp) * phi_dev + (F / (M * Lp)) * theta_angle
    
    # Translational dynamics
    ddx = -theta_angle * F / M
    ddz = F / M - g

    # Integration with damping
    damping = 0.95
    vx_new = damping * (vx + ddx * dt)
    vz_new = damping * (vz + ddz * dt)
    theta_dot_new = damping * (theta_dot + ddtheta * dt)
    phi_dot_new = damping * (phi_dot + ddphi * dt)

    x_new = x_pos + vx_new * dt
    z_new = z_pos + vz_new * dt

    # Angle integration
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

# Optimization setup
X = ca.MX.sym('X', nx, T+1)
U = ca.MX.sym('U', nu, T)

cost = 0
g = []

# Initial condition
g.append(X[:, 0] - x0_np)

# Stage costs - CORRECTED for upright pendulum
for t in range(T):
    xt = X[:, t]
    ut = U[:, t]
    xt_next = X[:, t+1]

    # Dynamics constraint
    g.append(f_dyn(xt, ut) - xt_next)

    # Cost terms
    pos_error = xt[0:2] - xg_np
    vel = xt[2:4]
    
    # CORRECTED: Upright pendulum cost (penalize deviation from cos(phi) = -1)
    pend_upright_error = 1.0 + xt[8]  # 1 + cos(phi), minimized when cos(phi) = -1
    
    # Hover thrust reference
    hover_thrust = params['g'] * (params['mq'] + params['mp']) / 2
    control_deviation = ut - hover_thrust
    
    # Quadratic costs
    cost += w_goal * ca.sumsqr(pos_error)
    cost += w_upright * pend_upright_error**2
    cost += w_effort * ca.sumsqr(control_deviation)
    cost += w_velocity * ca.sumsqr(vel)
    
    # Control smoothness
    if t > 0:
        control_change = ut - U[:, t-1]
        cost += w_smooth * ca.sumsqr(control_change)

# Terminal cost - CORRECTED
final_pos_error = X[0:2, -1] - xg_np
final_pend_error = 1.0 + X[8, -1]  # Corrected for upright
final_vel = X[2:4, -1]

cost += w_terminal * w_goal * ca.sumsqr(final_pos_error)
cost += w_terminal * w_upright * final_pend_error**2
cost += w_terminal * w_velocity * ca.sumsqr(final_vel)

# Setup NLP
g_flat = ca.vertcat(*g)
opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

nlp = {'x': opt_vars, 'f': cost, 'g': g_flat}

# Solver options
opts = {
    'ipopt.print_level': 1,
    'ipopt.max_iter': 1000,
    'ipopt.tol': 1e-6,
    'ipopt.acceptable_tol': 1e-4,
    'ipopt.mu_strategy': 'adaptive',
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Initial guess - keep pendulum near upright
x0_guess = np.zeros((nx, T+1))
total_mass = params['mq'] + params['mp']
hover_thrust = total_mass * params['g'] / 2

for k in range(T+1):
    # Linear trajectory
    alpha = k / T
    x0_guess[0, k] = x0_np[0] + alpha * (xg_np[0] - x0_np[0])
    x0_guess[1, k] = x0_np[1] + alpha * (xg_np[1] - x0_np[1])
    
    # Small velocities
    x0_guess[2, k] = (xg_np[0] - x0_np[0]) / (T * dt) * 0.1
    x0_guess[3, k] = (xg_np[1] - x0_np[1]) / (T * dt) * 0.1
    
    # Keep drone level
    x0_guess[4, k] = 0.0    # s_theta
    x0_guess[5, k] = 1.0    # c_theta
    x0_guess[6, k] = 0.0    # theta_dot
    
    # Keep pendulum near upright
    x0_guess[7, k] = 0.0    # s_phi ≈ 0 (near upright)
    x0_guess[8, k] = -0.98  # c_phi ≈ -1 (near upright)
    x0_guess[9, k] = 0.0    # phi_dot

# Control near hover
u0_guess = np.ones((nu, T)) * hover_thrust

# Bounds
initial_guess = np.concatenate([x0_guess.flatten(), u0_guess.flatten()])
lbx = -np.inf * np.ones_like(initial_guess)
ubx = np.inf * np.ones_like(initial_guess)

# State bounds
for i in range(T+1):
    idx = i * nx
    
    # Position bounds
    lbx[idx:idx+2] = -POSITION_LIMIT
    ubx[idx:idx+2] = POSITION_LIMIT
    
    # Velocity bounds
    lbx[idx+2:idx+4] = -VELOCITY_LIMIT
    ubx[idx+2:idx+4] = VELOCITY_LIMIT
    
    # Drone angle bounds
    max_sin_theta = np.sin(np.radians(ANGLE_LIMIT))
    min_cos_theta = np.cos(np.radians(ANGLE_LIMIT))
    lbx[idx+4] = -max_sin_theta
    ubx[idx+4] = max_sin_theta
    lbx[idx+5] = min_cos_theta
    ubx[idx+5] = 1.0
    
    # Pendulum angle bounds (±60° from upright = 180°)
    # So pendulum can swing from 120° to 240°
    upright_angle = np.pi  # 180°
    max_dev = np.radians(PENDULUM_LIMIT)  # 60°
    
    min_phi = upright_angle - max_dev  # 120°
    max_phi = upright_angle + max_dev  # 240°
    
    # Convert to sin/cos bounds
    lbx[idx+7] = min(np.sin(min_phi), np.sin(max_phi))  # s_phi
    ubx[idx+7] = max(np.sin(min_phi), np.sin(max_phi))
    lbx[idx+8] = min(np.cos(min_phi), np.cos(max_phi))  # c_phi
    ubx[idx+8] = max(np.cos(min_phi), np.cos(max_phi))
    
    # Angular velocity bounds
    lbx[idx+6] = -ANGULAR_VEL_LIMIT
    ubx[idx+6] = ANGULAR_VEL_LIMIT
    lbx[idx+9] = -ANGULAR_VEL_LIMIT
    ubx[idx+9] = ANGULAR_VEL_LIMIT

# Control bounds
lbx[nx*(T+1):] = u_min
ubx[nx*(T+1):] = u_max

# Constraint bounds
lbg = ubg = np.zeros(g_flat.shape)

print(f"\n=== SOLVING WITH UPRIGHT PENDULUM ===")
print(f"State variables: {nx*(T+1)}, Control variables: {nu*T}")

solution = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

success = solver.stats()['success']
print(f"✓ Optimization {'successful' if success else 'completed'}!")

# Extract results
sol = solution['x'].full().flatten()
X_sol = sol[:nx*(T+1)].reshape((nx, T+1))
U_sol = sol[nx*(T+1):].reshape((nu, T))

theta_traj = np.arctan2(X_sol[4, :], X_sol[5, :])
phi_traj = np.arctan2(X_sol[7, :], X_sol[8, :])

# Convert phi to degrees and handle discontinuity around 180°
phi_deg = np.degrees(phi_traj)
# Ensure angles are in proper range for upright pendulum
phi_deg = np.where(phi_deg < 0, phi_deg + 360, phi_deg)

print(f"\n=== UPRIGHT PENDULUM ANALYSIS ===")
print("Time [s] | Pendulum φ [deg] | cos(φ) | Deviation from Upright")
print("-" * 60)

time_vec = np.linspace(0, T*dt, T+1)

for i in range(0, T+1, max(1, T//10)):
    phi_deg_i = phi_deg[i]
    cos_phi = X_sol[8, i]
    deviation_from_upright = abs(phi_deg_i - 180.0)
    
    status = "✅" if deviation_from_upright < 15 else "⚠️" if deviation_from_upright < 45 else "❌"
    print(f"{time_vec[i]:6.2f}   |     {phi_deg_i:8.1f}    | {cos_phi:6.3f} | {deviation_from_upright:8.1f}° {status}")

print("-" * 60)
print(f"Initial pendulum angle: {phi_deg[0]:6.1f}° (target: 180°)")
print(f"Final pendulum angle:   {phi_deg[-1]:6.1f}° (target: 180°)")
print(f"Max deviation from upright: {np.max(np.abs(phi_deg - 180)):6.1f}°")

# Performance analysis
final_pos_error = np.linalg.norm(X_sol[0:2, -1] - xg_np)
final_pend_deviation = abs(phi_deg[-1] - 180.0)
max_pos = np.max(np.abs(X_sol[0:2, :]))
max_vel = np.max(np.abs(X_sol[2:4, :]))
max_drone_angle = np.max(np.abs(np.degrees(theta_traj)))
max_pend_deviation = np.max(np.abs(phi_deg - 180.0))

print(f"\n=== PERFORMANCE SUMMARY ===")
print(f"Final position error: {final_pos_error:.3f} m")
print(f"Final pendulum deviation: {final_pend_deviation:.1f}° from upright")
print(f"Max position: {max_pos:.2f} m (limit: {POSITION_LIMIT:.1f})")
print(f"Max velocity: {max_vel:.2f} m/s (limit: {VELOCITY_LIMIT:.1f})")
print(f"Max drone angle: {max_drone_angle:.1f}° (limit: {ANGLE_LIMIT:.1f})")
print(f"Max pendulum deviation: {max_pend_deviation:.1f}° (limit: {PENDULUM_LIMIT:.1f})")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Upright Pendulum MPC (Error: {final_pos_error:.2f}m, Pend Dev: {final_pend_deviation:.1f}°)', fontsize=14)

# 1. Trajectory
ax1 = axes[0, 0]
ax1.plot(X_sol[0, :], X_sol[1, :], 'b-', marker='o', markersize=3, linewidth=2, label='Trajectory')
ax1.scatter(X_sol[0, 0], X_sol[1, 0], c='green', s=150, label='Start', zorder=5, marker='s')
ax1.scatter(xg_np[0], xg_np[1], c='red', s=150, label='Goal', marker='*', zorder=5)
ax1.scatter(X_sol[0, -1], X_sol[1, -1], c='blue', s=100, label='Final', zorder=5)
ax1.axhline(y=POSITION_LIMIT, color='red', linestyle='--', alpha=0.5, label='Bounds')
ax1.axhline(y=-POSITION_LIMIT, color='red', linestyle='--', alpha=0.5)
ax1.axvline(x=POSITION_LIMIT, color='red', linestyle='--', alpha=0.5)
ax1.axvline(x=-POSITION_LIMIT, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('x [m]')
ax1.set_ylabel('z [m]')
ax1.set_title('Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 2. Position vs time
ax2 = axes[0, 1]
ax2.plot(time_vec, X_sol[0, :], 'b-', label='x position', linewidth=2)
ax2.plot(time_vec, X_sol[1, :], 'r-', label='z position', linewidth=2)
ax2.axhline(y=xg_np[0], color='b', linestyle=':', alpha=0.7, label='Goals')
ax2.axhline(y=xg_np[1], color='r', linestyle=':', alpha=0.7)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Position [m]')
ax2.set_title('Position vs Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Pendulum angle (focused on upright region)
ax3 = axes[0, 2]
ax3.plot(time_vec, phi_deg, 'm-', label='Pendulum φ', linewidth=3)
ax3.axhline(y=180, color='k', linestyle='-', alpha=0.8, label='Upright (180°)', linewidth=2)
ax3.axhline(y=180+PENDULUM_LIMIT, color='m', linestyle='--', alpha=0.5, label='Limits')
ax3.axhline(y=180-PENDULUM_LIMIT, color='m', linestyle='--', alpha=0.5)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Pendulum Angle [deg]')
ax3.set_title('Pendulum Angle (Upright = 180°)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(120, 240)  # Focus on upright region

# 4. Controls
ax4 = axes[1, 0]
time_u = np.linspace(0, (T-1)*dt, T)
ax4.plot(time_u, U_sol[0, :], 'b-', label='u1 (left)', linewidth=2)
ax4.plot(time_u, U_sol[1, :], 'r-', label='u2 (right)', linewidth=2)
ax4.plot(time_u, U_sol[0, :] + U_sol[1, :], 'k--', label='Total', linewidth=2)
ax4.axhline(y=hover_thrust*2, color='gray', linestyle=':', alpha=0.7, label='Hover')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Thrust [N]')
ax4.set_title('Control Inputs')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Drone angle
ax5 = axes[1, 1]
ax5.plot(time_vec, np.degrees(theta_traj), 'g-', label='Drone θ', linewidth=2)
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Level')
ax5.axhline(y=ANGLE_LIMIT, color='g', linestyle='--', alpha=0.5, label='Limits')
ax5.axhline(y=-ANGLE_LIMIT, color='g', linestyle='--', alpha=0.5)
ax5.set_xlabel('Time [s]')
ax5.set_ylabel('Drone Angle [deg]')
ax5.set_title('Drone Tilt Angle')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. System visualization at final time
ax6 = axes[1, 2]
drone_x, drone_z = X_sol[0, -1], X_sol[1, -1]
drone_theta = theta_traj[-1]
pend_phi = phi_traj[-1]

# Drone
drone_l = params['Lq'] * 2
cos_t, sin_t = np.cos(drone_theta), np.sin(drone_theta)
drone_x_pts = [-drone_l/2, drone_l/2]
drone_z_pts = [0, 0]
drone_rot_x = [cos_t*x - sin_t*z + drone_x for x, z in zip(drone_x_pts, drone_z_pts)]
drone_rot_z = [sin_t*x + cos_t*z + drone_z for x, z in zip(drone_x_pts, drone_z_pts)]
ax6.plot(drone_rot_x, drone_rot_z, 'k-', linewidth=6, label='Drone')

# Pendulum
pend_x = drone_x + params['Lp'] * np.sin(pend_phi)
pend_z = drone_z - params['Lp'] * np.cos(pend_phi)
ax6.plot([drone_x, pend_x], [drone_z, pend_z], 'r-', linewidth=4, label='Pendulum')
ax6.scatter(pend_x, pend_z, c='red', s=200, zorder=5, marker='o')
ax6.scatter(xg_np[0], xg_np[1], c='green', s=200, marker='*', label='Goal', zorder=5)

# Add upright reference
upright_x = drone_x
upright_z = drone_z + params['Lp']
ax6.plot([drone_x, upright_x], [drone_z, upright_z], 'gray', linestyle='--', alpha=0.5, linewidth=2, label='Upright ref')

ax6.set_xlabel('x [m]')
ax6.set_ylabel('z [m]')
ax6.set_title('Final Configuration')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.axis('equal')

plt.tight_layout()
plt.show()

print(f"\n=== UPRIGHT PENDULUM SUMMARY ===")
print(f"✅ Started with pendulum upright (180°)")
print(f"Final deviation from upright: {final_pend_deviation:.1f}°")
print(f"Successfully reached goal: {final_pos_error < 0.1}")
print(f"Pendulum stayed relatively upright: {max_pend_deviation < PENDULUM_LIMIT}")