import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 40                # Horizon length
dt = 0.05             # Time step
nx = 10               # State dimension
nu = 2                # Control dimension

# Dynamics parameters
params = dict(
    dt=dt, # time step
    mq=1.0, # mass of the drone
    mp=0.3, # mass of the pendulum
    Lq=0.1, # length of the drone
    Lp=0.5, # length of the pendulum
    I=0.02, # inertia of the drone
    g=9.81 # gravity
)

# Tunable weights
w_goal = 10.0 # weight for reaching the goal
w_upright = 20.0 # weight for upright pendulum
w_effort = 0.01 # weight for control effort

# Input data
x0_np = np.zeros(nx)         # initial state
x0_np[4] = 0.0   # s_theta
x0_np[5] = 1.0   # c_theta
x0_np[7] = 0.0   # s_phi
x0_np[8] = 1.0   # c_phi
xg_np = np.random.uniform(-2, 2, 2)  # goal position (x, y)

# dynamics function of the drone-pendulum system
# X = [x, z, dx, dz, s(theta), c(theta), dtheta, s(phi), c(phi), dphi]
def dynamics(dt, mq, mp, Lq, Lp, I, g):

    # Symbolic variables
    x = ca.MX.sym('x', nx) # state vector
    u = ca.MX.sym('u', nu) # control vector

    x_pos, z_pos, vx, vz = x[0], x[1], x[2], x[3]
    s_theta, c_theta, theta_dot = x[4], x[5], x[6]
    s_phi, c_phi, phi_dot = x[7], x[8], x[9]
    u1, u2 = u[0], u[1]

    # Common terms
    F = u1 + u2
    M = mq + mp

    # 1. Quadrotor attitude dynamics (theta)
    ddtheta = (Lq / I) * (u2 - u1)

    # 2. Pendulum dynamics (phi)
    # Derived from the coupling of translational accelerations with the pendulum equation.
    ddphi = -F * (s_phi * c_theta - s_theta * c_phi) / (mq * Lp)

    # 3. Translational dynamics
    ddx = (-s_theta * F - mp * Lp * c_phi * ddphi + mp * Lp * s_phi * phi_dot**2) / M
    ddz = (c_theta * F - M * g - mp * Lp * s_phi * ddphi - mp * Lp * c_phi * phi_dot**2) / M

    # 4. Semi-implicit Euler
    # First update the velocities using the computed accelerations.
    vx_new = vx + ddx * dt
    vz_new = vz + ddz * dt
    theta_dot_new = theta_dot + ddtheta * dt
    phi_dot_new = phi_dot + ddphi * dt

    # Then update the positions with the new velocities.
    x_new = x_pos + vx_new * dt
    z_new = z_pos + vz_new * dt

    # Direct update using rotation matrix approach
    ds_theta = c_theta * theta_dot_new * dt
    dc_theta = -s_theta * theta_dot_new * dt

    s_theta_new = s_theta + ds_theta
    c_theta_new = c_theta + dc_theta

    # Renormalize to maintain unit circle constraint
    norm_theta = ca.sqrt(s_theta_new**2 + c_theta_new**2)
    norm_theta = ca.fmax(norm_theta, 1e-6)
    s_theta_new /= norm_theta
    c_theta_new /= norm_theta

    phi = ca.atan2(s_phi, c_phi)
    s_phi_new = ca.sin(phi + phi_dot * dt)
    c_phi_new = ca.cos(phi + phi_dot * dt)

    # Pack into next state
    x_next = ca.vertcat(
        x_new, z_new,
        vx_new, vz_new,
        s_theta_new, c_theta_new,
        theta_dot_new,
        s_phi_new, c_phi_new,
        phi_dot_new
    )

    # Define symbolic function
    f = ca.Function("f", [x, u], [x_next])
    return f

# Define CasADi functions
f_dyn = dynamics(**params)

# Optimization variables
X = ca.MX.sym('X', nx, T+1)
U = ca.MX.sym('U', nu, T)

# Define cost and constraints
cost = 0
g = []

for t in range(T):
    xt = X[:, t]
    ut = U[:, t]
    xt_next = X[:, t+1]

    # Dynamics constraint
    g.append(f_dyn(xt, ut) - xt_next)

    # Cost terms
    pos_xy = xt[0:2]
    pend_theta = 1 - xt[5]  # 1 - cos(theta)
    
    cost += w_goal * ca.sumsqr(pos_xy - xg_np)
    cost += w_upright * pend_theta**2
    cost += w_effort * ca.sumsqr(ut)

# Final state cost
cost += w_goal * ca.sumsqr(X[0:2, -1] - xg_np)
cost += w_upright * (1 - X[5, -1])**2

# === Solve NLP ===
# Flatten constraints
g_flat = ca.vertcat(*g)

# Decision variables
opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

# Constraints and bounds
nlp = {
    'x': opt_vars,
    'f': cost,
    'g': g_flat
}

solver = ca.nlpsol('solver', 'ipopt', nlp)

# Initial guess
x0_guess = np.tile(x0_np.reshape(-1, 1), (1, T+1))
print(f"x0_guess values: {x0_guess}")
u0_guess = np.zeros((nu, T))
x_init = x0_guess.flatten()
u_init = u0_guess.flatten()
print(f"u_init values: {u_init}")
initial_guess = np.concatenate([x_init, u_init])

# Bounds
lbx = -ca.inf * np.ones_like(initial_guess)
ubx = ca.inf * np.ones_like(initial_guess)

# Control bounds
u_lb = np.zeros(nu * T)
u_ub = np.ones(nu * T) * 5.0
lbx[nx * (T+1):] = u_lb
ubx[nx * (T+1):] = u_ub

lbg = np.zeros(g_flat.shape)
ubg = np.zeros(g_flat.shape)

# Solve
solution = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

# Extract result
sol = solution['x'].full().flatten()
X_sol = sol[:nx*(T+1)].reshape((nx, T+1))
U_sol = sol[nx*(T+1):].reshape((nu, T))

# Done: X_sol, U_sol are your optimal state and action sequences

# Extract x, z over time
x_traj = X_sol[0, :]
z_traj = X_sol[1, :]

plt.figure(figsize=(6, 6))
plt.plot(x_traj, z_traj, marker='o', label='Trajectory')
plt.scatter(x_traj[0], z_traj[0], c='green', label='Start')
plt.scatter(x_traj[-1], z_traj[-1], c='red', label='Goal')
plt.xlabel("x [m]")
plt.ylabel("z [m]")
plt.title("Quadrotor Position Trajectory")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()