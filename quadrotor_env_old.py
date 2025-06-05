import numpy as np
import gymnasium as gym

class QuadPole2D():
    def __init__(
            self,
            env_name = 'QuadPole2D',
            max_steps = 500,
            timestep = 0.02):
        
        print('Environment init')
        # Quadrotor parameters
        self.mq = 1.5             # Quadrotor mass                 (kg)
        self.mp = 0.5             # Payload mass                   (kg)
        self.I   = 4e-1            # Quadrotor moment of inertia   (kg-m²)
        self.Lq = 0.5             # Quadrotor arm length           (m)
        self.Lp = 0.75            # Rigid tether length            (m)

        # Simulation parameters
        self.gravity = 9.80665     # Gravitational acceleration    (m/s²)
        self.timestep = timestep   # Simulation timestep           (s)
        self.max_steps = max_steps # Maximum number of steps

        # Environment Parameters
        self.spatial_bounds = ((-2.0, 2.0), (-2.0, 2.0)) # Spatial bounds (x, y)
        self.balance_radius = 0.25                       # Radius around origin for considering the system balanced (m) (TODO: Removed for now, may add back when doing waypoint control)
        self.env_name = env_name                         # Environment name
        self._is_3d = False                              # 2D environment
        self._xbounds = self.spatial_bounds[0]
        self._zbounds = self.spatial_bounds[1]

        # Hover force per rotor
        self.hover_force = (self.mq + self.mp)*self.gravity/2

        # Initial state dictionary
        self.state_dict = {
            'quadrotor': np.zeros(8),
            'pendulum': np.zeros(4)
        }

        # OpenAI Gym API attributes (not really used here)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=0.0, high=20.0, shape=(2,), dtype=np.float32
        )

    def _wrap_action(self, action):
        """
        Wrap the input action by computing an adjusted force based on the hover force.
        This function clips the provided action to ensure it remains within the range [-1, 1],
        then scales the clipped value by the hover_force, and finally adds it to the hover_force.
        This effectively produces a new action that is a deviation from the hover force baseline,
        bounded appropriately by the clip operation.
        Parameters:
            action (float or array-like): The input action(s) to be modulated. Expected to be within a range that,
                                          when scaled by hover_force, provides a meaningful deviation from the nominal hover force.
        Returns:
            float or array-like: The adjusted force computed as hover_force + (hover_force * clipped_action).
        Note:
            The function assumes that hover_force is defined and properly represents the baseline force required
            to hover. The clipping ensures that the adjustment does not exceed the predefined safe limits.
        """
        return self.hover_force + self.hover_force*np.clip(action, -1, 1)

    def reset(self):
        """
        Resets the environment to its initial state.
        This method resets the internal state variables and counters of the environment, including the step counter,
        time, and balanced time. The quadrotor and pendulum states are initialized:
            - Quadrotor: Set to a fixed state [0, 0, 0, 0, 0, 1, 0].
            - Pendulum: Set based on a randomly sampled angle phi in [-π, π], where the x-component is sin(phi)
                        and the y-component is cos(phi).
        The initial state of the environment is stored separately for potential future reference.
        Returns:
            tuple:
                A tuple containing:
                    - The observation state retrieved from the _get_obs() method.
                    - An info dictionary retrieved from the _get_info() method.
        """
        # Reset the counter variables
        self._steps = 0
        self._time = 0
        self._time_balanced = 0
        self.total_time_balanced = 0

        # Sample a random angle phi for the pendulum
        phi = np.random.uniform(-np.pi, np.pi)

        # Set the initial state of the quadrotor and pendulum
        self.state_dict['quadrotor'] = np.array([0, 0, 0, 0, 0, 1, 0])
        self.state_dict['pendulum'] = np.array([np.sin(phi), np.cos(phi), 0])

        # Save the initial state for potential restarts
        self._initial_state = self.state_dict.copy()

        # Return the initial observation and info
        return self._get_obs(), self._get_info()
    
    def restart(self):
        """
        Resets the environment to its initial state.
        This method reinitializes the state dictionary, step counter, simulation time, 
        and time balanced to their starting values. It then returns the current observation 
        and additional info by calling the _get_obs() and _get_info() methods respectively.
        Returns:
            tuple: A tuple containing the observation and ancillary information.
        """
        # Reset the state dictionary to the initial state
        self.state_dict = self._initial_state.copy()

        # Reset the counter variables
        self._steps = 0
        self._time = 0
        self._time_balanced = 0

        # Return the initial observation and info
        return self._get_obs(), self._get_info()
    
    def _get_obs(self):
        """
        Retrieve the combined observation from the quadrotor and pendulum states.
        This method concatenates the state arrays for the quadrotor and pendulum into a single
        observation vector using numpy's hstack function. The resulting observation can be used
        as the input for further processing or control within the environment.
        Returns:
            np.ndarray: A one-dimensional array combining the quadrotor state followed by the 
            pendulum state.
        Note:
            The function assumes that self.state_dict contains valid 'quadrotor' and 'pendulum'
            keys, each associated with a numpy array representing the respective state.
        """
        return np.hstack((self.state_dict['quadrotor'], self.state_dict['pendulum']))
    
    def _get_info(self):
        """
        Return a dictionary containing metrics for the quadrotor environment.
        Returns:
            dict: A dictionary with the following key-value pair:
                "time_balanced" (bool): Indicates whether the environment has achieved balanced time.
        Function Note:
            This method is used to encapsulate and return relevant state metrics for time balance in the simulation.
        """
        return {'time_balanced': self._time_balanced}
    
    def out_of_bounds(self):
        """
        Check if the quadrotor is out of the defined bounds.
        This method retrieves the x and z positions from the quadrotor's state and
        compares them against the preset x bounds (self._xbounds) and z bounds 
        (self._zbounds). It returns True if either the x position is outside the 
        x bounds or the z position is outside the z bounds, indicating that the 
        quadrotor is out of the permitted area.
        Returns:
            bool: True if the quadrotor is out-of-bounds, False otherwise.
        """
        x, z = self.state_dict['quadrotor'][0:2]
        return (x < self._xbounds[0] or x > self._xbounds[1] or
                z < self._zbounds[0] or z > self._zbounds[1])
    
    def _propogate(self, action):
        """
        Propagates the system state by integrating the dynamics for a single time step.
        This method concatenates the current states of the quadrotor and pendulum into a single state vector,
        applies the dynamics function (_dynamics) using the provided action, and then updates the internal state dictionary
        by splitting the resulting state back into the quadrotor and pendulum components.
        Parameters:
            action (np.ndarray): The action to apply during propagation. The expected structure and type of the action
                                 depend on the specific dynamics model.
        Returns:
            None
        Note:
            This method updates the internal state in-place and does not return a new state.
        """

        state = np.hstack((self.state_dict['quadrotor'], self.state_dict['pendulum']))
        state = self._dynamics(state, action)
        self.state_dict['quadrotor'] = state[:8]
        self.state_dict['pendulum'] = state[8:]
    
    def _dynamics(self, state, control):
        """
        Perform one integration step for the quadrotor-payload dynamics using semi-implicit Euler integration.
        This method integrates the state of a coupled quadrotor-payload system. The input state vector is expected
        to be in the form:
            x, z            : Cartesian positions,
            vx, vz          : Linear velocities,
            s_theta, c_theta: Sin and cos of the quadrotor's pitch angle (theta),
            theta_dot       : Angular velocity of the quadrotor,
            s_phi, c_phi    : Sin and cos of the payload's angle (phi),
            phi_dot         : Angular velocity of the payload.
        Parameters:
            state (list or array-like): The current state vector.
            control (list or array-like): Control inputs [u1, u2], corresponding to the forces from the two rotors.
        Returns:
            list: The updated state vector after one integration step, containing:
                  [x_new, z_new, vx_new, vz_new, s_theta_new, c_theta_new, theta_dot_new,
                   s_phi_new, c_phi_new, phi_dot_new].
        Notes:
            - The computation includes dynamics for both the translational and rotational motion of the quadrotor,
              as well as the motion of the payload modeled as a pendulum.
            - The formulation considers the effect of the combined forces and moments, including gravitational
              acceleration and coupling between the translational and angular accelerations.
            - Semi-implicit Euler integration is used: velocities are updated first based on computed accelerations,
              and then positions are updated using these new velocities.
        """
        # Unpack state variables
        x, z, vx, vz, s_theta, c_theta, theta_dot, s_phi, c_phi, phi_dot = state
        # Unpack control inputs
        u1, u2 = control

        # Parameters
        mq = self.mq             # Quadrotor mass
        mp = self.mp             # Payload mass
        Lq = self.Lq             # Distance from center to rotor
        Lp = self.Lp             # Tether length
        I = self.I               # Quadrotor moment of inertia (2D scalar)
        g = self.gravity         # Gravitational acceleration (positive scalar)
        dt = self.timestep       # Timestep
        
        # Total force from both rotors
        F = u2 + u1
        # Combined mass of quadrotor and pendulum
        M = mq + mp

        # 1. Quadrotor attitude dynamics (theta)
        ddtheta = (Lq / I) * (u2 - u1)
        
        # 2. Pendulum dynamics (phi)
        # Derived from the coupling of translational accelerations with the pendulum equation.
        ddphi = -F * (s_phi * c_theta - s_theta * c_phi) / (mq * Lp)
        
        # 3. Translational dynamics:
        # Equation for x: M*ddx + mp*Lp*( -sin(phi)*phi_dot^2 + cos(phi)*ddphi ) = -sin(theta)*F
        ddx = (-s_theta * F - mp * Lp * c_phi * ddphi + mp * Lp * s_phi * (phi_dot**2)) / M
        
        # Equation for z: M*ddz + mp*Lp*( cos(phi)*phi_dot^2 + sin(phi)*ddphi ) = cos(theta)*F - M*g
        ddz = (c_theta * F - M * g - mp * Lp * s_phi * ddphi - mp * Lp * c_phi * (phi_dot**2)) / M

        # 4. Semi-implicit Euler update:
        # First update the velocities using the computed accelerations.
        vx_new       = vx + ddx * dt
        vz_new       = vz + ddz * dt
        theta_dot_new = theta_dot + ddtheta * dt
        phi_dot_new   = phi_dot + ddphi * dt
        
        # Then update the positions with the new velocities.
        x_new = x + vx_new * dt
        z_new = z + vz_new * dt
        
        # Update sin and cos for theta using chain rule:
        # d/dt(s_theta) = c_theta * theta_dot, and d/dt(c_theta) = -s_theta * theta_dot.
        theta = np.arctan2(s_theta, c_theta)
        s_theta_new = np.sin(theta + theta_dot * dt)
        c_theta_new = np.cos(theta + theta_dot * dt)

        # Update sin and cos for phi using chain rule:
        # d/dt(s_phi) = c_phi * phi_dot, and d/dt(c_phi) = -s_phi * phi_dot.
        phi = np.arctan2(s_phi, c_phi)
        s_phi_new = np.sin(phi + phi_dot * dt)
        c_phi_new = np.cos(phi + phi_dot * dt)

        # Pack the new state vector and return
        new_state = [x_new, z_new, vx_new, vz_new, s_theta_new, c_theta_new, theta_dot_new,
                    s_phi_new, c_phi_new, phi_dot_new]
        
        return new_state

    def step(self, action):
        """
        Take a simulation step in the quadrotor environment.
        This method processes the provided action by wrapping it, propagating the state,
        and computing a reward based on various penalty terms, including position, velocity,
        orientation, and angular velocities for both the quadrotor and payload. It also
        provides a bonus reward if a balanced state is achieved and penalizes heavily for
        out-of-bounds conditions.
        Steps:
            1. Wrap the action via _wrap_action and propagate it via _propogate.
            2. Obtain the updated state, observation, and additional environment info.
            3. Compute individual cost terms:
                  - pos_cost: A combination of absolute deviation and squared deviation from the origin.
                  - vel_cost: Squared velocity penalty for the x and z components.
                  - theta_cost: Deviation penalty using the cosine of the quadrotor's orientation angle.
                  - omega_cost: Penalty for high angular velocity of the quadrotor.
                  - phi_cost: Cubic cost term for the payload orientation.
                  - phi_dot_cost: Squared cost term for the payload angular velocity.
            4. Combine these penalties with respective weights scaled by the timestep to calculate
               the overall reward.
            5. Apply a bonus reward if the vehicle is within a set balance radius, the payload
               orientation is near its target, and the payload's angular velocity is low.
            6. Apply a heavy penalty if the state is determined to be out-of-bounds.
            7. Increment the step count and simulation time.
            8. Determine if the simulation should be truncated (when the maximum number of steps is
               reached or out-of-bounds) while termination is deliberately kept False.
        Args:
            action: The action applied at the current time step. It is first wrapped to conform
                    to the expected action space.
        Returns:
            tuple: A tuple containing:
                - state: The updated state of the environment after applying the action.
                - reward: The computed reward for the step.
                - terminated: A boolean flag indicating episode termination (always False here).
                - truncated: A boolean flag indicating if the episode was truncated (True if the maximum
                             number of steps is reached or the state is out-of-bounds).
                - info: Additional information provided by the environment.
        Note:
            The reward strategy incorporates multiple penalties to ensure the quadrotor and its payload
            maintain desired states, penalizing deviations and rewarding balance. The heavy out-of-bounds
            penalty enforces safe operation within the defined limits.
        """
        
        # Wrap the action
        action = self._wrap_action(action)

        # Propagate the state using the dynamics function
        self._propogate(action)

        # Obtain the updated state, observation, and additional info
        state = self._get_obs()
        info = self._get_info()

        # Compute cost terms
        #pos_cost = np.sum(np.abs(state[0:2])) + np.sum((state[0:2])**2)  # Position cost: L1 and L2 norms
        vel_cost = np.sum(state[2:4]**2)                                 # Velocity cost: L2 norm
        theta_cost = 1 - np.abs(state[5])                                # Quadrotor orientation cost (cosine of theta): 1 - cos(theta)
        omega_cost = state[6]**2                                         # Quadrotor angular velocity cost: L2 norm
        phi_cost = state[8]**3                                           # Payload orientation cost: L3 norm
        phi_dot_cost = state[9]**2                                       # Payload angular velocity cost: L2 norm

        # Compute the reward using the timestep-scaled cost terms
        reward = 0
        reward += self.timestep * np.sum([
            #- 15.0*pos_cost,
            - 0.5*vel_cost,       
            - 5.0*theta_cost,    
            - 5*omega_cost,
            - (25.0*phi_cost - 25.0)*(1/(1 + 5*phi_dot_cost)) # Balancing reward
        ])

        # Apply a bonus reward if the quadrotor is balanced
        if state[8] < -0.95 and abs(state[9]) < 0.1: # Removed: np.sum(state[0:2]**2)**0.5 < self.balance_radius
            #print('BALANCED')
            reward += 100*self.timestep
            self._time_balanced += self.timestep
            self.total_time_balanced += 1
        else:
            self._time_balanced = 0

        # Increment the step count and simulation time
        self._steps += 1
        self._time += self.timestep

        # Apply heavy penalty if out-of-bounds
        oob = self.out_of_bounds()
        if oob:
            reward -= 1_000 * self.timestep

        # Determine if the episode should be truncated
        truncated = self._steps >= self.max_steps or oob
        terminated = False

        return state, reward, terminated, truncated, info

    def render(self, ax, observation=None, color='black', alpha=1.0):
        """
        Renders the quadrotor and its suspended payload on the given matplotlib axis.
        It draws the quadrotor body, its arms with rotors, and the tethered payload.
        Note:
            - The state vector is expected to be of length 10 with the following elements:
              [x, z, vx, vz, s_theta, c_theta, theta_dot, s_phi, c_phi, phi_dot]
            - If an observation is provided, it is used as the state vector; otherwise, `self.state`
              is assumed to hold the current state.
            - The appearance of the render (color and transparency) can be adjusted via the `color`
              and `alpha` arguments.
        Parameters:
            ax (matplotlib.axes.Axes): The axis object on which to render the quadrotor.
            observation (array-like, optional): The state vector to be rendered. If not provided,
                                                  `self.state` is used.
            color (str or tuple, optional): The color used for drawing the quadrotor, arms, rotors,
                                            tether, and payload. Default is 'black'.
            alpha (float, optional): The transparency (opacity) level for the rendered elements.
                                     Default is 1.0.
        Returns:
            None
        """
        # --- State Extraction ---
        # Assume self.state holds the current 2D state if observation is not provided.
        if observation is None:
            state = self.state
        else:
            state = observation

        # Unpack the state vector
        x, z, vx, vz, s_theta, c_theta, theta_dot, s_phi, c_phi, phi_dot = state
        pos = np.array([x, z])

        ax.axhline(0, color=(0, 0, 0, 0.3), lw=1, linestyle='--')
        ax.axvline(0, color=(0, 0, 0, 0.3), lw=1, linestyle='--') 

        # Draw circle around origin
        radius = 0.25  # Radius of the circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = radius * np.cos(theta)
        circle_y = radius * np.sin(theta)
        ax.plot(circle_x, circle_y, color=(0, 0, 0, 0.3), lw=1, linestyle='--')
        
        # --- Quadrotor Rendering ---
        # Draw the quadrotor body as a scatter point.
        ax.scatter(pos[0], pos[1], color=color, s=50, zorder=3, alpha=alpha)

        # Compute the rotation matrix for the quadrotor (2D)
        R = np.array([[c_theta, -s_theta],
                    [s_theta, c_theta]])

        # Arm and rotor parameters
        Lq = self.Lq  # Arm length
        rotor_line_length = 0.4 * Lq  # Length of line representing the rotor
        half_line = rotor_line_length / 2.0

        # Define rotor offsets in the body frame (one on each side)
        rotor_offset1 = np.array([Lq, 0.2])
        rotor_offset2 = np.array([-Lq, 0.2])

        # Transform rotor positions to inertial frame
        rotor1 = pos + R @ rotor_offset1
        rotor2 = pos + R @ rotor_offset2

        # Draw arms: lines from the quadrotor center to each rotor position
        ax.plot([pos[0], rotor1[0]], [pos[1], rotor1[1]], color=color, lw=2, alpha=alpha)
        ax.plot([pos[0], rotor2[0]], [pos[1], rotor2[1]], color=color, lw=2, alpha=alpha)

        # --- Rotated Rotor Representation ---
        # Create a rotor line in its local frame (centered at zero)
        rotor_line_local = np.array([[-half_line, half_line], [0, 0]])

        # Rotate the rotor line by the same rotation matrix R (to align with vehicle orientation)
        rotor_line_rotated = R @ rotor_line_local  # shape (2,2)

        # Draw the rotor line for rotor1 at its computed position
        ax.plot(rotor_line_rotated[0, :] + rotor1[0],
                rotor_line_rotated[1, :] + rotor1[1],
                color=color, lw=3, alpha=alpha)

        # Draw the rotor line for rotor2 similarly
        ax.plot(rotor_line_rotated[0, :] + rotor2[0],
                rotor_line_rotated[1, :] + rotor2[1],
                color=color, lw=3, alpha=alpha)

        # --- Payload (Suspended Load) Rendering ---
        # Compute payload position using the pendulum angle.
        # Reconstruct phi from sin(phi) and cos(phi)
        Lp = self.Lp
        payload_pos = pos + np.array([Lp * s_phi, -Lp * c_phi])

        # Draw the tether as a line from the quadrotor to the payload
        ax.plot([pos[0], payload_pos[0]],
                [pos[1], payload_pos[1]],
                color=color, lw=1.5, alpha=alpha)

        # Draw the payload as a small circle (scatter point)
        ax.scatter(payload_pos[0], payload_pos[1], color=color, s=50, zorder=3, alpha=alpha)

        # --- Aesthetic Adjustments ---
        # ax.set_aspect('equal')
        ax.set_xlim(self._xbounds)
        ax.set_ylim(self._zbounds)
        ax.set_xticks([])
        ax.set_yticks([])