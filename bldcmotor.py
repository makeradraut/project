import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class BldcFocModel:
    """
    Implements a dynamic model of a BLDC motor with a cascaded PI-based
    Field-Oriented Control (FOC) system.
    The simulation uses the RK45 method provided by SciPy.
    """
    def __init__(self, motor_params, controller_gains):
        # --- Store parameters ---
        self.p = motor_params
        self.c = controller_gains
        
        # --- Initial State Vector: [id, iq, omega_m, theta_e] ---
        self.X = np.array([0.0, 0.0, 0.0, 0.0])
        
        # --- PI Controller Integral Terms ---
        self.integrator = {
            'speed': 0.0,
            'id': 0.0,
            'iq': 0.0
        }
        
        # --- Setpoints ---
        self.omega_m_ref = 0.0
        self.T_load = 0.0
        
        # --- Voltage limits (saturation) based on DC bus voltage ---
        self.v_limit = self.p['v_dc'] / np.sqrt(3)

    def _pi_controller(self, error, Kp, Ki, integral_term, dt, limit):
        """A generic PI controller with anti-windup."""
        # Proportional term
        p_term = Kp * error
        
        # Update integral term
        integral_term += Ki * error * dt
        
        # Anti-windup: Clamp the integral term
        if limit is not None:
            integral_term = np.clip(integral_term, -limit, limit)
            
        # Controller output
        output = p_term + integral_term
        
        # Final output saturation
        if limit is not None:
            output = np.clip(output, -limit, limit)
            
        return output, integral_term

    def _state_derivative_function(self, t, X, vd, vq):
        """
        Calculates the derivatives of the state vector.
        This function is passed to the ODE solver (RK45).
        X = [id, iq, omega_m, theta_e]
        """
        id, iq, omega_m, theta_e = X
        
        # Electrical angular velocity
        omega_e = self.p['P'] * omega_m

        # Electromagnetic Torque (for SPMSM, Ld=Lq)
        Te = (3/2) * self.p['P'] * self.p['lambda_pm'] * iq

        # State derivatives from the model equations
        did_dt = (1/self.p['Ls']) * (vd - self.p['Rs']*id + omega_e*self.p['Ls']*iq)
        diq_dt = (1/self.p['Ls']) * (vq - self.p['Rs']*iq - omega_e*self.p['Ls']*id - omega_e*self.p['lambda_pm'])
        domega_m_dt = (1/self.p['J']) * (Te - self.T_load - self.p['B']*omega_m)
        dtheta_e_dt = omega_e

        return [did_dt, diq_dt, domega_m_dt, dtheta_e_dt]

    def run_simulation(self, t_span, dt):
        """
        Runs the full simulation loop.
        """
        t_start, t_end = t_span
        t_eval = np.arange(t_start, t_end, dt)
        
        # History lists to store results for plotting
        history = {
            't': [], 'id': [], 'iq': [], 'omega_m': [], 'theta_e': [],
            'id_ref': [], 'iq_ref': [], 'omega_m_ref': [],
            'vd': [], 'vq': [], 'Te': [], 'T_load': []
        }
        
        # --- Main Simulation Loop ---
        for t in t_eval:
            # --- Dynamic Setpoints (to test controller response) ---
            if t < 1.0:
                self.omega_m_ref = 50.0 # rad/s (~1432 RPM)
                self.T_load = 0.5       # Nm
            else:
                self.omega_m_ref = 150.0 # Step change in speed ref (~2387 RPM)
                self.T_load = .5      # Step change in load

            # 1. Get current state from the state vector
            id, iq, omega_m, _ = self.X

            # 2. Outer Speed PI Controller -> generates iq_ref
            e_speed = self.omega_m_ref - omega_m
            iq_ref, self.integrator['speed'] = self._pi_controller(
                e_speed, self.c['Kp_speed'], self.c['Ki_speed'],
                self.integrator['speed'], dt, limit=self.p['i_max']
            )
            
            # 3. Inner Current PI Controllers -> generate vd, vq
            id_ref = 0.0 # Standard FOC for SPMSM
            
            e_d = id_ref - id
            vd, self.integrator['id'] = self._pi_controller(
                e_d, self.c['Kp_curr'], self.c['Ki_curr'],
                self.integrator['id'], dt, limit=self.v_limit
            )
            
            e_q = iq_ref - iq
            vq, self.integrator['iq'] = self._pi_controller(
                e_q, self.c['Kp_curr'], self.c['Ki_curr'],
                self.integrator['iq'], dt, limit=self.v_limit
            )
            
            # 4. Solve the ODE for the next time step using RK45
            sol = solve_ivp(
                fun=lambda t, y: self._state_derivative_function(t, y, vd, vq),
                t_span=[t, t + dt],
                y0=self.X,
                method='RK45'
            )
            
            # Update the state to the result of the integration
            self.X = sol.y[:, -1]

            # 5. Store results for plotting
            Te = (3/2) * self.p['P'] * self.p['lambda_pm'] * self.X[1]
            history['t'].append(t)
            history['id'].append(self.X[0])
            history['iq'].append(self.X[1])
            history['omega_m'].append(self.X[2])
            history['theta_e'].append(self.X[3])
            history['id_ref'].append(id_ref)
            history['iq_ref'].append(iq_ref)
            history['omega_m_ref'].append(self.omega_m_ref)
            history['vd'].append(vd)
            history['vq'].append(vq)
            history['Te'].append(Te)
            history['T_load'].append(self.T_load)
            
        return history

    def plot_results(self, data):
        """Plots the simulation results."""
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle('BLDC FOC Simulation Results', fontsize=16)

        # Speed Plot
        axs[0].plot(data['t'], data['omega_m'], label='Actual Speed ($\omega_m$)')
        axs[0].plot(data['t'], data['omega_m_ref'], 'r--', label='Reference Speed ($\omega_{m,ref}$)')
        axs[0].set_ylabel('Speed (rad/s)')
        axs[0].legend()
        axs[0].grid(True)

        # Currents Plot
        axs[1].plot(data['t'], data['iq'], label='Actual $i_q$')
        axs[1].plot(data['t'], data['iq_ref'], 'r--', label='Reference $i_{q,ref}$')
        axs[1].plot(data['t'], data['id'], label='Actual $i_d$')
        axs[1].plot(data['t'], data['id_ref'], 'k--', label='Reference $i_{d,ref}$')
        axs[1].set_ylabel('Current (A)')
        axs[1].legend()
        axs[1].grid(True)

        # Torque Plot
        axs[2].plot(data['t'], data['Te'], label='Electromagnetic Torque ($T_e$)')
        axs[2].plot(data['t'], data['T_load'], 'g--', label='Load Torque ($T_L$)')
        axs[2].set_ylabel('Torque (Nm)')
        axs[2].legend()
        axs[2].grid(True)

        # Voltage Plot
        axs[3].plot(data['t'], data['vd'], label='Voltage $v_d$')
        axs[3].plot(data['t'], data['vq'], label='Voltage $v_q$')
        axs[3].set_xlabel('Time (s)')
        axs[3].set_ylabel('Voltage (V)')
        axs[3].legend()
        axs[3].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    # ⚙️ Define motor parameters (example: small hobby BLDC motor)
    motor_parameters = {
        'Rs': 0.5,         # Stator Resistance (Ohm)
        'Ls': 0.0015,      # Stator Inductance (H)
        'lambda_pm': 0.02, # Permanent Magnet Flux Linkage (Wb)
        'P': 4,            # Number of pole pairs
        'J': 0.0005,       # Inertia (kg*m^2)
        'B': 0.0001,       # Friction Coefficient (N*m*s)
        'v_dc': 24.0,      # DC bus voltage (V)
        'i_max': 10.0,     # Max current for saturation (A)
    }

    # 📈 Define PI controller gains (tuning is critical for performance)
    controller_gains = {
        'Kp_speed': 0.1,
        'Ki_speed': 1.5,
        'Kp_curr': 5.0,
        'Ki_curr': 500.0,
    }

    # 1. Instantiate the model
    foc_sim = BldcFocModel(motor_parameters, controller_gains)

    # 2. Run the simulation
    sim_time_span = (0, 2.0) # seconds
    sim_dt = 0.0001          # time step in seconds
    results = foc_sim.run_simulation(t_span=sim_time_span, dt=sim_dt)

    # 3. Plot the results
    foc_sim.plot_results(results)