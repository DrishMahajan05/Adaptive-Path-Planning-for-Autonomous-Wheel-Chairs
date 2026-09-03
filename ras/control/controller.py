"""
controller.py
=============
Low-level actuation controllers for the differential-drive wheelchair.

Provides two controller implementations with the SAME interface:
  1. DifferentialDriveController  — PID-based (two independent wheel PIDs)
  2. MPCController                — Model Predictive Control (scipy L-BFGS-B)

Both expose:
    compute(v_cmd, omega_cmd, v_left_actual, v_right_actual)
        → (torque_left, torque_right)
    reset()

The MPC variant jointly optimises both wheel torques over a 10-step
prediction horizon, yielding smoother, anticipatory torque profiles
that reduce jerk and improve cornering precision.
"""

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
#  Generic PID Controller
# ---------------------------------------------------------------------------

class PIDController:
    """
    Discrete PID controller with anti-windup clamping.

    Parameters
    ----------
    kp, ki, kd : float
        Proportional, integral, derivative gains.
    dt : float
        Control loop timestep (seconds).
    output_limits : tuple (lo, hi)
        Symmetric clamp on the output signal.
    """

    def __init__(self, kp: float, ki: float, kd: float,
                 dt: float, output_limits: tuple = (-10.0, 10.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.lo, self.hi = output_limits

        # Internal state
        self._integral   = 0.0
        self._prev_error = 0.0

    def update(self, error: float) -> float:
        """
        Compute PID output for the given error.

        Parameters
        ----------
        error : float
            Signed error = (desired - actual).

        Returns
        -------
        float
            Control signal, clamped to output_limits.
        """
        # Proportional
        p = self.kp * error

        # Integral with anti-windup
        self._integral += error * self.dt
        i = self.ki * self._integral

        # Derivative (finite difference)
        d = self.kd * (error - self._prev_error) / self.dt
        self._prev_error = error

        # Sum and clamp
        output = p + i + d
        output = np.clip(output, self.lo, self.hi)

        # Anti-windup: if output is saturated, freeze integral
        if output == self.lo or output == self.hi:
            self._integral -= error * self.dt

        return float(output)

    def reset(self):
        """Zero all internal state."""
        self._integral   = 0.0
        self._prev_error = 0.0


# ---------------------------------------------------------------------------
#  Differential-Drive Controller (PID)
# ---------------------------------------------------------------------------

class DifferentialDriveController:
    """
    Converts (v_linear, omega) commands into per-wheel torques via PID.

    The differential-drive kinematic model:
        v_left  = v - omega * L / 2
        v_right = v + omega * L / 2

    where L = wheel_base.

    Each wheel has its own PID loop tracking the desired tangential speed.
    """

    def __init__(self, wheel_base: float, dt: float,
                 kp: float = 8.0, ki: float = 0.5, kd: float = 0.3,
                 torque_limit: float = 10.0):
        """
        Parameters
        ----------
        wheel_base : float
            Distance between drive wheel centres (m).
        dt : float
            Control loop timestep (s).
        kp, ki, kd : float
            PID gains for each wheel speed controller.
        torque_limit : float
            Maximum absolute torque output per wheel (N·m).
        """
        self.wheel_base = wheel_base
        self.dt = dt

        self.pid_left  = PIDController(kp, ki, kd, dt,
                                        output_limits=(-torque_limit,
                                                       torque_limit))
        self.pid_right = PIDController(kp, ki, kd, dt,
                                        output_limits=(-torque_limit,
                                                       torque_limit))

    # ── Main interface ─────────────────────────────────────────────

    def compute(self, v_cmd: float, omega_cmd: float,
                v_left_actual: float, v_right_actual: float):
        """
        Compute wheel torques to track the commanded body velocity.

        Parameters
        ----------
        v_cmd : float
            Desired forward speed (m/s).
        omega_cmd : float
            Desired yaw rate (rad/s).
        v_left_actual, v_right_actual : float
            Current measured tangential speeds of left/right wheels (m/s).

        Returns
        -------
        (torque_left, torque_right) : tuple of float
        """
        # Desired per-wheel tangential speeds
        v_left_des  = v_cmd - omega_cmd * self.wheel_base / 2.0
        v_right_des = v_cmd + omega_cmd * self.wheel_base / 2.0

        # PID errors
        err_left  = v_left_des  - v_left_actual
        err_right = v_right_des - v_right_actual

        torque_left  = self.pid_left.update(err_left)
        torque_right = self.pid_right.update(err_right)

        return torque_left, torque_right

    def reset(self):
        """Reset both wheel PID controllers."""
        self.pid_left.reset()
        self.pid_right.reset()


# ---------------------------------------------------------------------------
#  MPC Controller (Model Predictive Control)
# ---------------------------------------------------------------------------

class MPCController:
    """
    Receding-horizon MPC controller for a differential-drive wheelchair.

    Jointly optimises both wheel torques over a prediction horizon using
    scipy's L-BFGS-B solver.  The cost function penalises:
      - Tracking error  (desired vs predicted wheel speeds)
      - Control effort   (total torque magnitude)
      - Jerk             (torque rate-of-change for passenger comfort)

    The MPC runs at a lower rate (default 50 Hz) than the physics engine
    (500 Hz) and holds torques constant between solves.

    Same interface as DifferentialDriveController:
        compute(v_cmd, omega_cmd, v_left, v_right) → (τ_L, τ_R)
        reset()
    """

    def __init__(self,
                 wheel_base: float,
                 dt: float,
                 torque_limit: float = 10.0,
                 horizon: int = 10,
                 dt_mpc: float = 0.02,
                 Q_track: float = 50.0,
                 R_effort: float = 0.1,
                 Q_jerk: float = 2.0,
                 wheel_radius: float = 0.20,
                 chassis_mass: float = 80.0,
                 wheel_mass: float = 2.0):
        """
        Parameters
        ----------
        wheel_base : float
            Distance between drive wheel centres (m).
        dt : float
            Physics timestep (s) — typically 0.002.
        torque_limit : float
            Max absolute torque per wheel (N·m).
        horizon : int
            MPC prediction horizon (number of steps).
        dt_mpc : float
            MPC timestep (s) — controls how far ahead each step looks.
        Q_track : float
            Weight for speed tracking error in cost function.
        R_effort : float
            Weight for control effort (torque magnitude).
        Q_jerk : float
            Weight for torque rate-of-change (smoothness).
        wheel_radius : float
            Drive wheel radius (m).
        chassis_mass : float
            Total chassis mass (kg).
        wheel_mass : float
            Mass per drive wheel (kg).
        """
        self.wheel_base   = wheel_base
        self.dt           = dt
        self.torque_limit = torque_limit
        self.N            = horizon
        self.dt_mpc       = dt_mpc
        self.Q_track      = Q_track
        self.R_effort     = R_effort
        self.Q_jerk       = Q_jerk

        # Wheel dynamics: effective inertia per wheel
        # Each wheel shares the chassis load with the other wheel and
        # the two casters.  ~15% of chassis mass per drive wheel gives
        # MPC torques that match MuJoCo's actual accelerations well.
        m_share = chassis_mass * 0.15
        I_coupled = m_share * wheel_radius ** 2
        I_wheel   = 0.5 * wheel_mass * wheel_radius ** 2
        self.I_eff = I_coupled + I_wheel  # effective rotational inertia

        # MPC runs every mpc_interval physics steps
        self.mpc_interval = max(1, int(round(dt_mpc / dt)))
        # Start at interval-1 so the FIRST compute() triggers an MPC solve
        self._step_count  = self.mpc_interval - 1

        # Warm-start: previous solution
        self._u_prev = np.zeros(2 * horizon)
        self._prev_torques = np.zeros(2)  # last applied torques (for jerk)
        self._held_torques = (0.0, 0.0)   # torques held between MPC solves
        self._cold_start   = True          # True until first solve

        # Solver bounds
        self._bounds = [(-torque_limit, torque_limit)] * (2 * horizon)

        # Stats
        self.fallback_count = 0

    def compute(self, v_cmd: float, omega_cmd: float,
                v_left_actual: float, v_right_actual: float):
        """
        Compute wheel torques via MPC or hold previous between solves.

        Same interface as DifferentialDriveController.
        """
        self._step_count += 1

        if self._step_count >= self.mpc_interval:
            self._step_count = 0

            # Desired per-wheel tangential speeds
            v_left_des  = v_cmd - omega_cmd * self.wheel_base / 2.0
            v_right_des = v_cmd + omega_cmd * self.wheel_base / 2.0

            torques = self._solve_mpc(
                v_left_actual, v_right_actual,
                v_left_des, v_right_des)

            self._held_torques = torques
            self._prev_torques = np.array(torques)
            self._cold_start = False

        return self._held_torques

    def _solve_mpc(self, vL0, vR0, vL_des, vR_des):
        """
        Solve the MPC optimisation problem.

        Decision variables: u = [τ_L0, τ_R0, τ_L1, τ_R1, ..., τ_L(N-1), τ_R(N-1)]

        Dynamics (per wheel): v[k+1] = v[k] + (τ[k] / I_eff) * dt_mpc
        """
        N  = self.N
        dt = self.dt_mpc
        I  = self.I_eff
        Qt = self.Q_track
        Re = self.R_effort
        # Disable jerk penalty on cold start so first solve can be aggressive
        Qj = 0.0 if self._cold_start else self.Q_jerk
        prev = self._prev_torques.copy()

        def cost_and_grad(u):
            cost = 0.0
            grad = np.zeros_like(u)

            vL, vR = vL0, vR0

            for k in range(N):
                tL = u[2 * k]
                tR = u[2 * k + 1]

                # Predicted next speeds
                vL_next = vL + (tL / I) * dt
                vR_next = vR + (tR / I) * dt

                # Tracking error (at end of step)
                eL = vL_next - vL_des
                eR = vR_next - vR_des
                cost += Qt * (eL ** 2 + eR ** 2)
                grad[2 * k]     += Qt * 2 * eL * (dt / I)
                grad[2 * k + 1] += Qt * 2 * eR * (dt / I)

                # Control effort
                cost += Re * (tL ** 2 + tR ** 2)
                grad[2 * k]     += Re * 2 * tL
                grad[2 * k + 1] += Re * 2 * tR

                # Jerk (change from previous step)
                if k == 0:
                    dL = tL - prev[0]
                    dR = tR - prev[1]
                    grad[0] += Qj * 2 * dL
                    grad[1] += Qj * 2 * dR
                else:
                    dL = tL - u[2 * (k - 1)]
                    dR = tR - u[2 * (k - 1) + 1]
                    grad[2 * k]         += Qj * 2 * dL
                    grad[2 * k + 1]     += Qj * 2 * dR
                    grad[2 * (k - 1)]     -= Qj * 2 * dL
                    grad[2 * (k - 1) + 1] -= Qj * 2 * dR

                cost += Qj * (dL ** 2 + dR ** 2)

                vL, vR = vL_next, vR_next

            return cost, grad

        try:
            result = minimize(
                cost_and_grad,
                self._u_prev,
                method='L-BFGS-B',
                jac=True,
                bounds=self._bounds,
                options={'maxiter': 30, 'ftol': 1e-8},
            )

            u_opt = result.x

            # Warm-start next solve by shifting the solution
            self._u_prev[:] = 0.0
            self._u_prev[:-2] = u_opt[2:]  # shift left by one step
            self._u_prev[-2:] = u_opt[-2:]  # repeat last

            return (float(u_opt[0]), float(u_opt[1]))

        except Exception:
            # Proportional fallback
            self.fallback_count += 1
            K_fb = 5.0
            tL = np.clip(K_fb * (vL_des - vL0),
                         -self.torque_limit, self.torque_limit)
            tR = np.clip(K_fb * (vR_des - vR0),
                         -self.torque_limit, self.torque_limit)
            return (float(tL), float(tR))

    def reset(self):
        """Reset MPC internal state."""
        self._u_prev[:] = 0.0
        self._prev_torques[:] = 0.0
        self._held_torques = (0.0, 0.0)
        self._step_count = self.mpc_interval - 1
        self._cold_start = True
        self.fallback_count = 0
