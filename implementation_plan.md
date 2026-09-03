# Autonomous Electric Wheelchair Simulation with dm_control

Real-time interactive simulation of an autonomous electric wheelchair navigating user-drawn paths, implementing the 4-stage classical control pipeline from Jung et al. (2020).

## User Review Required

> [!IMPORTANT]
> This project requires **dm_control**, **mujoco**, and **glfw** to be installed. The viewer leverages dm_control's built-in viewer with GLFW hooks for mouse interactivity. Please confirm these dependencies are available, or I will install them.

> [!WARNING]
> dm_control's viewer does not expose a first-class mouse-click callback API. The implementation will use `mujoco.viewer` (the native MuJoCo Python viewer) which provides better support for custom keyboard/mouse callbacks and overlay rendering. If you strictly require `dm_control.viewer`, the approach will be more involved (monkey-patching GLFW). Please confirm preference.

## Proposed Changes

All new files live under `c:\Users\DELL\OneDrive\Desktop\Projects\RAS\`.

---

### Wheelchair MuJoCo Model

#### [NEW] [wheelchair_model.py](file:///c:/Users/DELL/OneDrive/Desktop/Projects/RAS/wheelchair_model.py)

- **`WHEELCHAIR_XML`** — inline MuJoCo XML string:
  - Ground plane with chequered texture
  - `chassis` free-body (box geometry ~0.6 × 0.4 × 0.15 m)
  - Two rear drive wheels (`left_wheel`, `right_wheel`) — hinge joints on Y-axis, actuated via `motor` actuators
  - Two front caster assemblies — each has a vertical yaw hinge (free-spinning) and a wheel hinge (free-spinning), providing passive steering
  - Sensor sites for IMU / heading readout
- **`class WheelchairPhysics`**:
  - `__init__()` — loads XML via `mujoco.MjModel.from_xml_string()`, creates `MjData`
  - `get_state()` → returns `(x, y, theta, vx, vy, omega)` from `qpos`/`qvel`
  - `set_ctrl(left_torque, right_torque)` — sets `data.ctrl[0:2]`
  - `step()` — calls `mujoco.mj_step()`
  - Properties: `model`, `data`, `wheel_radius`, `wheel_base`

---

### PID Controller & Differential-Drive Actuation

#### [NEW] [controller.py](file:///c:/Users/DELL/OneDrive/Desktop/Projects/RAS/controller.py)

- **`class PIDController`**:
  - `__init__(kp, ki, kd, dt, output_limits)` — stores gains and integral/derivative state
  - `update(error)` → returns control output with anti-windup clamping
  - `reset()` — zeros internal state
- **`class DifferentialDriveController`**:
  - Holds two `PIDController` instances (left/right wheel speed tracking)
  - `compute(desired_v_left, desired_v_right, actual_v_left, actual_v_right)` → `(torque_left, torque_right)`
  - Converts linear/angular velocity commands from the planner to per-wheel desired speeds via `v_l = v - ω·L/2`, `v_r = v + ω·L/2`

---

### 4-Stage Path Planning Pipeline (Jung et al. 2020)

#### [NEW] [path_planner.py](file:///c:/Users/DELL/OneDrive/Desktop/Projects/RAS/path_planner.py)

- **`class WaypointAttitudePlanner` (WAP)**:
  - `waypoints: deque` — FIFO queue of `(x, y)` from user clicks
  - `add_waypoint(x, y)` — appends to queue
  - `get_desired_attitude(cx, cy)` → `theta_d` — heading angle toward active waypoint
  - `get_distance_to_active()` — Euclidean distance to current target
  - `advance()` — pops active waypoint when within threshold; returns next
  - `active_waypoint` property — current target `(x, y)` or `None`

- **`class SpeedProfileDesigner` (SPD)**:
  - `__init__(v_max, a_x_lb, a_x_ub, v_min_turn)` — max speed, accel bounds, low-speed for sharp turns
  - `compute(dist_to_wp, curvature, current_v)` → `v_desired` — trapezoidal profile: accelerate → cruise → decelerate as approaching waypoint; near-zero at sharp-turn waypoints
  - Uses kinematic equation `v² = v₀² + 2·a·d` for deceleration planning

- **`class AngularRateGainAdapter` (ARGA)**:
  - `__init__(a_y_max, tau_c)` — max lateral accel, system time delay
  - `compute(v, theta_error)` → `K` — angular rate gain clamped so `v · K · |θ_err| ≤ a_y_max`
  - `omega_desired(v, theta_error)` → `ω = K · theta_error` — the desired angular velocity

- **`class ModifiedHRVO` (mHRVO)**:
  - `__init__(tau_c, safety_margin)` — time delay, obstacle clearance
  - `obstacles: list` — list of `(x, y, vx, vy, radius)` (initially empty)
  - `compute(pos, vel, heading, v_desired, omega_desired)` → `(v_safe, omega_safe)` — if no obstacles, passes through; otherwise clips velocity to stay outside HRVO cones
  - `add_obstacle(…)` / `clear_obstacles()` — hooks for future dynamic integration

- **`class PathPlanner`** — orchestrator:
  - Composes WAP → SPD → ARGA → mHRVO in `plan(state)` → `(v_cmd, omega_cmd)`
  - Called once per physics timestep from the main loop

---

### Interactive Viewer & Main Loop

#### [NEW] [interactive_viewer.py](file:///c:/Users/DELL/OneDrive/Desktop/Projects/RAS/interactive_viewer.py)

- **`class InteractiveViewer`**:
  - `__init__(wheelchair_physics, controller, path_planner)` — stores references to all modules
  - Uses `mujoco.viewer.launch_passive()` for a non-blocking viewer
  - Registers custom keyboard callback (e.g., `R` to reset, `Q` to quit)
  - On mouse double-click with Ctrl held: ray-casts from camera through cursor to ground plane, extracts `(x, y)`, appends to WAP, and creates an overlay sphere marker
  - `_render_waypoints(viewer)` — draws small red spheres at each waypoint location using `mujoco.mjv_initGeom()` to create custom `mjvGeom` objects in the viewer scene
  - `run()` — main simulation loop:
    1. Read wheelchair state
    2. Call `path_planner.plan(state)` → `(v_cmd, ω_cmd)`
    3. Convert to per-wheel speeds
    4. Call `controller.compute(…)` → `(τ_l, τ_r)`
    5. Apply to `wheelchair_physics.set_ctrl(…)`
    6. Step physics
    7. Sync viewer
    8. Repeat until window closed

---

### Entry Point

#### [NEW] [main.py](file:///c:/Users/DELL/OneDrive/Desktop/Projects/RAS/main.py)

- Instantiates all four classes
- Calls `interactive_viewer.run()`
- Clean shutdown on exit

---

## Verification Plan

### Automated Tests

Since this is a real-time interactive simulation, automated unit tests are limited. The following smoke-test script will be created:

#### [NEW] [test_smoke.py](file:///c:/Users/DELL/OneDrive/Desktop/Projects/RAS/test_smoke.py)

**Run:** `python test_smoke.py` from the project directory.

Tests:
1. **XML loads** — `WheelchairPhysics()` instantiates without error; `model.nq > 0`
2. **PID output** — `PIDController` returns non-zero output for non-zero error
3. **WAP queue** — add waypoints, verify `active_waypoint` cycles correctly
4. **SPD profile** — `v_desired` decreases as `dist_to_wp → 0`
5. **ARGA gain clamping** — `K` is reduced at high speeds
6. **mHRVO pass-through** — with no obstacles, output equals input

### Manual Verification

**Run:** `python main.py` from the project directory.

1. **Window appears** — a MuJoCo viewer window opens showing the wheelchair on a ground plane
2. **Click to add waypoints** — hold **Ctrl** and **double-click** on the ground; red sphere markers should appear
3. **Wheelchair navigates** — the wheelchair drives toward each waypoint in sequence, decelerating as it approaches, then turning toward the next
4. **Speed profile visible** — wheelchair accelerates smoothly and decelerates near turns (observable visually)
5. **Press Q** — simulation exits cleanly
