"""
Headless test: simulate wheelchair from (32.37, 73.47) to (32.45, 52.92)
and print step-by-step position data.
"""
import math
import mujoco
from wheelchair_model import WheelchairPhysics
from controller import DifferentialDriveController
from path_planner import PathPlanner
from obstacles import ObstacleManager

# ── Setup ──
physics = WheelchairPhysics()
controller = DifferentialDriveController(
    wheel_base=physics.wheel_base, dt=physics.dt,
    kp=15.0, ki=2.0, kd=2.5, torque_limit=10.0)
planner = PathPlanner(
    v_max=1.2, a_x_lb=-1.0, a_x_ub=0.1,
    a_y_max=1.5, tau_c=0.1, capture_radius=2.0)
obs_mgr = ObstacleManager(num_obstacles=3)

# ── Place wheelchair at start ──
start = (32.37, 73.47)
goal  = (32.45, 52.92)

physics.data.qpos[0] = start[0]
physics.data.qpos[1] = start[1]
physics.data.qpos[3] = 1.0  # qw
physics.data.qpos[4:7] = 0.0  # qx, qy, qz
physics.data.qvel[:] = 0
mujoco.mj_forward(physics.model, physics.data)

# ── Set waypoint ──
state = physics.get_state()
planner._current_pos = (state["x"], state["y"])
planner.add_waypoint(goal[0], goal[1])

wps = list(planner.wap.waypoints)
print(f"Start: ({start[0]:.2f}, {start[1]:.2f})")
print(f"Goal:  ({goal[0]:.2f}, {goal[1]:.2f})")
print(f"Waypoints: {len(wps)}")
for i, (x, y) in enumerate(wps):
    print(f"  [{i}] ({x:.1f}, {y:.1f})")
print()

# ── Simulate ──
dt = physics.dt
max_time = 120.0  # 2 minutes max
steps = int(max_time / dt)
print_interval = int(1.0 / dt)  # every 1 second

print(f"{'Time':>6s}  {'X':>7s}  {'Y':>7s}  {'Heading':>8s}  {'Speed':>6s}  {'Omega':>6s}  {'Dist':>6s}  {'WP#':>4s}")
print("-" * 70)

reached = False
for step in range(steps):
    state = physics.get_state()
    x, y = state["x"], state["y"]
    theta = state["theta"]
    
    # Plan
    v_cmd, omega_cmd = planner.plan(state)
    
    # Control
    torque_l, torque_r = controller.compute(
        v_cmd, omega_cmd, state["v_left"], state["v_right"])
    
    # Actuate
    physics.set_ctrl(torque_l, torque_r)
    physics.step()
    
    # Update obstacles
    obs_mgr.step(dt)
    planner.hrvo.update_obstacles(obs_mgr.as_planner_obstacles())
    
    # Print every second
    if step % print_interval == 0:
        dist = math.hypot(goal[0] - x, goal[1] - y)
        speed = math.hypot(state["vx"], state["vy"])
        n_wp = len(planner.wap.waypoints)
        t = step * dt
        print(f"{t:6.1f}s  {x:7.2f}  {y:7.2f}  {math.degrees(theta):7.1f}°  "
              f"{speed:5.2f}  {omega_cmd:+5.2f}  {dist:5.1f}m  {n_wp:4d}")
    
    # Check if reached goal
    dist_to_goal = math.hypot(goal[0] - x, goal[1] - y)
    if dist_to_goal < 1.0:
        t = step * dt
        print(f"\n*** GOAL REACHED at t={t:.1f}s  pos=({x:.2f}, {y:.2f})  "
              f"dist={dist_to_goal:.2f}m ***")
        reached = True
        break

if not reached:
    state = physics.get_state()
    final_dist = math.hypot(goal[0] - state["x"], goal[1] - state["y"])
    print(f"\n*** TIMEOUT at t={max_time:.0f}s  pos=({state['x']:.2f}, {state['y']:.2f})  "
          f"dist_to_goal={final_dist:.1f}m ***")
