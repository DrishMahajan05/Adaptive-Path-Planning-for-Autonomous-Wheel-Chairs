"""Quick integration test for the new DXF map."""
from wheelchair_model import WheelchairPhysics
from controller import DifferentialDriveController
from path_planner import PathPlanner
from obstacles import ObstacleManager

# 1. Physics
physics = WheelchairPhysics()
s = physics.get_state()
print(f"Start: ({s['x']:.2f}, {s['y']:.2f})")

# 2. Controller
ctrl = DifferentialDriveController(
    wheel_base=physics.wheel_base, dt=physics.dt,
    kp=15.0, ki=2.0, kd=2.5, torque_limit=10.0)

# 3. Planner
planner = PathPlanner(
    v_max=1.2, a_x_lb=-1.0, a_x_ub=0.1,
    a_y_max=1.5, tau_c=0.1, capture_radius=0.60)

# 4. Obstacles
obs = ObstacleManager(
    num_obstacles=3, radius_range=(0.12, 0.30),
    speed_range=(0.2, 0.5), spawn_range=(5.0, 30.0), boundary=80.0)
obs.spawn()

# 5. Add a waypoint and run 100 steps
planner.add_waypoint(s['x'] + 5.0, s['y'] + 5.0)
print(f"Nav waypoints: {len(list(planner.all_nav_waypoints))}")
print(f"Wall segments (guard): {len(planner._wall_guard.WALL_SEGMENTS)}")
print(f"Graph nodes: {len(planner._graph.NODES)}")

for _ in range(100):
    state = physics.get_state()
    obs.step(physics.dt)
    planner.hrvo.update_obstacles(obs.as_planner_obstacles())
    v, w = planner.plan(state)
    tl, tr = ctrl.compute(v, w, state['v_left'], state['v_right'])
    physics.set_ctrl(tl, tr)
    physics.step()

s2 = physics.get_state()
print(f"After 100 steps: ({s2['x']:.2f}, {s2['y']:.2f})")
print("Integration test PASSED")
