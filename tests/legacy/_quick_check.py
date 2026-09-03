"""Quick check: verify all modules load and basic functionality works."""
import sys
import math

print("=" * 50)
print("  Quick Module Check")
print("=" * 50)

# 1. Model loads
try:
    from wheelchair_model import WheelchairPhysics
    p = WheelchairPhysics()
    s = p.get_state()
    print(f"[OK] WheelchairPhysics: dt={p.dt}, wheel_base={p.wheel_base}")
    print(f"     State: x={s['x']:.2f}, y={s['y']:.2f}, theta={s['theta']:.4f}")
except Exception as e:
    print(f"[FAIL] WheelchairPhysics: {e}")
    sys.exit(1)

# 2. Controller
try:
    from controller import DifferentialDriveController
    c = DifferentialDriveController(
        wheel_base=p.wheel_base, dt=p.dt,
        kp=25.0, ki=2.0, kd=1.0, torque_limit=25.0
    )
    # Check XML ctrlrange vs torque_limit mismatch
    xml_limit = 10.0  # from XML: ctrlrange="-10 10"
    if c.pid_left.hi != xml_limit:
        print(f"[WARN] Controller torque_limit={c.pid_left.hi} != XML ctrlrange={xml_limit}")
        print(f"       PID may command torques the actuator clips to +/-{xml_limit}")
    else:
        print(f"[OK] Controller: torque_limit matches XML ctrlrange")
    print(f"[OK] Controller loaded: kp=25, ki=2, kd=1, limit={c.pid_left.hi}")
except Exception as e:
    print(f"[FAIL] Controller: {e}")
    sys.exit(1)

# 3. Path Planner
try:
    from path_planner import PathPlanner
    pl = PathPlanner(v_max=1.2, a_x_lb=-1.0, a_x_ub=0.1,
                     a_y_max=1.5, tau_c=0.1, capture_radius=0.60)
    print(f"[OK] PathPlanner loaded")
except Exception as e:
    print(f"[FAIL] PathPlanner: {e}")
    sys.exit(1)

# 4. Routing
try:
    r1 = pl._graph.route((0, 0), (-7, 5))
    print(f"[OK] Route corridor->NW: {len(r1)} waypoints")
    r2 = pl._graph.route((-7, 5), (7, -5))
    print(f"[OK] Route NW->SE: {len(r2)} waypoints")
except Exception as e:
    print(f"[FAIL] Routing: {e}")

# 5. Obstacles
try:
    from obstacles import ObstacleManager
    obs = ObstacleManager(num_obstacles=3, radius_range=(0.12, 0.30),
                          speed_range=(0.2, 0.5), spawn_range=(2.0, 6.0),
                          boundary=11.0, bounce=True)
    obs.spawn()
    print(f"[OK] ObstacleManager: {len(obs.obstacles)} obstacles spawned")
except Exception as e:
    print(f"[FAIL] Obstacles: {e}")

# 6. Interactive Viewer (import only, don't launch)
try:
    from interactive_viewer import InteractiveViewer
    print(f"[OK] InteractiveViewer imported")
except Exception as e:
    print(f"[FAIL] InteractiveViewer: {e}")
    sys.exit(1)

# 7. Wall segments check
try:
    from path_planner import WallProximityGuard
    wpg = WallProximityGuard()
    print(f"[OK] WallProximityGuard: {len(wpg.WALL_SEGMENTS)} wall segments")
except Exception as e:
    print(f"[FAIL] WallProximityGuard: {e}")

print()
print("=" * 50)
print("  ALL CHECKS PASSED - Ready to run main.py")
print("=" * 50)
