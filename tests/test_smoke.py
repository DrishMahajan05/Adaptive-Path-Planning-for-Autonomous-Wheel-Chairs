# -*- coding: utf-8 -*-
"""
test_smoke.py
=============
Smoke tests for the wheelchair simulation modules.

Run:  python test_smoke.py

Tests verify that:
  1. MuJoCo XML loads and model is valid
  2. PID controller produces non-zero output for non-zero error
  3. WAP waypoint queue advances correctly
  4. SPD speed decreases as distance → 0
  5. ARGA gain is reduced at high speeds
  6. mHRVO passes through commands when no obstacles are present
  7. Full pipeline returns valid commands
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─── Test utilities ───────────────────────────────────────────────

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS]  {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL]  {name}"
        if detail:
            msg += f"  -- {detail}"
        print(msg)


# ─── 1. Wheelchair model ─────────────────────────────────────────

def test_model():
    print("\n--- WheelchairPhysics ---")
    from ras.physics.wheelchair_model import WheelchairPhysics

    wp = WheelchairPhysics()
    check("XML loads without error", wp.model is not None)
    check("nq > 0 (generalised coordinates exist)", wp.model.nq > 0)
    check("2 actuators present", wp.model.nu == 2)

    state = wp.get_state()
    check("get_state() returns dict with keys",
          all(k in state for k in ["x", "y", "theta", "vx", "vy", "omega"]))

    wp.set_ctrl(1.0, 1.0)
    wp.step()
    check("Step executes without error", True)

    state2 = wp.get_state()
    from ras.map.dxf_parser import get_start_position
    sx, sy = get_start_position()
    check("Wheelchair is near start position after one step",
          abs(state2["x"] - sx) < 1.0 and abs(state2["y"] - sy) < 1.0)


# ─── 2. PID Controller ───────────────────────────────────────────

def test_pid():
    print("\n--- PIDController ---")
    from ras.control.controller import PIDController

    pid = PIDController(kp=2.0, ki=0.1, kd=0.05, dt=0.01)
    out = pid.update(1.0)
    check("Non-zero output for error=1.0", abs(out) > 0.0,
          f"got {out:.4f}")

    pid.reset()
    out0 = pid.update(0.0)
    check("Zero output for error=0.0", abs(out0) < 1e-6)


# ─── 3. Differential Drive Controller ────────────────────────────

def test_dd_controller():
    print("\n--- DifferentialDriveController ---")
    from ras.control.controller import DifferentialDriveController

    ddc = DifferentialDriveController(wheel_base=0.6, dt=0.002)
    tl, tr = ddc.compute(1.0, 0.0, 0.0, 0.0)
    check("Positive torques when v_cmd>0 and wheels stationary",
          tl > 0 and tr > 0, f"tl={tl:.3f}  tr={tr:.3f}")


# ─── 4. WAP ──────────────────────────────────────────────────────

def test_wap():
    print("\n--- WaypointAttitudePlanner ---")
    from ras.planning.path_planner import WaypointAttitudePlanner

    wap = WaypointAttitudePlanner(capture_radius=0.3)
    check("No active waypoint initially", wap.active_waypoint is None)

    wap.add_waypoint(2.0, 0.0)
    wap.add_waypoint(4.0, 3.0)
    check("Active WP is first added", wap.active_waypoint == (2.0, 0.0))

    theta = wap.get_desired_attitude(0.0, 0.0)
    check("Desired attitude toward (2,0) is ~0 rad",
          abs(theta) < 0.01, f"got {theta:.4f}")

    dist = wap.get_distance_to_active(0.0, 0.0)
    check("Distance to (2,0) from origin is 2.0",
          abs(dist - 2.0) < 0.01, f"got {dist:.4f}")

    # Simulate reaching the first waypoint
    reached = wap.advance_if_reached(2.0, 0.1)
    check("First waypoint consumed when within radius", reached)
    check("Active WP advances to second", wap.active_waypoint == (4.0, 3.0))


# ─── 5. SPD ──────────────────────────────────────────────────────

def test_spd():
    print("\n--- SpeedProfileDesigner ---")
    from ras.planning.path_planner import SpeedProfileDesigner

    spd = SpeedProfileDesigner(v_max=1.2, a_x_lb=-1.0, a_x_ub=0.8)

    # Simulate many steps far from WP to let _v_ref ramp up
    v_far = 0.0
    for _ in range(500):
        v_far = spd.compute(dist_to_wp=10.0, heading_error=0.0, current_v=v_far)

    # Reset internal state before the near-WP test
    spd._v_ref = 0.0

    v_near = spd.compute(dist_to_wp=0.2, heading_error=0.0, current_v=0.0)
    check("Speed is higher far from WP than near",
          v_far > v_near, f"far={v_far:.3f}  near={v_near:.3f}")

    spd._v_ref = 0.0
    v_no_wp = spd.compute(dist_to_wp=float("inf"), heading_error=0.0,
                           current_v=0.5)
    check("Speed is 0 when no active waypoint", v_no_wp == 0.0)


# ─── 6. ARGA ─────────────────────────────────────────────────────

def test_arga():
    print("\n--- AngularRateGainAdapter ---")
    from ras.planning.path_planner import AngularRateGainAdapter

    arga = AngularRateGainAdapter(a_y_max=1.5)

    K_slow = arga.compute(v=0.01, theta_error=1.0)
    K_fast = arga.compute(v=2.0,  theta_error=1.0)
    check("Gain at low speed >= gain at high speed",
          K_slow >= K_fast, f"K_slow={K_slow:.3f}  K_fast={K_fast:.3f}")

    arga._prev_err = 0.5
    omega = arga.omega_desired(v=1.0, theta_error=0.5)
    check("Omega is finite and has correct sign",
          0 < omega < 10, f"got {omega:.4f}")


# ─── 7. mHRVO ────────────────────────────────────────────────────

def test_hrvo():
    print("\n--- ModifiedHRVO ---")
    from ras.planning.path_planner import ModifiedHRVO

    hrvo = ModifiedHRVO()
    v_s, o_s = hrvo.compute(pos=(0, 0), vel=(1, 0), heading=0.0,
                             v_desired=1.0, omega_desired=0.5)
    check("Pass-through when no obstacles",
          v_s == 1.0 and o_s == 0.5,
          f"v_safe={v_s:.3f}  omega_safe={o_s:.3f}")


# ─── 8. Full pipeline ────────────────────────────────────────────

def test_pipeline():
    print("\n--- PathPlanner (full pipeline) ---")
    from ras.planning.path_planner import PathPlanner

    pp = PathPlanner()
    # Use a nearby waypoint within the corridor so routing is trivial
    # Place wheelchair at a known nav-node location heading toward
    # a close destination along the same corridor segment.
    pp.wap.add_waypoint(5.0, 0.0)   # directly add to WAP (skip routing)

    state = {"x": 0, "y": 0, "theta": 0, "vx": 0, "vy": 0, "omega": 0,
             "v_left": 0, "v_right": 0}
    v, omega = pp.plan(state)
    check("Pipeline returns positive v toward (5,0)",
          v > 0, f"v={v:.3f}")
    check("Pipeline returns near-zero omega (already heading +x)",
          abs(omega) < 1.0, f"omega={omega:.4f}")


# ─── Runner ──────────────────────────────────────────────────────

if __name__ == "__main__":
    test_model()
    test_pid()
    test_dd_controller()
    test_wap()
    test_spd()
    test_arga()
    test_hrvo()
    test_pipeline()

    print(f"\n{'='*50}")
    print(f"  Results:  {PASS} passed  /  {FAIL} failed")
    print(f"{'='*50}")
    sys.exit(1 if FAIL else 0)
