"""
main.py
=======
Entry point for the autonomous wheelchair simulation.

Wires together:
  - WheelchairPhysics  (MuJoCo model + data)
  - MPCController  (MPC actuation)
  - PathPlanner  (WAP -> SPD -> ARGA -> mHRVO)
  - ObstacleManager  (random moving obstacles)
  - InteractiveViewer  (GUI, mouse events, render loop)

Usage:
    python main.py

Interaction (inside the viewer window):
    1. Hover mouse + press X             ->  place waypoints
    2. Press S                            ->  START navigation
    Backspace                             ->  clear waypoints & stop
    R                                     ->  reset simulation
    O                                     ->  respawn obstacles
    Close window or Esc                   ->  exit

Notes:
    - Place ALL waypoints before pressing S
    - Wheelchair can move forward or backward (auto-selects)
    - Doorways are 0.9 m wide (1.5× wheelchair width)
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ras.physics.wheelchair_model import WheelchairPhysics
from ras.control.controller import MPCController
from ras.planning.path_planner import PathPlanner
from ras.planning.obstacles import ObstacleManager
from ras.ui.interactive_viewer import InteractiveViewer
from ras.ui.tuning_panel import TuningPanel


# ======================================================================
#  Obstacle configuration  (EDIT THESE TO TASTE)
# ======================================================================

OBSTACLE_CONFIG = dict(
    num_obstacles = 3,                  # fewer obstacles in indoor space
    radius_range  = (0.12, 0.30),       # (min, max) radius in metres
    speed_range   = (0.2, 0.5),         # (min, max) velocity magnitude m/s
    spawn_range   = (10.0, 60.0),       # spawn distance from origin
    boundary      = 100.0,              # arena half-size (~86x184m map)
    bounce        = True,               # True = bounce, False = wrap
    seed          = None,               # set an int for reproducible layout
)


# ======================================================================
#  Main
# ======================================================================

def main():
    """Instantiate all modules and launch the simulation."""

    print("Initializing wheelchair simulation...")

    # -- 1. Physics --
    physics = WheelchairPhysics()

    # -- 2. Controller (MPC — Model Predictive Control) --
    controller = MPCController(
        wheel_base=physics.wheel_base,
        dt=physics.dt,
        torque_limit=10.0,  # must match XML ctrlrange="-10 10"
        horizon=10,         # 10-step look-ahead
        dt_mpc=0.02,        # 50 Hz solve rate
        Q_track=120.0,      # speed-tracking weight
        R_effort=0.05,      # control-effort weight
        Q_jerk=0.8,         # smoothness weight
    )

    # -- 3. Path Planner --
    planner = PathPlanner(
        v_max=1.2,            # m/s  -- comfortable walking pace
        a_x_lb=-1.0,          # m/s2 -- braking
        a_x_ub=0.1,           # m/s2 -- acceleration (≤0.1 safety limit)
        a_y_max=1.5,          # m/s2 -- lateral comfort limit
        tau_c=0.1,            # s    -- system delay
        capture_radius=2.0,
    )

    # -- 4. Obstacle Manager --
    obs_mgr = ObstacleManager(**OBSTACLE_CONFIG)
    obs_mgr.spawn()
    print(obs_mgr)

    # You can also add custom obstacles with exact parameters:
    # obs_mgr.add_custom(x=5, y=2, vx=-0.5, vy=0.3, radius=0.4)

    # -- 5. Interactive Viewer --
    viewer = InteractiveViewer(physics, controller, planner,
                                obstacle_mgr=obs_mgr)

    # -- Launch tuning panel --
    panel = TuningPanel(controller, planner)
    panel.start()

    # -- 6. Run (blocking) --
    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        panel.stop()
        print("Goodbye.")


if __name__ == "__main__":
    main()

