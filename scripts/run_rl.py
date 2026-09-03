"""
run_rl.py
=========
Launch the interactive MuJoCo viewer with a trained RL agent providing
corrections on top of the classical pipeline.

If no trained model is found, falls back to the pure classical pipeline
(identical to main.py).

Usage:
    python run_rl.py                          # Use best model
    python run_rl.py --model rl_models/wheelchair_ppo.zip
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ras.physics.wheelchair_model import WheelchairPhysics
from ras.control.controller import DifferentialDriveController
from ras.planning.path_planner import PathPlanner
from ras.planning.obstacles import ObstacleManager
from ras.ui.interactive_viewer import InteractiveViewer
from ras.ui.tuning_panel import TuningPanel


def main():
    parser = argparse.ArgumentParser(
        description="Run wheelchair simulation with trained RL agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model .zip file")
    parser.add_argument("--model-dir", type=str, default="rl_models",
                        help="Directory to search for models")
    args = parser.parse_args()

    # ── Find model ──
    model = None
    model_path = args.model

    if model_path is None:
        # Auto-detect model
        candidates = [
            os.path.join(args.model_dir, "best_model.zip"),
            os.path.join(args.model_dir, "wheelchair_ppo.zip"),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if model_path and os.path.exists(model_path):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            print(f"[RL] Loaded trained model from: {model_path}")
        except Exception as e:
            print(f"[RL] Could not load model: {e}")
            print("[RL] Falling back to classical pipeline.")
            model = None
    else:
        print("[RL] No trained model found. Using classical pipeline only.")
        print(f"     (Run 'python train_rl.py' first to train a model)")

    # ── Build simulation ──
    print("Initializing wheelchair simulation...")

    physics = WheelchairPhysics()

    controller = DifferentialDriveController(
        wheel_base=physics.wheel_base,
        dt=physics.dt,
        kp=15.0, ki=2.0, kd=2.5,
        torque_limit=10.0,  # must match XML ctrlrange="-10 10"
    )

    planner = PathPlanner(
        v_max=1.2, a_x_lb=-1.0, a_x_ub=0.1,
        a_y_max=1.5, tau_c=0.1, capture_radius=2.0,
    )

    obs_mgr = ObstacleManager(
        num_obstacles=3,
        radius_range=(0.12, 0.30),
        speed_range=(0.2, 0.5),
        spawn_range=(2.0, 6.0),
        boundary=11.0,
        bounce=True,
    )
    obs_mgr.spawn()
    print(obs_mgr)

    viewer = InteractiveViewer(
        physics, controller, planner,
        obstacle_mgr=obs_mgr,
        rl_agent=model,
    )

    # -- Launch tuning panel --
    panel = TuningPanel(controller, planner)
    panel.start()

    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        panel.stop()
        print("Goodbye.")


if __name__ == "__main__":
    main()
