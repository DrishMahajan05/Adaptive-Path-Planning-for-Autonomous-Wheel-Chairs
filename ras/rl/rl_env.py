"""
rl_env.py
=========
Gymnasium-compatible Reinforcement Learning environment for the autonomous
wheelchair simulation.

The RL agent acts as a **refinement layer** on top of the classical 4-stage
pipeline (WAP → SPD → ARGA → mHRVO).  It observes the wheelchair state plus
the classical pipeline's recommended commands, and outputs small corrections
(delta_v, delta_omega) to produce smoother, more collision-aware trajectories.

Usage:
    from rl_env import WheelchairRLEnv
    env = WheelchairRLEnv()
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ras.physics.wheelchair_model import WheelchairPhysics
from ras.control.controller import DifferentialDriveController
from ras.planning.path_planner import PathPlanner, WallProximityGuard
from ras.planning.obstacles import ObstacleManager


# ──────────────────────────────────────────────────────────────────────────
#  Pre-defined training scenarios (start → goal)
# ──────────────────────────────────────────────────────────────────────────

_TRAINING_GOALS = [
    # (start_x, start_y, goal_x, goal_y)
    # Derived from nav-graph nodes of the new DXF hospital corridor map.
    # Map extents: X ≈ [-36, 54], Y ≈ [-88, 96]

    # ── Long corridor traversals (end-to-end) ──
    (-20.1, -85.9,  39.4,  93.4),   # bottom → top-left
    (-20.1, -85.9,  50.3,  79.3),   # bottom → top-right
    ( 39.4,  93.4, -20.1, -85.9),   # top-left → bottom
    ( 50.3,  79.3, -20.1, -85.9),   # top-right → bottom
    (-33.6, -23.4,  39.4,  93.4),   # mid-left → top-left
    ( 50.3,  79.3, -33.6, -23.4),   # top-right → mid-left

    # ── Medium corridor routes ──
    (-20.1, -85.9,  -7.0,  38.1),   # bottom → mid-upper
    ( -7.0,  38.1,  50.3,  79.3),   # mid-upper → top-right
    (-24.0,  10.9,  13.2,  63.0),   # mid-left → mid-right-upper
    ( 13.2,  63.0, -24.0,  10.9),   # mid-right-upper → mid-left
    (-18.6, -69.9,  -2.5,  44.7),   # lower → mid-corridor
    (  2.7,  50.8, -18.6, -69.9),   # mid → lower

    # ── Short hops (neighbouring nav nodes) ──
    (-20.1, -85.9, -18.5, -69.9),   # bottom → near-bottom
    (-18.5, -69.9, -16.1, -54.0),   # step up
    (-16.1, -54.0, -11.0, -30.9),   # mid-low section
    (-11.0, -30.9, -24.0,  10.9),   # cross through junction
    (-24.0,  10.9,  -7.0,  38.1),   # junction → upper corridor
    (  8.0,  56.9,  23.7,  75.1),   # upper corridor hop
    ( 23.7,  75.1,  39.4,  93.4),   # upper → top

    # ── Junction / corridor-change routes ──
    (-33.6, -23.4, -11.0, -30.9),   # left wing → right wing (low)
    (-11.0, -30.9, -33.6, -23.4),   # right wing → left wing (low)
    (-22.8,   5.6,   5.9,   1.1),   # mid-left junction → mid-right
    (  5.9,   1.1, -22.8,   5.6),   # mid-right → mid-left junction
    (-26.3, -26.0, -18.6, -28.4),   # short junction traverse
    (  1.7,  27.0,  14.0,  17.9),   # upper junction region

    # ── U-turn / heading-recovery scenarios ──
    ( 39.4,  93.4,  34.2,  87.3),   # nearly same spot, forces u-turn
    (-20.1, -85.9, -19.3, -77.9),   # short reverse
    ( 50.3,  79.3,  44.0,  89.9),   # top area reversal
    (-15.5,  24.5, -19.8,  17.7),   # mid corridor reverse

    # ── Target corridor: (32, 74) → (32, 53) and variations ──
    ( 32.37, 73.47,  32.45, 52.92),  # exact user path
    ( 32.0,  74.0,   32.5,  53.0),   # variation 1
    ( 31.5,  73.0,   33.0,  52.5),   # variation 2
    ( 33.0,  75.0,   31.5,  51.0),   # variation 3
    ( 32.45, 52.92,  32.37, 73.47),  # reverse direction
    ( 32.0,  53.0,   32.0,  74.0),   # reverse variation
    ( 30.0,  72.0,   35.0,  55.0),   # wider start
    ( 34.0,  70.0,   30.0,  50.0),   # diagonal variation
    ( 32.37, 73.47,  32.45, 52.92),  # duplicate for more weight
    ( 32.37, 73.47,  32.45, 52.92),  # duplicate for more weight
]


class WheelchairRLEnv(gym.Env):
    """
    Gymnasium environment for RL-based wheelchair navigation refinement.

    Observation (19-dim):
        0-1:    dx, dy to active waypoint (body frame, normalized)
        2:      distance to waypoint (normalized 0..1)
        3:      heading error to waypoint ([-1, 1], normalized from [-pi, pi])
        4-5:    body-frame velocity (vx_body, vy_body), normalized
        6:      yaw rate omega, normalized
        7-8:    classical pipeline commands (v_classical, w_classical), normalized
        9-12:   wall distances in 4 directions (front/back/left/right), clamped
        13-18:  2 nearest obstacles (dx, dy, radius) in body frame

    Action (2-dim):
        0:      delta_v      correction to classical speed     [-0.3, +0.3]
        1:      delta_omega  correction to classical yaw rate  [-1.0, +1.0]
    """

    metadata = {"render_modes": []}

    # Episode limits
    MAX_STEPS        = 2000   # increased for the large ~90x184m hospital map
    PHYSICS_SUBSTEPS = 25     # MuJoCo steps per RL step (25 * 0.002 = 0.05s)

    # Reward weights
    R_PROGRESS        =  2.0
    R_WAYPOINT_BONUS  = 50.0
    R_GOAL_BONUS      = 100.0
    R_WALL_COLLISION  = -100.0   # Massive penalty for walls/doorways
    R_OBS_COLLISION   = -15.0
    R_SMOOTHNESS      = -0.5
    R_TIME_PENALTY    = -0.1

    # Thresholds
    WALL_COLLISION_DIST = 0.45   # 0.40 is wheel radius, so 0.45 avoids scrapes
    OBS_COLLISION_DIST  = 0.15   # margin on top of radii

    def __init__(self,
                 num_obstacles: int = 3,
                 max_steps: int = 2000,
                 seed: Optional[int] = None):
        super().__init__()

        self.MAX_STEPS = max_steps
        self._rng = random.Random(seed)

        # ── Core simulation modules ──
        self.physics = WheelchairPhysics()
        self.controller = DifferentialDriveController(
            wheel_base=self.physics.wheel_base,
            dt=self.physics.dt,
            kp=15.0, ki=2.0, kd=2.5,
            torque_limit=10.0,
        )
        self.planner = PathPlanner(
            v_max=1.2, a_x_lb=-1.0, a_x_ub=0.1,
            a_y_max=1.5, tau_c=0.1, capture_radius=0.60,
        )
        self.obstacle_mgr = ObstacleManager(
            num_obstacles=num_obstacles,
            radius_range=(0.12, 0.30),
            speed_range=(0.2, 0.5),
            spawn_range=(5.0, 40.0),    # larger range for the 90x184m map
            boundary=100.0,             # covers the full hospital map
            bounce=True,
            seed=seed,
        )
        self._wall_guard = WallProximityGuard()

        # ── Gym spaces ──
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(19,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-0.3, -1.0], dtype=np.float32),
            high=np.array([0.3, 1.0], dtype=np.float32),
            dtype=np.float32)

        # ── Episode state ──
        self._step_count  = 0
        self._prev_dist   = 0.0
        self._prev_v_cmd  = 0.0
        self._prev_w_cmd  = 0.0
        self._goal_xy: Tuple[float, float] = (0.0, 0.0)
        self._total_waypoints = 0

    # ==================================================================
    #  Reset
    # ==================================================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        # Reset physics
        self.physics.reset()
        self.controller.reset()

        # Clear planner state
        self.planner.wap.waypoints.clear()
        self.planner._user_waypoints.clear()
        self.planner._recovery.reset()

        # Pick a random training scenario
        scenario = self._rng.choice(_TRAINING_GOALS)
        sx, sy, gx, gy = scenario

        # Add small random jitter to start position
        sx += self._rng.uniform(-0.5, 0.5)
        sy += self._rng.uniform(-0.3, 0.3)

        # Set wheelchair position via qpos
        self.physics.data.qpos[0] = sx
        self.physics.data.qpos[1] = sy
        # Random initial heading
        yaw = self._rng.uniform(-math.pi, math.pi)
        self.physics.data.qpos[3] = math.cos(yaw / 2)
        self.physics.data.qpos[6] = math.sin(yaw / 2)
        import mujoco
        mujoco.mj_forward(self.physics.model, self.physics.data)

        # Set goal
        self._goal_xy = (gx, gy)
        self.planner.add_waypoint(gx, gy)
        self._total_waypoints = len(self.planner.wap.waypoints)

        # Spawn obstacles
        self.obstacle_mgr.spawn()
        self.planner.hrvo.update_obstacles(
            self.obstacle_mgr.as_planner_obstacles())

        # Episode bookkeeping
        self._step_count = 0
        state = self.physics.get_state()
        self._prev_dist = math.hypot(
            gx - state["x"], gy - state["y"])
        self._prev_v_cmd = 0.0
        self._prev_w_cmd = 0.0

        obs = self._get_obs(state, 0.0, 0.0)
        return obs, {}

    # ==================================================================
    #  Step
    # ==================================================================

    def step(self, action: np.ndarray):
        self._step_count += 1
        delta_v = float(np.clip(action[0], -0.3, 0.3))
        delta_w = float(np.clip(action[1], -1.0, 1.0))

        # ── Get classical pipeline commands ──
        state = self.physics.get_state()
        v_classical, w_classical = self.planner.plan_classical(state)

        # ── Apply RL correction ──
        v_cmd = np.clip(v_classical + delta_v, 0.0, 1.5)
        w_cmd = np.clip(w_classical + delta_w, -4.0, 4.0)

        # ── Apply wall guard (safety net) ──
        v_cmd, w_cmd = self._wall_guard.compute(
            state["x"], state["y"], state["theta"], v_cmd, w_cmd)

        # ── Step physics multiple substeps ──
        dt = self.physics.dt
        for _ in range(self.PHYSICS_SUBSTEPS):
            self.obstacle_mgr.step(dt)

            torque_l, torque_r = self.controller.compute(
                v_cmd, w_cmd,
                state["v_left"], state["v_right"])
            self.physics.set_ctrl(torque_l, torque_r)
            self.physics.step()
            state = self.physics.get_state()

        # Update obstacle info for planner
        self.planner.hrvo.update_obstacles(
            self.obstacle_mgr.as_planner_obstacles())

        # ── Compute reward ──
        x, y = state["x"], state["y"]
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        # Progress toward goal
        dist_to_goal = math.hypot(
            self._goal_xy[0] - x, self._goal_xy[1] - y)
        progress = self._prev_dist - dist_to_goal
        reward += self.R_PROGRESS * progress
        self._prev_dist = dist_to_goal

        # Waypoint capture bonus
        wp_before = len(self.planner.wap.waypoints)
        self.planner.wap.advance_if_reached(x, y)
        wp_after = len(self.planner.wap.waypoints)
        if wp_after < wp_before:
            reward += self.R_WAYPOINT_BONUS

        # Goal reached
        if dist_to_goal < 0.8:
            reward += self.R_GOAL_BONUS
            terminated = True
            info["success"] = True

        # Wall collision check
        min_wall_dist = self._min_wall_distance(x, y)
        if min_wall_dist < self.WALL_COLLISION_DIST:
            reward += self.R_WALL_COLLISION
            terminated = True
            info["wall_collision"] = True

        # Obstacle collision check
        for obs in self.obstacle_mgr.obstacles:
            d = math.hypot(obs.x - x, obs.y - y)
            if d < (obs.radius + 0.40 + self.OBS_COLLISION_DIST):
                reward += self.R_OBS_COLLISION
                terminated = True
                info["obs_collision"] = True
                break

        # Smoothness penalty
        dv = abs(v_cmd - self._prev_v_cmd)
        dw = abs(w_cmd - self._prev_w_cmd)
        reward += self.R_SMOOTHNESS * (dv + 0.3 * dw)
        self._prev_v_cmd = v_cmd
        self._prev_w_cmd = w_cmd

        # Time penalty
        reward += self.R_TIME_PENALTY

        # Truncation (timeout)
        if self._step_count >= self.MAX_STEPS:
            truncated = True

        obs = self._get_obs(state, v_classical, w_classical)
        return obs, reward, terminated, truncated, info

    # ==================================================================
    #  Observation building
    # ==================================================================

    def _get_obs(self, state: dict,
                 v_classical: float, w_classical: float) -> np.ndarray:
        x, y, theta = state["x"], state["y"], state["theta"]
        vx, vy = state["vx"], state["vy"]
        omega = state["omega"]

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Waypoint in body frame
        wp = self.planner.wap.active_waypoint
        if wp is None:
            wp = self._goal_xy
        dx_w = wp[0] - x
        dy_w = wp[1] - y
        dx_body =  cos_t * dx_w + sin_t * dy_w
        dy_body = -sin_t * dx_w + cos_t * dy_w
        dist_wp = math.hypot(dx_w, dy_w)

        # Heading error
        theta_d = math.atan2(dy_w, dx_w)
        theta_err = (theta_d - theta + math.pi) % (2 * math.pi) - math.pi

        # Body-frame velocity
        vx_body =  cos_t * vx + sin_t * vy
        vy_body = -sin_t * vx + cos_t * vy

        # Wall distances (4 directions)
        wall_front, wall_back, wall_left, wall_right = \
            self._directional_wall_dists(x, y, theta)

        # Nearest 2 obstacles in body frame
        obs_features = self._nearest_obstacles_body(x, y, theta, n=2)

        obs = np.array([
            np.clip(dx_body / 15.0, -1, 1),
            np.clip(dy_body / 15.0, -1, 1),
            np.clip(dist_wp / 25.0, 0, 1),
            theta_err / math.pi,
            np.clip(vx_body / 1.5, -1, 1),
            np.clip(vy_body / 1.5, -1, 1),
            np.clip(omega / 4.0, -1, 1),
            np.clip(v_classical / 1.5, -1, 1),
            np.clip(w_classical / 4.0, -1, 1),
            np.clip(wall_front / 3.0, 0, 1),
            np.clip(wall_back / 3.0, 0, 1),
            np.clip(wall_left / 3.0, 0, 1),
            np.clip(wall_right / 3.0, 0, 1),
            *obs_features,
        ], dtype=np.float32)

        return obs

    # ==================================================================
    #  Helpers
    # ==================================================================

    def _min_wall_distance(self, x: float, y: float) -> float:
        """Return minimum distance to any wall segment."""
        min_d = float("inf")
        for seg in WallProximityGuard.WALL_SEGMENTS:
            d, _, _ = WallProximityGuard._point_to_segment_dist(
                x, y, *seg)
            min_d = min(min_d, d)
        return min_d

    def _directional_wall_dists(self, x: float, y: float,
                                theta: float) -> Tuple[float, float, float, float]:
        """Compute approximate wall distance in 4 body-frame directions."""
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Direction vectors in world frame
        dirs = {
            "front": (cos_t, sin_t),
            "back":  (-cos_t, -sin_t),
            "left":  (-sin_t, cos_t),
            "right": (sin_t, -cos_t),
        }

        results = {}
        for name, (dx, dy) in dirs.items():
            min_d = 10.0
            for seg in WallProximityGuard.WALL_SEGMENTS:
                d, nx, ny = WallProximityGuard._point_to_segment_dist(
                    x, y, *seg)
                if d > 3.0:
                    continue
                # Check if wall is in this direction
                wall_dir_x, wall_dir_y = -nx, -ny
                dot = wall_dir_x * dx + wall_dir_y * dy
                if dot > 0.3:
                    min_d = min(min_d, d)
            results[name] = min_d

        return results["front"], results["back"], results["left"], results["right"]

    def _nearest_obstacles_body(self, x: float, y: float,
                                 theta: float, n: int = 2) -> list:
        """Return features for n nearest obstacles in body frame."""
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        obs_list = []
        for obs in self.obstacle_mgr.obstacles:
            dx_w = obs.x - x
            dy_w = obs.y - y
            dist = math.hypot(dx_w, dy_w)
            dx_b =  cos_t * dx_w + sin_t * dy_w
            dy_b = -sin_t * dx_w + cos_t * dy_w
            obs_list.append((dist, dx_b, dy_b, obs.radius))

        obs_list.sort(key=lambda t: t[0])

        features = []
        for i in range(n):
            if i < len(obs_list):
                _, dx_b, dy_b, r = obs_list[i]
                features.extend([
                    np.clip(dx_b / 10.0, -1, 1),
                    np.clip(dy_b / 10.0, -1, 1),
                    np.clip(r / 0.5, 0, 1),
                ])
            else:
                features.extend([1.0, 0.0, 0.0])  # far away, no obstacle

        return features
