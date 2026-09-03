"""
interactive_viewer.py
=====================
MuJoCo passive-viewer integration with:
  - Keyboard + mouse waypoint placement
  - Visual overlay rendering of waypoint markers, path, and obstacles
  - Real-time simulation loop wiring physics, planner, controller,
    and obstacle manager

Interaction:
  X (while hovering mouse)  ->  place a waypoint on the ground
  Backspace                 ->  clear all waypoints and stop
  R                         ->  reset wheelchair to origin
  O                         ->  respawn random obstacles
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Optional

import mujoco
import mujoco.viewer
import numpy as np

try:
    import glfw
    _HAS_GLFW = True
except ImportError:
    _HAS_GLFW = False

if TYPE_CHECKING:
    from ras.physics.wheelchair_model import WheelchairPhysics
    from ras.control.controller import DifferentialDriveController
    from ras.planning.path_planner import PathPlanner
    from ras.planning.obstacles import ObstacleManager


class InteractiveViewer:
    """
    Launches a MuJoCo passive viewer and runs the real-time control loop.

    Waypoints are placed by Ctrl+double-clicking on the ground plane.
    Red sphere markers are rendered at each waypoint.  The wheelchair
    autonomously navigates toward the waypoints using the PathPlanner
    and DifferentialDriveController.  Moving obstacles are rendered as
    coloured cylinders and fed into the mHRVO collision-avoidance module.
    """

    # -- Visual constants --
    MARKER_RADIUS = 0.20
    MARKER_COLOR  = (1, 0.15, 0.15, 0.9)
    GHOST_COLOR   = (0.3, 1.0, 0.3, 0.35)  # semi-transparent green ghost
    PATH_COLOR    = (1.0, 0.7, 0.0, 0.6)  # Yellow full path
    OBS_HEIGHT    = 0.6   # visual height of obstacle cylinders

    def __init__(self,
                 physics: "WheelchairPhysics",
                 controller: "DifferentialDriveController",
                 planner: "PathPlanner",
                 obstacle_mgr: Optional["ObstacleManager"] = None,
                 rl_agent=None):
        self.physics      = physics
        self.controller   = controller
        self.planner      = planner
        self.obstacle_mgr = obstacle_mgr
        self.rl_agent     = rl_agent   # Trained SB3 model (or None)

        # Viewer handle (set in run())
        self._viewer = None

        # Scene used for ray-casting (click_on_ground) — created once
        self._pick_scn = None
        self._pick_opt = None

        # Start-button state: wheelchair waits until user presses S
        self._started = False

        # Position placement mode: user must place and confirm start pos first
        self._position_confirmed = False
        self._dragging = False  # True while wheelchair follows cursor
        self._glfw_window = None  # cached from render thread

        # Trail: list of (x, y) positions the wheelchair has visited
        self._trail: list = []
        self._trail_interval = 0.3  # metres between trail points
        self._last_trail_x = None
        self._last_trail_y = None

    # ==================================================================
    #  Main loop
    # ==================================================================

    def run(self):
        """
        Launch the viewer and spin the simulation + control loop.
        Blocks until the viewer window is closed.
        """
        model = self.physics.model
        data  = self.physics.data

        with mujoco.viewer.launch_passive(
                model, data,
                key_callback=self._key_callback,
                show_left_ui=False,
                show_right_ui=False,
        ) as viewer:
            self._viewer = viewer

            # Allocate pick-scene once for ray-casting (mjv_select)
            self._pick_scn = mujoco.MjvScene(model, maxgeom=2000)
            self._pick_opt = mujoco.MjvOption()

            # Nice initial camera (sized for DXF hospital map ~90x184m)
            viewer.cam.azimuth   = 0
            viewer.cam.elevation = -60
            viewer.cam.distance  = 120.0
            viewer.cam.lookat[:] = [0, 0, 0.3]

            print("=" * 56)
            print("  Autonomous Wheelchair Simulation")
            print("  1. Press X to PICK UP wheelchair (follows cursor)")
            print("  2. Press X again to DROP it, then C to CONFIRM")
            print("  3. Press X to place waypoints")
            print("  4. Press S to START navigation")
            print("  Backspace = clear waypoints  |  R = full reset")
            print("  O = respawn obstacles")
            print("=" * 56)
            print("\n[MODE] PLACEMENT — press X to pick up wheelchair")

            dt = model.opt.timestep

            while viewer.is_running():
                step_start = time.time()

                # -- 0. Drag-and-drop: wheelchair follows cursor --
                if self._dragging:
                    pos = self._get_cursor_ground_pos_select()
                    if pos is not None:
                        self.physics.data.qpos[0] = pos[0]
                        self.physics.data.qpos[1] = pos[1]
                        self.physics.data.qvel[:] = 0
                        mujoco.mj_forward(self.physics.model,
                                          self.physics.data)

                # -- 1. Read state --
                state = self.physics.get_state()

                # -- 2. Update obstacles & feed to mHRVO --
                if self.obstacle_mgr is not None:
                    self.obstacle_mgr.step(dt)
                    self.planner.hrvo.update_obstacles(
                        self.obstacle_mgr.as_planner_obstacles())

                # -- 3. Plan (only after user presses Start) --
                if self._started:
                    v_cmd, omega_cmd = self.planner.plan(state)

                    # -- 3b. RL correction (retrained model) --
                    if self.rl_agent is not None:
                        rl_obs = self._build_rl_obs(state, v_cmd, omega_cmd)
                        rl_action, _ = self.rl_agent.predict(
                            rl_obs, deterministic=True)
                        delta_v = float(np.clip(rl_action[0], -0.3, 0.3))
                        delta_w = float(np.clip(rl_action[1], -1.0, 1.0))
                        v_cmd = float(np.clip(v_cmd + delta_v, 0.0, 1.5))
                        omega_cmd = float(np.clip(
                            omega_cmd + delta_w, -4.0, 4.0))
                else:
                    v_cmd, omega_cmd = 0.0, 0.0

                # -- 4. Control --
                torque_l, torque_r = self.controller.compute(
                    v_cmd, omega_cmd,
                    state["v_left"], state["v_right"])

                # -- 5. Actuate --
                self.physics.set_ctrl(torque_l, torque_r)

                # -- 6. Step physics --
                self.physics.step()

                # -- 7. Record trail --
                if self._started:
                    sx, sy = state["x"], state["y"]
                    if self._last_trail_x is None:
                        self._trail.append((sx, sy))
                        self._last_trail_x, self._last_trail_y = sx, sy
                    else:
                        dd = math.hypot(sx - self._last_trail_x,
                                        sy - self._last_trail_y)
                        if dd >= self._trail_interval:
                            self._trail.append((sx, sy))
                            self._last_trail_x, self._last_trail_y = sx, sy

                # -- 8. Render overlays (into user_scn) --
                viewer.user_scn.ngeom = 0  # clear previous frame overlays
                if self._dragging:
                    self._render_drag_marker(viewer, state)
                self._render_trail(viewer)
                self._render_waypoints(viewer)
                self._render_obstacles(viewer)
                if self._started:
                    self._render_future_path(viewer, state)

                # -- 8. Sync viewer --
                viewer.sync()

                # -- 9. Real-time pacing --
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        self._viewer = None
        print("\nSimulation ended.")

    # ==================================================================
    #  Keyboard callback
    # ==================================================================

    def _key_callback(self, keycode: int):
        """
        Handle keyboard events.

        Keys:
            88  ('X')        ->  place wheelchair (placement) / waypoint (confirmed)
            67  ('C')        ->  CONFIRM starting position
            83  ('S')        ->  START navigation
            259 (Backspace)  ->  clear waypoints & stop
            82  ('R')        ->  full reset (back to placement mode)
            79  ('O')        ->  respawn obstacles
        """
        # Cache window handle from render thread for cross-thread use
        if _HAS_GLFW and self._glfw_window is None:
            self._glfw_window = glfw.get_current_context()

        if keycode == 88:  # 'X'
            if self._started:
                print("[INFO] Already started — press Backspace to "
                      "clear and re-plan.")
                return
            if not self._position_confirmed:
                # PLACEMENT MODE: toggle drag
                if self._dragging:
                    # DROP the wheelchair
                    self._dragging = False
                    state = self.physics.get_state()
                    print(f"[DROPPED] Wheelchair at "
                          f"({state['x']:.2f}, {state['y']:.2f})  "
                          f"— press C to confirm or X to pick up again")
                else:
                    # PICK UP the wheelchair
                    self._dragging = True
                    print("[DRAG] Wheelchair follows cursor — "
                          "move mouse, press X to drop")
            else:
                # WAYPOINT MODE: place a navigation waypoint
                self._place_waypoint_at_cursor()

        elif keycode == 67:  # 'C' -- CONFIRM starting position
            if self._position_confirmed:
                print("[INFO] Position already confirmed.")
                return
            self._dragging = False  # drop if still dragging
            self._position_confirmed = True
            state = self.physics.get_state()
            print(f"[CONFIRMED] Starting position set at "
                  f"({state['x']:.2f}, {state['y']:.2f})")
            print("[MODE] WAYPOINT — hover mouse + press X to place "
                  "waypoints, then S to start")

        elif keycode == 83:  # 'S' -- START
            if not self._position_confirmed:
                print("[INFO] Confirm starting position first (C).")
                return
            if self._started:
                print("[INFO] Already running.")
                return
            n = len(self.planner.wap.waypoints)
            if n == 0:
                print("[INFO] Place waypoints first (X), then press S.")
                return
            self._started = True
            print(f"[START] Navigating through {n} waypoint(s)...")

        elif keycode == 259:  # Backspace
            self._started = False
            self.planner.wap.waypoints.clear()
            self.planner._user_waypoints.clear()
            self.planner._recovery.reset()
            self.controller.reset()
            self._trail.clear()
            self._last_trail_x = self._last_trail_y = None
            if self._position_confirmed:
                print("[INFO] Waypoints cleared. Place new ones with X.")
            else:
                print("[INFO] Cleared. Press X to place wheelchair.")

        elif keycode == 82:  # 'R' -- full reset back to placement mode
            self._started = False
            self._position_confirmed = False
            self._dragging = False
            self.physics.reset()
            self.planner.wap.waypoints.clear()
            self.planner._user_waypoints.clear()
            self.planner._recovery.reset()
            self.controller.reset()
            self._trail.clear()
            self._last_trail_x = self._last_trail_y = None
            if self.obstacle_mgr is not None:
                self.obstacle_mgr.spawn()
            print("[INFO] Full reset. Back to PLACEMENT mode.")
            print("[MODE] PLACEMENT — hover mouse + press X to set "
                  "start position")

        elif keycode == 79:  # 'O'
            if self.obstacle_mgr is not None:
                self.obstacle_mgr.spawn()
                print("[INFO] Obstacles respawned.")

    def _get_cursor_ground_pos_select(self):
        """
        Compute ground-plane position under the mouse cursor using
        MuJoCo's built-in mjv_select ray-cast.  This is accurate
        regardless of camera angle.
        """
        if not _HAS_GLFW or self._viewer is None:
            return None
        if self._pick_scn is None:
            return None

        try:
            window = self._glfw_window
            if window is None:
                window = glfw.get_current_context()
            if window is None:
                return None

            xpos, ypos = glfw.get_cursor_pos(window)
            win_w, win_h = glfw.get_window_size(window)
            if win_w <= 0 or win_h <= 0:
                return None

            # Normalised screen coordinates for mjv_select
            rel_x = xpos / win_w
            rel_y = 1.0 - ypos / win_h

            model = self.physics.model
            data  = self.physics.data

            # Update the pick scene with current camera
            mujoco.mjv_updateScene(
                model, data, self._pick_opt, None,
                self._viewer.cam, mujoco.mjtCatBit.mjCAT_ALL,
                self._pick_scn)

            viewport = self._viewer.viewport
            aspect = float(viewport.width) / max(viewport.height, 1)

            selpnt      = np.zeros(3, dtype=np.float64)
            geom_id_arr = np.zeros((1, 1), dtype=np.int32)
            flex_id_arr = np.zeros((1, 1), dtype=np.int32)
            skin_id_arr = np.zeros((1, 1), dtype=np.int32)

            body_id = mujoco.mjv_select(
                model, data, self._pick_opt,
                aspect, rel_x, rel_y,
                self._pick_scn,
                selpnt, geom_id_arr, flex_id_arr, skin_id_arr,
            )

            if body_id >= 0:
                return (float(selpnt[0]), float(selpnt[1]))
            return None

        except Exception as e:
            print(f"[WARN] Cursor ground calc failed: {e}")
            return None

    def _place_wheelchair_at_cursor(self):
        """
        Teleport the wheelchair to the mouse cursor position (placement mode).
        """
        pos = self._get_cursor_ground_pos()
        if pos is None:
            return
        wx, wy = pos
        # Teleport wheelchair: set qpos x,y directly
        self.physics.data.qpos[0] = wx
        self.physics.data.qpos[1] = wy
        # Reset velocities
        self.physics.data.qvel[:] = 0
        mujoco.mj_forward(self.physics.model, self.physics.data)
        print(f"[PLACEMENT] Wheelchair placed at ({wx:.2f}, {wy:.2f})  "
              f"— press C to confirm")

    def _place_waypoint_at_cursor(self):
        """
        Place a waypoint using MuJoCo's accurate mjv_select ray-cast.
        Safe because this is called from the key callback (render thread).
        """
        if not _HAS_GLFW or self._viewer is None:
            return
        try:
            window = glfw.get_current_context()
            if window is None:
                return
            xpos, ypos = glfw.get_cursor_pos(window)
            win_w, win_h = glfw.get_window_size(window)
            if win_w <= 0 or win_h <= 0:
                return
            rel_x = xpos / win_w
            rel_y = 1.0 - ypos / win_h
            self.click_on_ground(rel_x, rel_y)
        except Exception as e:
            print(f"[WARN] Waypoint placement failed: {e}")

    # ==================================================================
    #  Mouse -> ground-plane ray-cast
    # ==================================================================

    def click_on_ground(self, x_screen: float, y_screen: float):
        """
        Inject a waypoint from normalised screen coordinates.

        Called from the GLFW mouse callback installed by main.py.
        Performs a ray-cast from camera through (x_screen, y_screen)
        onto the ground plane.
        """
        if self._viewer is None:
            return

        model = self.physics.model
        data  = self.physics.data

        # Update the pick-scene so ray-cast uses the current camera
        mujoco.mjv_updateScene(
            model, data, self._pick_opt, None,
            self._viewer.cam, mujoco.mjtCatBit.mjCAT_ALL,
            self._pick_scn)

        viewport = self._viewer.viewport
        aspect = float(viewport.width) / max(viewport.height, 1)

        selpnt      = np.zeros(3, dtype=np.float64)
        geom_id_arr = np.zeros((1, 1), dtype=np.int32)
        flex_id_arr = np.zeros((1, 1), dtype=np.int32)
        skin_id_arr = np.zeros((1, 1), dtype=np.int32)

        body_id = mujoco.mjv_select(
            model, data, self._pick_opt,
            aspect, x_screen, y_screen,
            self._pick_scn,
            selpnt, geom_id_arr, flex_id_arr, skin_id_arr,
        )

        if body_id >= 0:
            wx, wy = float(selpnt[0]), float(selpnt[1])
            self.planner.add_waypoint(wx, wy)
            print(f"[WAYPOINT] Added ({wx:.2f}, {wy:.2f})  "
                  f"queue={len(self.planner.waypoints)}")

    # ==================================================================
    #  Overlay rendering - drag indicator
    # ==================================================================

    def _render_drag_marker(self, viewer, state):
        """Draw a pulsing green ring around the wheelchair during drag."""
        scn = viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        # Pulsing effect via time
        pulse = 0.6 + 0.4 * math.sin(time.time() * 6.0)
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.5, 0.02, 0],
            pos=[state["x"], state["y"], 0.02],
            mat=np.eye(3).flatten(),
            rgba=np.array([0.2, 1.0, 0.3, pulse * 0.6],
                          dtype=np.float32),
        )
        scn.ngeom += 1

    # ==================================================================
    #  Overlay rendering - ghost preview marker
    # ==================================================================

    def _render_trail(self, viewer):
        """Draw a persistent black trail showing where the wheelchair has been."""
        if len(self._trail) < 2:
            return
        scn = viewer.user_scn
        trail_color = (0.05, 0.05, 0.05, 0.85)  # near-black
        trail_radius = 0.05  # half of 0.1m thickness
        for i in range(len(self._trail) - 1):
            if scn.ngeom >= scn.maxgeom - 20:  # leave room for other overlays
                break
            x1, y1 = self._trail[i]
            x2, y2 = self._trail[i + 1]
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < 0.01:
                continue
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            angle = math.atan2(dy, dx)
            ca, sa = math.cos(angle), math.sin(angle)
            mat = np.array([
                ca, -sa, 0,
                sa,  ca, 0,
                0,   0,  1,
            ], dtype=np.float64)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[trail_radius, length / 2, 0],
                pos=[mx, my, 0.01],
                mat=mat,
                rgba=np.array(trail_color, dtype=np.float32),
            )
            scn.ngeom += 1

    # ==================================================================
    #  Overlay rendering - waypoint markers + path lines
    # ==================================================================

    def _render_waypoints(self, viewer):
        """Draw red sphere markers at user-placed waypoints and green
        path lines from the wheelchair to each one in sequence.

        Only the user's actual clicked positions are rendered — the
        routing intermediates (corridor nodes, door approach nodes)
        are invisible internal navigation detail.
        """
        # Remove user waypoints that the wheelchair has already reached
        state = self.physics.get_state()
        wx_pos, wy_pos = state["x"], state["y"]
        while self.planner._user_waypoints:
            ux, uy = self.planner._user_waypoints[0]
            if math.hypot(ux - wx_pos, uy - wy_pos) < 0.5:
                self.planner._user_waypoints.pop(0)
            else:
                break

        wps = self.planner.waypoints
        if not wps:
            return

        scn = viewer.user_scn

        # Sphere markers — only user-placed positions
        for i, (wx, wy) in enumerate(wps):
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.MARKER_RADIUS, 0, 0],
                pos=[wx, wy, self.MARKER_RADIUS + 0.01],
                mat=np.eye(3).flatten(),
                rgba=np.array(self.MARKER_COLOR, dtype=np.float32),
            )
            if i == 0:
                g.size[0] = self.MARKER_RADIUS * 1.4
            scn.ngeom += 1

    # ==================================================================
    #  Overlay rendering - obstacles
    # ==================================================================

    def _render_obstacles(self, viewer):
        """Draw coloured cylinders at each obstacle position into user_scn."""
        if self.obstacle_mgr is None:
            return

        scn = viewer.user_scn
        for obs in self.obstacle_mgr.obstacles:
            if scn.ngeom >= scn.maxgeom:
                break

            g = scn.geoms[scn.ngeom]
            half_h = self.OBS_HEIGHT / 2.0
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[obs.radius, half_h, 0],
                pos=[obs.x, obs.y, half_h],
                mat=np.eye(3).flatten(),
                rgba=np.array(obs.color, dtype=np.float32),
            )
            scn.ngeom += 1

            # Draw velocity arrow (direction indicator)
            speed = math.hypot(obs.vx, obs.vy)
            if speed > 0.05 and scn.ngeom < scn.maxgeom:
                arrow_len = min(speed * 0.8, obs.radius * 3)
                angle = math.atan2(obs.vy, obs.vx)
                end_x = obs.x + arrow_len * math.cos(angle)
                end_y = obs.y + arrow_len * math.sin(angle)
                self._draw_line(scn, obs.x, obs.y, end_x, end_y,
                                z=self.OBS_HEIGHT + 0.05,
                                color=(1.0, 1.0, 1.0, 0.7))

    def _render_future_path(self, viewer, state):
        """
        Draw a real-time predicted trajectory rollout that stretches out to
        the goal, using dynamic obstacle and wall avoidance.
        """
        nav_wps = list(self.planner.all_nav_waypoints)
        if not nav_wps:
            return

        scn = viewer.user_scn
        dt = 0.15
        steps = 150  # Roughly 22 seconds of unrolling
        
        x, y, theta = state["x"], state["y"], state["theta"]
        prev_x, prev_y = x, y
        wp_idx = 0

        color = self.PATH_COLOR  # Solid Yellow

        for i in range(steps):
            if scn.ngeom >= scn.maxgeom or wp_idx >= len(nav_wps):
                break
                
            # Pop waypoints locally in the unroll simulation
            target = nav_wps[wp_idx]
            dist_to_wp = math.hypot(target[0] - x, target[1] - y)
            if dist_to_wp < 0.6:
                wp_idx += 1
                if wp_idx >= len(nav_wps):
                    break
                target = nav_wps[wp_idx]
            
            # Simulated local steering towards target
            dx, dy = target[0] - x, target[1] - y
            theta_d = math.atan2(dy, dx)
            theta_err = (theta_d - theta + math.pi) % (2 * math.pi) - math.pi
            
            v_sim = 1.0 if abs(theta_err) < 0.5 else 0.5
            w_sim = np.clip(theta_err * 2.0, -3.0, 3.0)
                
            # Apply reactive wall/collision guard to the predicted state!
            v_safe, w_safe = self.planner._wall_guard.compute(x, y, theta, v_sim, w_sim)
            
            # Simple collision check with moving obstacles in the rollout
            if self.obstacle_mgr is not None:
                sim_time = i * dt
                for obs in self.obstacle_mgr.obstacles:
                    obs_x_t = obs.x + obs.vx * sim_time
                    obs_y_t = obs.y + obs.vy * sim_time
                    if math.hypot(x - obs_x_t, y - obs_y_t) < (self.planner.hrvo.robot_radius + obs.radius + 0.2):
                        cross = math.cos(theta)*(obs_y_t - y) - math.sin(theta)*(obs_x_t - x)
                        w_safe += 1.5 if cross <= 0 else -1.5
                        v_safe *= 0.5
            
            # Kinematic update
            theta += w_safe * dt
            next_x = x + v_safe * math.cos(theta) * dt
            next_y = y + v_safe * math.sin(theta) * dt
            
            self._draw_line(scn, prev_x, prev_y, next_x, next_y, z=0.04, color=color)
            
            x, y = next_x, next_y
            prev_x, prev_y = x, y



    # ==================================================================
    #  RL observation builder (mirrors rl_env.py observation space)
    # ==================================================================

    def _build_rl_obs(self, state: dict,
                      v_classical: float,
                      w_classical: float) -> np.ndarray:
        """
        Build the 19-dim observation vector expected by the trained RL agent.
        Must match the observation space defined in rl_env.py exactly.
        """
        from ras.planning.path_planner import WallProximityGuard

        x, y, theta = state["x"], state["y"], state["theta"]
        vx, vy = state["vx"], state["vy"]
        omega = state["omega"]
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # Waypoint in body frame
        wp = self.planner.wap.active_waypoint
        if wp is None:
            wp = (x, y)  # no target → zero vector
        dx_w = wp[0] - x
        dy_w = wp[1] - y
        dx_body =  cos_t * dx_w + sin_t * dy_w
        dy_body = -sin_t * dx_w + cos_t * dy_w
        dist_wp = math.hypot(dx_w, dy_w)

        theta_d = math.atan2(dy_w, dx_w)
        theta_err = (theta_d - theta + math.pi) % (2 * math.pi) - math.pi

        vx_body =  cos_t * vx + sin_t * vy
        vy_body = -sin_t * vx + cos_t * vy

        # Wall distances (simplified — 4 directions)
        wall_dists = [3.0, 3.0, 3.0, 3.0]
        dirs = [(cos_t, sin_t), (-cos_t, -sin_t),
                (-sin_t, cos_t), (sin_t, -cos_t)]
        for seg in WallProximityGuard.WALL_SEGMENTS:
            d, nx, ny = WallProximityGuard._point_to_segment_dist(
                x, y, *seg)
            if d > 3.0:
                continue
            wall_dx, wall_dy = -nx, -ny
            for di, (ddx, ddy) in enumerate(dirs):
                dot = wall_dx * ddx + wall_dy * ddy
                if dot > 0.3:
                    wall_dists[di] = min(wall_dists[di], d)

        # Nearest 2 obstacles in body frame
        obs_features = []
        obs_list = []
        if self.obstacle_mgr is not None:
            for obs in self.obstacle_mgr.obstacles:
                odx = obs.x - x
                ody = obs.y - y
                odist = math.hypot(odx, ody)
                odx_b =  cos_t * odx + sin_t * ody
                ody_b = -sin_t * odx + cos_t * ody
                obs_list.append((odist, odx_b, ody_b, obs.radius))
            obs_list.sort(key=lambda t: t[0])

        for i in range(2):
            if i < len(obs_list):
                _, odx_b, ody_b, r = obs_list[i]
                obs_features.extend([
                    float(np.clip(odx_b / 10.0, -1, 1)),
                    float(np.clip(ody_b / 10.0, -1, 1)),
                    float(np.clip(r / 0.5, 0, 1)),
                ])
            else:
                obs_features.extend([1.0, 0.0, 0.0])

        return np.array([
            np.clip(dx_body / 15.0, -1, 1),
            np.clip(dy_body / 15.0, -1, 1),
            np.clip(dist_wp / 25.0, 0, 1),
            theta_err / math.pi,
            np.clip(vx_body / 1.5, -1, 1),
            np.clip(vy_body / 1.5, -1, 1),
            np.clip(omega / 4.0, -1, 1),
            np.clip(v_classical / 1.5, -1, 1),
            np.clip(w_classical / 4.0, -1, 1),
            np.clip(wall_dists[0] / 3.0, 0, 1),
            np.clip(wall_dists[1] / 3.0, 0, 1),
            np.clip(wall_dists[2] / 3.0, 0, 1),
            np.clip(wall_dists[3] / 3.0, 0, 1),
            *obs_features,
        ], dtype=np.float32)

    # ==================================================================
    #  Line drawing helper
    # ==================================================================

    def _draw_line(self, scn, x1, y1, x2, y2, z=0.02, color=None):
        """Draw a thin capsule between two ground points."""
        if color is None:
            color = self.PATH_COLOR

        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-4:
            return

        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        angle = math.atan2(dy, dx)

        ca, sa = math.cos(angle), math.sin(angle)
        mat = np.array([
            ca, -sa, 0,
            sa,  ca, 0,
            0,   0,  1,
        ], dtype=np.float64)

        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.015, length / 2, 0],
            pos=[mx, my, z],
            mat=mat,
            rgba=np.array(color, dtype=np.float32),
        )
        scn.ngeom += 1
