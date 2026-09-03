"""
path_planner.py
===============
4-stage classical path-planning pipeline inspired by Jung et al. (2020):
"Path Planning Algorithm for an Autonomous Electric Wheelchair in Hospitals".

Pipeline stages (executed every physics step):
  1. WAP  — Waypoint & Attitude Planning
  2. SPD  — Speed Profile Design
  3. ARGA — Angular Rate Gain Adaptation
  4. mHRVO — Modified Hybrid Reciprocal Velocity Obstacle (collision avoidance)

Additional module:
  HospitalGraph — Topological D* Lite router that plans door-to-door paths
                  through the hospital corridor layout, preventing wall
                  collisions.  Supports incremental replanning when edge
                  costs change (e.g., obstacles block a corridor).

Each stage is its own class; the PathPlanner orchestrator composes them.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  0. Hospital Topological Graph  (DXF corridor-centreline routing)
# ═══════════════════════════════════════════════════════════════════════════

class HospitalGraph:
    """
    Corridor-centreline navigation graph generated from the DXF floor plan.

    Nodes are placed along the inset centreline of the single wall boundary.
    A* search finds the shortest path between any two points through the
    corridor network.
    """

    _INF = float("inf")

    def __init__(self):
        from ras.map.dxf_parser import load_wall_polyline, generate_nav_graph
        verts, _, _ = load_wall_polyline()
        self.NODES, edges = generate_nav_graph(verts)
        # Build adjacency
        self._adj: Dict[str, List[Tuple[str, float]]] = {n: [] for n in self.NODES}
        for a, b in edges:
            d = self._node_dist(a, b)
            self._adj[a].append((b, d))
            self._adj[b].append((a, d))

    def _node_dist(self, a: str, b: str) -> float:
        xa, ya = self.NODES[a]
        xb, yb = self.NODES[b]
        return math.hypot(xb - xa, yb - ya)

    def _nearest_node(self, x: float, y: float) -> str:
        best_id: Optional[str] = None
        best_d = float("inf")
        for nid, (nx, ny) in self.NODES.items():
            d = math.hypot(nx - x, ny - y)
            if d < best_d:
                best_d = d
                best_id = nid
        assert best_id is not None
        return best_id

    def astar(self, src: str, dst: str) -> List[str]:
        if src == dst:
            return [src]
        open_set: list = []
        heapq.heappush(open_set, (0.0, src))
        came_from: Dict[str, str] = {}
        g_score: Dict[str, float] = {n: float("inf") for n in self.NODES}
        g_score[src] = 0.0
        closed: set = set()
        while open_set:
            _, current = heapq.heappop(open_set)
            if current in closed:
                continue
            closed.add(current)
            if current == dst:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            for neighbor, cost in self._adj.get(current, []):
                if neighbor in closed:
                    continue
                tentative_g = g_score[current] + cost
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._node_dist(neighbor, dst)
                    heapq.heappush(open_set, (f, neighbor))
        return [src]  # no path found — don't fabricate a wall-crossing route

    def update_edge_cost(self, a: str, b: str, new_cost: float) -> None:
        for i, (nb, _) in enumerate(self._adj.get(a, [])):
            if nb == b:
                self._adj[a][i] = (b, new_cost)
                break
        for i, (nb, _) in enumerate(self._adj.get(b, [])):
            if nb == a:
                self._adj[b][i] = (a, new_cost)
                break

    @staticmethod
    def _simplify_path(waypoints: List[Tuple[float, float]],
                       tolerance: float = 2.0
                       ) -> List[Tuple[float, float]]:
        """
        Remove collinear intermediate waypoints using perpendicular
        distance.  If a point deviates less than `tolerance` metres
        from the line between its neighbours, it is removed.

        This collapses long straight corridor segments (many nav-graph
        nodes in a line) into a single segment, reducing unnecessary
        waypoints before subdivision re-inserts evenly-spaced ones.
        """
        if len(waypoints) <= 2:
            return waypoints
        result: List[Tuple[float, float]] = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            ax, ay = result[-1]
            bx, by = waypoints[i]
            cx, cy = waypoints[i + 1]
            # Perpendicular distance of B from line A→C
            acx, acy = cx - ax, cy - ay
            ac_len = math.hypot(acx, acy)
            if ac_len < 1e-6:
                continue
            # Cross product gives signed area of parallelogram
            cross = abs(acx * (by - ay) - acy * (bx - ax))
            perp_dist = cross / ac_len
            if perp_dist >= tolerance:
                result.append((bx, by))  # keep — it's a real turn
        result.append(waypoints[-1])
        return result

    @staticmethod
    def _subdivide_path(waypoints: List[Tuple[float, float]],
                        max_spacing: float = 10.0
                        ) -> List[Tuple[float, float]]:
        """
        Insert intermediate waypoints along any segment longer than
        `max_spacing` metres.

        For a segment of length L > max_spacing, the number of
        sub-segments is ceil(L / max_spacing), and intermediate points
        are placed at equal intervals.

        Examples (max_spacing=10):
          20 m apart  →  1 midpoint  (2 sub-segments of 10 m)
          30 m apart  →  2 intermediates  (3 sub-segments of 10 m)
          25 m apart  →  2 intermediates  (3 sub-segments of ~8.3 m)
        """
        if len(waypoints) < 2:
            return waypoints
        result: List[Tuple[float, float]] = [waypoints[0]]
        for i in range(1, len(waypoints)):
            x0, y0 = waypoints[i - 1]
            x1, y1 = waypoints[i]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len > max_spacing:
                n_subs = math.ceil(seg_len / max_spacing)
                for k in range(1, n_subs):
                    t = k / n_subs
                    result.append((x0 + t * (x1 - x0),
                                   y0 + t * (y1 - y0)))
            result.append((x1, y1))
        return result

    def route(self, from_xy: Tuple[float, float],
              to_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Compute a route from from_xy to to_xy through corridor nodes."""
        src_node = self._nearest_node(*from_xy)
        dst_node = self._nearest_node(*to_xy)
        if src_node == dst_node:
            subdivided = self._subdivide_path([from_xy, to_xy])
            return subdivided[1:]  # skip from_xy
        node_path = self.astar(src_node, dst_node)
        # Build full path: from_xy → intermediate nodes → to_xy
        full_path: List[Tuple[float, float]] = [from_xy]
        for nid in node_path[1:]:
            full_path.append(self.NODES[nid])
        full_path.append(to_xy)
        # 1. Simplify: remove collinear intermediates (straight corridors)
        simplified = self._simplify_path(full_path)
        # 2. Subdivide: re-insert evenly-spaced waypoints on long segments
        subdivided = self._subdivide_path(simplified)
        return subdivided[1:]  # skip from_xy



# ═══════════════════════════════════════════════════════════════════════════
#  1. Waypoint & Attitude Planning (WAP)
# ═══════════════════════════════════════════════════════════════════════════

class WaypointAttitudePlanner:
    """
    Manages the waypoint queue and computes the desired heading (attitude)
    towards the currently active waypoint.

    Waypoints are added dynamically via user mouse-clicks during simulation.
    The planner advances to the next waypoint once the wheelchair is within
    a configurable capture radius.
    """

    def __init__(self, capture_radius: float = 0.30):
        """
        Parameters
        ----------
        capture_radius : float
            Distance (m) at which a waypoint is considered "reached".
        """
        self.capture_radius = capture_radius
        self.waypoints: deque[Tuple[float, float]] = deque()
        self._visited: List[Tuple[float, float]] = []

    # ── Public API ─────────────────────────────────────────────────

    def add_waypoint(self, x: float, y: float):
        """Append a new waypoint to the end of the queue."""
        self.waypoints.append((x, y))

    @property
    def active_waypoint(self) -> Optional[Tuple[float, float]]:
        """The waypoint the wheelchair is currently heading toward."""
        return self.waypoints[0] if self.waypoints else None

    @property
    def all_waypoints(self) -> List[Tuple[float, float]]:
        """All pending waypoints (for rendering)."""
        return list(self.waypoints)

    def get_desired_attitude(self, cx: float, cy: float) -> float:
        """
        Compute desired heading angle toward the active waypoint.

        Parameters
        ----------
        cx, cy : float
            Current wheelchair position.

        Returns
        -------
        float
            Desired heading in radians (atan2 convention), or 0.0 if no
            active waypoint.
        """
        wp = self.active_waypoint
        if wp is None:
            return 0.0
        dx = wp[0] - cx
        dy = wp[1] - cy
        return math.atan2(dy, dx)

    def get_distance_to_active(self, cx: float, cy: float) -> float:
        """Euclidean distance from (cx, cy) to active waypoint, or inf."""
        wp = self.active_waypoint
        if wp is None:
            return float("inf")
        return math.hypot(wp[0] - cx, wp[1] - cy)

    def advance_if_reached(self, cx: float, cy: float) -> bool:
        """
        Pop the active waypoint if the wheelchair is within capture_radius.

        Returns True if a waypoint was consumed.
        """
        if self.active_waypoint is None:
            return False
        if self.get_distance_to_active(cx, cy) < self.capture_radius:
            reached = self.waypoints.popleft()
            self._visited.append(reached)
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  2. Speed Profile Design (SPD)
# ═══════════════════════════════════════════════════════════════════════════

class SpeedProfileDesigner:
    """
    Designs a trapezoidal speed profile that:
      - Accelerates up to v_max respecting a_x_ub (upper accel bound)
      - Cruises at v_max when far from the waypoint
      - Decelerates smoothly respecting a_x_lb (lower accel bound / brake)
      - Reduces to v_min_turn at waypoints with high curvature

    Uses a stateful internal speed reference (`_v_ref`) that ramps
    independently of the measured speed.  This is critical because the
    0.1 m/s² acceleration constraint produces tiny per-step increments
    (0.0002 m/s at dt=0.002), and if the reference is anchored to the
    measured speed, the PID torques are too small to overcome static
    friction — causing the wheelchair to stall.

    By maintaining _v_ref independently, the reference climbs at 0.1 m/s²
    and the PID receives progressively larger targets, generating enough
    torque to start and maintain motion.
    """

    def __init__(self,
                 v_max: float = 1.2,
                 a_x_lb: float = -0.1,
                 a_x_ub: float = 0.1,
                 v_min_turn: float = 0.1,
                 dt: float = 0.002):
        """
        Parameters
        ----------
        v_max : float
            Maximum cruising speed (m/s).
        a_x_lb : float
            Maximum braking deceleration (m/s², negative).
        a_x_ub : float
            Maximum forward acceleration (m/s²).  Capped at 0.1 m/s² per
            the wheelchair safety spec (Jung et al. 2020).
        v_min_turn : float
            Speed at sharp-turn waypoints (m/s).
        dt : float
            Physics timestep (s) — used to enforce the acceleration
            constraint correctly.
        """
        self.v_max      = v_max
        self.a_x_lb     = a_x_lb   # negative
        self.a_x_ub     = a_x_ub   # positive
        self.v_min_turn = v_min_turn
        self.dt         = dt

        # Internal speed reference — ramps independently of measured speed
        self._v_ref: float = 0.0

    def reset(self):
        """Reset the internal speed reference (call on sim reset)."""
        self._v_ref = 0.0

    def compute(self, dist_to_wp: float, heading_error: float,
                current_v: float) -> float:
        """
        Compute the desired forward speed.

        Enforces the 0.1 m/s² acceleration constraint by ramping an
        internal reference (`_v_ref`) at a_x_ub per timestep.  The
        reference is independent of the measured speed so the PID always
        receives a target large enough to overcome friction.

        Parameters
        ----------
        dist_to_wp : float
            Distance to the active waypoint (m).
        heading_error : float
            Heading error (rad).
        current_v : float
            Current forward speed (m/s).  Used only for safety checks,
            NOT for the acceleration ramp.

        Returns
        -------
        float
            Desired speed v_desired (m/s), non-negative.
        """
        if dist_to_wp == float("inf"):
            self._v_ref = 0.0
            return 0.0

        abs_err = abs(heading_error)

        # ── Braking speed limit ──
        v_brake = math.sqrt(max(0.0,
                                2.0 * abs(self.a_x_lb) * dist_to_wp))

        # ── Heading-error speed limiting (cosine-based, smooth) ──
        # Only reduce speed for large heading errors (>35°) to avoid
        # oscillation caused by small heading corrections
        if abs_err > math.radians(35):
            # Cosine taper: full speed at 35°, crawl at 90°+
            t = min(1.0, (abs_err - math.radians(35)) / math.radians(55))
            heading_scale = 0.5 * (1.0 + math.cos(math.pi * t))
            v_heading = max(0.20, self.v_max * heading_scale)
        else:
            v_heading = self.v_max

        # ── Near-waypoint deceleration zone ──
        crawl_zone = 2.0
        if dist_to_wp < crawl_zone:
            frac = dist_to_wp / crawl_zone
            v_approach = self.v_min_turn + (self.v_max - self.v_min_turn) * frac * frac
        else:
            v_approach = self.v_max

        # ── Target ceiling (minimum of all limits) ──
        v_target = min(v_brake, v_heading, v_approach, self.v_max)

        # ── Ramp internal reference at 0.1 m/s² ──
        # Stop completely for large heading errors to prevent arc drift
        if abs_err > math.radians(60):
            accel_scale = 0.0    # full stop — turn in place
            self._v_ref = 0.0    # reset speed reference
        elif abs_err > math.radians(30):
            accel_scale = 0.3 * (1.0 - (abs_err - math.radians(30)) / math.radians(30))
        else:
            accel_scale = 1.0

        # Ramp up (independent of measured speed)
        self._v_ref += self.a_x_ub * accel_scale * self.dt

        # If target is lower, decelerate the reference (a_x_lb is fast)
        if v_target < self._v_ref:
            self._v_ref = max(v_target,
                              self._v_ref - abs(self.a_x_lb) * self.dt)

        # Clamp
        self._v_ref = max(0.0, min(self._v_ref, v_target))

        return self._v_ref


# ═══════════════════════════════════════════════════════════════════════════
#  3. Angular Rate Gain Adaptation (ARGA)
# ═══════════════════════════════════════════════════════════════════════════

class AngularRateGainAdapter:
    """
    Dynamically adapts the angular rate gain K so that the lateral body
    acceleration constraint is not violated:

        |a_y| = |v · ω| = |v · K · θ_err| ≤ a_y_max

    Therefore:
        K ≤ a_y_max / (|v| · |θ_err|)    when v ≠ 0 and θ_err ≠ 0

    At standstill (v ≈ 0), K is set to a high default to allow rapid
    reorientation.
    """

    def __init__(self,
                 a_y_max: float = 1.5,
                 tau_c: float = 0.1,
                 K_default: float = 4.0,
                 K_min: float = 0.5):
        """
        Parameters
        ----------
        a_y_max : float
            Maximum allowable lateral acceleration (m/s²).
        tau_c : float
            System time delay (s) — used for first-order lag compensation.
        K_default : float
            Default gain when the wheelchair is nearly stationary.
        K_min : float
            Lower bound on K to ensure the wheelchair always steers.
        """
        self.a_y_max   = a_y_max
        self.tau_c     = tau_c
        self.K_default = K_default
        self.K_min     = K_min
        self._prev_err = 0.0   # for derivative damping

    def compute(self, v: float, theta_error: float) -> float:
        """
        Compute the adapted angular rate gain K.

        Parameters
        ----------
        v : float
            Current forward speed (m/s).
        theta_error : float
            Signed heading error (rad).

        Returns
        -------
        float
            Angular rate gain K.
        """
        abs_v   = abs(v)
        abs_err = abs(theta_error)

        if abs_v < 0.05 or abs_err < 0.01:
            # At standstill or nearly aligned → use full default gain
            return self.K_default

        # Clamp gain so lateral accel stays within budget
        K_max = self.a_y_max / (abs_v * abs_err)
        K = min(self.K_default, K_max)
        return max(self.K_min, K)

    def omega_desired(self, v: float, theta_error: float,
                      dt: float = 0.002) -> float:
        """
        Compute the desired angular velocity ω = K · θ_err with derivative
        damping to prevent oscillation.

        The gain K is adapted to respect the lateral-acceleration constraint.

        Parameters
        ----------
        v : float
            Current forward speed (m/s).
        theta_error : float
            Signed heading error (rad), wrapped to [-π, π].
        dt : float
            Physics timestep for derivative computation.

        Returns
        -------
        float
            Desired yaw rate ω (rad/s).
        """
        K = self.compute(v, theta_error)
        omega = K * theta_error

        # Derivative damping: reduce oscillation when error is changing fast
        d_err = (theta_error - self._prev_err) / dt if dt > 0 else 0.0
        self._prev_err = theta_error
        # Only apply damping when derivative opposes the error (overshoot)
        Kd = 0.05
        omega -= Kd * d_err

        # Clamp to a sane maximum (physical limit of the platform)
        omega = np.clip(omega, -4.0, 4.0)
        return float(omega)


# ═══════════════════════════════════════════════════════════════════════════
#  4. Modified HRVO (mHRVO) — Collision Avoidance
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Obstacle:
    """Representation of a circular dynamic obstacle."""
    x: float       # position
    y: float
    vx: float = 0.0   # velocity
    vy: float = 0.0
    radius: float = 0.3


class ModifiedHRVO:
    """
    Local collision-avoidance wrapper implementing a simplified Modified
    Hybrid Reciprocal Velocity Obstacle (mHRVO).

    Currently mocked for an obstacle-free environment: the wrapper simply
    passes through the incoming velocity commands.  All parameters needed
    for a full implementation are exposed.

    To enable obstacle avoidance, populate `self.obstacles` with Obstacle
    instances and implement the HRVO cone calculation in `_compute_hrvo()`.

    References
    ----------
    - Snape et al., "The Hybrid Reciprocal Velocity Obstacle", 2011.
    - Jung et al. (2020), Section IV-D.
    """

    def __init__(self, tau_c: float = 0.1, safety_margin: float = 0.15,
                 robot_radius: float = 0.35):
        """
        Parameters
        ----------
        tau_c : float
            System time delay (s) — determines the planning horizon.
        safety_margin : float
            Extra clearance around obstacles (m).
        robot_radius : float
            Effective radius of the wheelchair footprint (m).
        """
        self.tau_c         = tau_c
        self.safety_margin = safety_margin
        self.robot_radius  = robot_radius
        self.obstacles: List[Obstacle] = []

    # ── Obstacle management ────────────────────────────────────────

    def add_obstacle(self, x: float, y: float,
                     vx: float = 0.0, vy: float = 0.0,
                     radius: float = 0.3):
        """Register a dynamic obstacle."""
        self.obstacles.append(Obstacle(x, y, vx, vy, radius))

    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles.clear()

    def update_obstacles(self, obstacle_list: List[Obstacle]):
        """Replace the full obstacle list (for integration with sensors)."""
        self.obstacles = list(obstacle_list)

    # ── Core computation ───────────────────────────────────────────

    def compute(self, pos: Tuple[float, float],
                vel: Tuple[float, float],
                heading: float,
                v_desired: float,
                omega_desired: float) -> Tuple[float, float]:
        """
        Compute collision-free velocity commands.

        In the current mock (obstacle-free) mode, commands are passed
        through unmodified.  When obstacles are present, the method
        projects the desired velocity onto the feasible set outside
        all HRVO cones.

        Parameters
        ----------
        pos : (x, y)
            Current position.
        vel : (vx, vy)
            Current velocity.
        heading : float
            Current heading (rad).
        v_desired : float
            Desired forward speed from SPD (m/s).
        omega_desired : float
            Desired yaw rate from ARGA (rad/s).

        Returns
        -------
        (v_safe, omega_safe)
        """
        if not self.obstacles:
            return v_desired, omega_desired

        # Filter out distant obstacles (beyond 6 meters)
        close_obs = []
        for obs in self.obstacles:
            if math.hypot(obs.x - pos[0], obs.y - pos[1]) < 6.0:
                close_obs.append(obs)
                
        if not close_obs:
            return v_desired, omega_desired

        # Setup projection horizon
        plan_time = 2.0
        dt = 0.2
        steps = int(plan_time / dt)
        
        best_score = float('-inf')
        best_v = 0.0
        best_omega = 0.0

        # Define search grid for v and omega
        v_grid = [0.0, 0.3, 0.6, 0.9, 1.2]
        w_grid = [-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
        
        # Include target command as a candidate
        candidates = set((v, w) for v in v_grid for w in w_grid)
        candidates.add((v_desired, omega_desired))

        for cand_v, cand_w in candidates:
            # Simulate trajectory
            sim_x, sim_y, sim_theta = pos[0], pos[1], heading
            min_dist = float('inf')
            collision = False
            
            for t_step in range(1, steps + 1):
                t = t_step * dt
                
                if abs(cand_w) < 1e-4:
                    sim_x += cand_v * math.cos(sim_theta) * dt
                    sim_y += cand_v * math.sin(sim_theta) * dt
                else:
                    sim_x += (cand_v / cand_w) * (math.sin(sim_theta + cand_w * dt) - math.sin(sim_theta))
                    sim_y -= (cand_v / cand_w) * (math.cos(sim_theta + cand_w * dt) - math.cos(sim_theta))
                    sim_theta += cand_w * dt
                    
                # Check collision with moving obstacles
                for obs in close_obs:
                    obs_x_t = obs.x + obs.vx * t
                    obs_y_t = obs.y + obs.vy * t
                    
                    dist_to_obs = math.hypot(sim_x - obs_x_t, sim_y - obs_y_t)
                    safe_dist = self.robot_radius + obs.radius + self.safety_margin
                    
                    if dist_to_obs < safe_dist:
                        collision = True
                        break
                        
                    min_dist = min(min_dist, dist_to_obs)
                    
                if collision:
                    break
                    
            if collision:
                continue

            # Score this valid candidate
            v_err = abs(cand_v - v_desired) / 1.2
            w_err = abs(cand_w - omega_desired) / 4.0
            clearance_bonus = min(2.0, min_dist)
            
            # Penalize deviation from desired, reward clearance
            score = - (v_err * 2.0) - (w_err * 1.5) + (clearance_bonus * 0.5)
            
            if score > best_score:
                best_score = score
                best_v = cand_v
                best_omega = cand_w
                
        # If no safe paths, emergency stop
        if best_score == float('-inf'):
            return 0.0, 0.0
            
        return best_v, best_omega

    # ── Attitude-from-standstill support ───────────────────────────

    def allows_standstill_turn(self, pos: Tuple[float, float],
                               heading: float,
                               desired_heading: float) -> bool:
        """
        Check whether rotating in place at `pos` from `heading` to
        `desired_heading` is collision-free.

        Always True in obstacle-free mode.
        """
        if not self.obstacles:
            return True

        # Check if any obstacle is within robot_radius + safety_margin
        for obs in self.obstacles:
            dist = math.hypot(obs.x - pos[0], obs.y - pos[1])
            if dist < self.robot_radius + obs.radius + self.safety_margin:
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
#  5. Stuck Recovery
# ═══════════════════════════════════════════════════════════════════════════

import time as _time

class StuckRecovery:
    """
    Detects when the wheelchair is pinned against a wall and executes a
    3-phase escape manoeuvre to free it:

      Phase 0 — NORMAL   : regular planning, monitoring displacement
      Phase 1 — REVERSE  : drive backward for `t_reverse` seconds
      Phase 2 — PIVOT    : turn in place for `t_pivot` seconds
      Phase 3 — RESUME   : hand control back to the normal pipeline

    Detection condition
    -------------------
    Stuck if BOTH of the following hold for `stuck_timeout` seconds while
    a navigation target is active:
      1. Linear displacement below `disp_threshold`
      2. Angular displacement below `angle_threshold`

    The pivot direction is chosen intelligently based on wall clearance:
    the wheelchair turns AWAY from the nearest wall.

    Parameters
    ----------
    stuck_timeout : float
        Seconds without progress before declaring stuck (default 6.0).
    disp_threshold : float
        Minimum linear displacement (m) required to *not* be considered
        stuck (default 0.15 m).
    angle_threshold : float
        Minimum angular displacement (rad) required to *not* be considered
        stuck (default 0.3 rad ≈ 17°).
    t_reverse : float
        Duration (s) of the backward-escape burst (default 2.0).
    t_pivot : float
        Duration (s) of the pivot turn after reversing (default 1.5).
    v_escape : float
        Reverse speed magnitude (m/s) during escape (default 0.4).
    omega_pivot : float
        Yaw rate (rad/s) during pivot phase (default 1.5).
    cooldown : float
        Seconds after escape completes before re-detection is allowed
        (default 5.0).
    """

    # ── Phase IDs ──
    NORMAL  = 0
    REVERSE = 1
    PIVOT   = 2

    def __init__(self,
                 stuck_timeout:    float = 10.0,
                 disp_threshold:   float = 0.3,
                 angle_threshold:  float = 0.5,
                 t_reverse:        float = 1.0,
                 t_pivot:          float = 0.8,
                 v_escape:         float = 0.2,
                 omega_pivot:      float = 1.0,
                 cooldown:         float = 5.0,
                 wall_guard=None):
        self.stuck_timeout   = stuck_timeout
        self.disp_threshold  = disp_threshold
        self.angle_threshold = angle_threshold
        self.t_reverse       = t_reverse
        self.t_pivot         = t_pivot
        self.v_escape        = v_escape
        self.omega_pivot     = omega_pivot
        self.cooldown        = cooldown
        self._wall_guard     = wall_guard  # WallProximityGuard for smart pivot

        # Internal state
        self._phase:          int   = self.NORMAL
        self._phase_start:    float = 0.0
        self._ref_x:          float = 0.0
        self._ref_y:          float = 0.0
        self._ref_heading:    float = 0.0
        self._window_start:   float = _time.monotonic()
        self._pivot_sign:     float = 1.0
        self._last_sign:      float = 1.0
        self._cooldown_until: float = 0.0
        self._stuck_count:    int   = 0  # escalating recovery

    # ── Public API ─────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """True while the escape manoeuvre is running."""
        return self._phase != self.NORMAL

    def reset(self):
        """Call whenever the waypoint queue is cleared or sim is reset."""
        self._phase          = self.NORMAL
        self._window_start   = _time.monotonic()
        self._ref_x = self._ref_y = 0.0
        self._ref_heading    = 0.0
        self._cooldown_until = 0.0
        self._stuck_count    = 0

    def _choose_pivot_direction(self, x: float, y: float,
                                heading: float) -> float:
        """Choose pivot direction: turn AWAY from nearest wall."""
        if self._wall_guard is not None:
            left_c, right_c = self._wall_guard.get_lateral_clearances(
                x, y, heading)
            if left_c > right_c + 0.5:
                return 1.0   # turn left (more space)
            elif right_c > left_c + 0.5:
                return -1.0  # turn right (more space)
        # Fallback: alternate direction
        sign = -self._last_sign
        self._last_sign = sign
        return sign

    def update(self, x: float, y: float,
               has_target: bool,
               heading: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Call once per planning step.  Returns (v_cmd, omega_cmd) while
        the escape is active; returns (None, None) during normal operation
        (caller should use its own planning output instead).
        """
        now = _time.monotonic()

        # ── Phase transitions ───────────────────────────────────────
        if self._phase == self.REVERSE:
            # Escalate: longer reverse for repeat stucks
            t_rev = self.t_reverse + 0.5 * min(self._stuck_count, 3)
            if now - self._phase_start >= t_rev:
                self._phase       = self.PIVOT
                self._phase_start = now
                print("[RECOVERY] Switching to pivot phase")
            return (-self.v_escape, self._pivot_sign * 0.3)

        if self._phase == self.PIVOT:
            t_piv = self.t_pivot + 0.5 * min(self._stuck_count, 3)
            if now - self._phase_start >= t_piv:
                self._phase          = self.NORMAL
                self._window_start   = now
                self._ref_x, self._ref_y = x, y
                self._ref_heading    = heading
                self._cooldown_until = now + self.cooldown
                print("[RECOVERY] Escape complete — resuming navigation")
            else:
                return (0.0, self._pivot_sign * self.omega_pivot)

        # ── NORMAL phase: monitor for stuck condition ────────────────
        if not has_target:
            self._window_start = now
            self._ref_x, self._ref_y = x, y
            self._ref_heading = heading
            return (None, None)

        # Check BOTH linear AND angular displacement
        disp = math.hypot(x - self._ref_x, y - self._ref_y)
        angle_disp = abs((heading - self._ref_heading + math.pi)
                         % (2 * math.pi) - math.pi)

        if disp >= self.disp_threshold or angle_disp >= self.angle_threshold:
            # Made progress (either moved or turned) — slide window
            self._window_start = now
            self._ref_x, self._ref_y = x, y
            self._ref_heading = heading
            self._stuck_count = 0  # reset escalation
        elif (now - self._window_start >= self.stuck_timeout
              and now >= self._cooldown_until):
            # No linear OR angular progress AND cooldown expired
            self._stuck_count += 1
            print(f"[RECOVERY] Stuck detected at ({x:.2f}, {y:.2f}) — "
                  f"initiating escape (attempt {self._stuck_count})")
            self._pivot_sign   = self._choose_pivot_direction(x, y, heading)
            self._phase        = self.REVERSE
            self._phase_start  = now
            self._window_start = now
            self._ref_x, self._ref_y = x, y
            self._ref_heading  = heading
            return (-self.v_escape, self._pivot_sign * 0.3)

        return (None, None)


# ═══════════════════════════════════════════════════════════════════════════
#  6. Wall Proximity Guard  (static wall repulsion)
# ═══════════════════════════════════════════════════════════════════════════

class WallProximityGuard:
    """
    Prevents wall collisions by computing a repulsive steering correction
    and speed reduction when the wheelchair is near a wall.

    Walls are modelled as axis-aligned line segments extracted from the
    hospital layout in wheelchair_model.py.  For each wall, we compute
    the perpendicular distance from the wheelchair centre and, when
    within `danger_dist`, blend in:
      • A repulsive omega that steers the wheelchair **away** from the wall
      • A speed reduction proportional to proximity

    This acts as a safety layer AFTER the planner pipeline, overriding
    commands that would drive the wheelchair into a wall.
    """

    # Wall segments loaded from DXF at import time
    @staticmethod
    def _load_dxf_segments():
        from ras.map.dxf_parser import get_wall_segments
        return get_wall_segments()  # loads ALL polylines (outer + inner)

    WALL_SEGMENTS: List[Tuple[float, float, float, float]] = []

    def __init__(self,
                 danger_dist: float = 2.0,
                 critical_dist: float = 0.8,
                 repulsion_gain: float = 5.0,
                 robot_radius: float = 0.40):
        """
        Parameters
        ----------
        danger_dist : float
            Distance (m) at which wall repulsion starts.
        critical_dist : float
            Distance (m) at which emergency braking occurs.
        repulsion_gain : float
            Gain for repulsive steering correction.
        robot_radius : float
            Effective radius of the wheelchair.
        """
        self.danger_dist    = danger_dist
        self.critical_dist  = critical_dist
        self.repulsion_gain = repulsion_gain
        self.robot_radius   = robot_radius
        # Load wall segments from DXF if not already loaded
        if not WallProximityGuard.WALL_SEGMENTS:
            WallProximityGuard.WALL_SEGMENTS = self._load_dxf_segments()

    @staticmethod
    def _point_to_segment_dist(px: float, py: float,
                               x1: float, y1: float,
                               x2: float, y2: float) -> Tuple[float, float, float]:
        """
        Compute distance from point (px, py) to line segment (x1,y1)-(x2,y2).

        Returns (distance, nx, ny) where (nx, ny) is the unit normal from
        the wall toward the point.
        """
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            # Degenerate segment
            d = math.hypot(px - x1, py - y1)
            if d < 1e-12:
                return d, 0.0, 0.0
            return d, (px - x1) / d, (py - y1) / d

        # Project point onto line, clamped to [0, 1]
        t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))

        # Closest point on segment
        cx = x1 + t * dx
        cy = y1 + t * dy

        # Vector from closest point to the wheelchair
        rx = px - cx
        ry = py - cy
        d = math.hypot(rx, ry)
        if d < 1e-12:
            # On the wall — push perpendicular to segment
            # Normal is (-dy, dx) or (dy, -dx) of the segment
            seg_len = math.sqrt(seg_len_sq)
            return 0.0, -dy / seg_len, dx / seg_len
        return d, rx / d, ry / d

    def get_lateral_clearances(self, x: float, y: float, heading: float) -> Tuple[float, float]:
        """
        Calculates wall clearance on the left and right sides of the wheelchair.
        Returns (left_clearance, right_clearance).
        """
        min_left = float('inf')
        min_right = float('inf')
        
        # Left direction vector
        left_dx = -math.sin(heading)
        left_dy = math.cos(heading)
        
        for (x1, y1, x2, y2) in self.WALL_SEGMENTS:
            dist, nx, ny = self._point_to_segment_dist(x, y, x1, y1, x2, y2)
            if dist > 3.0: # ignore far walls
                continue
                
            # Vector FROM wheelchair TO wall is (-nx, -ny) * dist
            vx = -nx * dist
            vy = -ny * dist
            
            # Project onto the left-pointing axis
            dot = vx * left_dx + vy * left_dy
            
            if dot > 0.1: # definitively on the left
                min_left = min(min_left, dist)
            elif dot < -0.1: # definitively on the right
                min_right = min(min_right, dist)
                
        return min_left, min_right

    def compute(self, x: float, y: float, heading: float,
                v_cmd: float, omega_cmd: float) -> Tuple[float, float]:
        """
        Apply wall-proximity repulsion to velocity commands.

        Parameters
        ----------
        x, y : float
            Current wheelchair position.
        heading : float
            Current heading (rad).
        v_cmd, omega_cmd : float
            Incoming velocity commands from the planner pipeline.

        Returns
        -------
        (v_safe, omega_safe)
        """
        # Determine actual travel direction for repulsion logic
        is_reversing = v_cmd < 0
        move_heading = heading + math.pi if is_reversing else heading

        hx = math.cos(move_heading)
        hy = math.sin(move_heading)

        total_omega_correction = 0.0
        min_speed_factor = 1.0

        for (x1, y1, x2, y2) in self.WALL_SEGMENTS:
            dist, nx, ny = self._point_to_segment_dist(x, y, x1, y1, x2, y2)

            if dist > self.danger_dist:
                continue

            # ── Speed reduction ──
            if dist < self.critical_dist:
                # Very close — strong brake
                speed_factor = 0.15
            else:
                # Linear ramp: 1.0 at danger_dist → 0.2 at critical_dist
                t = (dist - self.critical_dist) / (self.danger_dist - self.critical_dist)
                speed_factor = 0.2 + 0.8 * t
            min_speed_factor = min(min_speed_factor, speed_factor)

            # ── Repulsive steering ──
            # Only steer if we're heading TOWARD the wall
            # dot of heading with wall-normal: negative means heading toward wall
            heading_toward_wall = -(hx * nx + hy * ny)
            if heading_toward_wall > 0.1:  # Only push when actually heading toward wall
                # Cross product (hx,hy) × (nx,ny) determines which way to turn
                cross = hx * ny - hy * nx
                turn_sign = 1.0 if cross >= 0.0 else -1.0

                # Stronger repulsion when closer
                proximity = 1.0 - (dist - self.critical_dist) / (
                    self.danger_dist - self.critical_dist + 1e-6)
                proximity = max(0.0, min(1.0, proximity))

                # Scale by how directly we're heading toward the wall
                approach_factor = min(1.0, heading_toward_wall)
                omega_correction = turn_sign * self.repulsion_gain * proximity * approach_factor
                total_omega_correction += omega_correction

        # Apply corrections
        v_safe = v_cmd * min_speed_factor
        omega_safe = omega_cmd + total_omega_correction

        # Clamp omega to physical limits
        omega_safe = max(-4.0, min(4.0, omega_safe))

        return v_safe, omega_safe


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class PathPlanner:
    """
    Orchestrates the full 4-stage pipeline:
        WAP → SPD → ARGA → mHRVO

    Call `plan(state)` once per physics step; it returns the (v, ω) commands
    to be fed into the DifferentialDriveController.
    """

    def __init__(self,
                 v_max: float = 1.2,
                 a_x_lb: float = -0.1,
                 a_x_ub: float = 0.1,
                 a_y_max: float = 1.5,
                 tau_c: float = 0.1,
                 capture_radius: float = 0.50):
        """
        Parameters
        ----------
        v_max, a_x_lb, a_x_ub : float
            Speed / longitudinal-acceleration constraints.
            a_x_ub is capped at 0.1 m/s² per the wheelchair safety spec.
        a_y_max : float
            Lateral-acceleration constraint.
        tau_c : float
            System time delay (s).
        capture_radius : float
            Distance at which a waypoint is considered reached (m).
        """
        self.wap   = WaypointAttitudePlanner(capture_radius=capture_radius)
        self.spd   = SpeedProfileDesigner(v_max=v_max, a_x_lb=a_x_lb,
                                           a_x_ub=a_x_ub)
        self.arga  = AngularRateGainAdapter(a_y_max=a_y_max, tau_c=tau_c)
        self.hrvo  = ModifiedHRVO(tau_c=tau_c)
        self._graph = HospitalGraph()   # wall-aware topological router
        self._wall_guard = WallProximityGuard()  # wall-proximity repulsion
        self._current_pos: Tuple[float, float] = (0.0, 0.0)
        self._recovery = StuckRecovery(wall_guard=self._wall_guard)
        self._user_waypoints: List[Tuple[float, float]] = []  # user clicked targets

    # ── Convenience wrappers ───────────────────────────────────────

    def add_waypoint(self, x: float, y: float):
        """
        Add a user-placed waypoint.

        Goes directly if the path is clear.  Falls back to nav-graph
        routing only when a wall blocks the straight line.
        """
        self._user_waypoints.append((x, y))

        if self.wap.waypoints:
            from_xy = self.wap.waypoints[-1]
        else:
            from_xy = self._current_pos

        # Check if straight line crosses any wall
        if self._path_crosses_wall(from_xy, (x, y)):
            # Wall in the way — use nav-graph routing
            route = self._graph.route(from_xy, (x, y))
            for wx, wy in route:
                self.wap.add_waypoint(wx, wy)
        else:
            # Clear path — go direct with subdivision
            subdivided = HospitalGraph._subdivide_path([from_xy, (x, y)])
            for wx, wy in subdivided[1:]:
                self.wap.add_waypoint(wx, wy)

    def _path_crosses_wall(self, a: Tuple[float, float],
                           b: Tuple[float, float],
                           margin: float = 1.0) -> bool:
        """Check if the straight line from a to b passes within `margin` of any wall."""
        ax, ay = a
        bx, by = b
        # Sample points along the line
        dist = math.hypot(bx - ax, by - ay)
        if dist < 0.1:
            return False
        n_samples = max(int(dist / 0.5), 5)
        for i in range(n_samples + 1):
            t = i / n_samples
            px = ax + t * (bx - ax)
            py = ay + t * (by - ay)
            for (x1, y1, x2, y2) in self._wall_guard.WALL_SEGMENTS:
                d, _, _ = self._wall_guard._point_to_segment_dist(
                    px, py, x1, y1, x2, y2)
                if d < margin:
                    return True
        return False

    @property
    def waypoints(self):
        """User-placed waypoint positions (for rendering)."""
        return list(self._user_waypoints)

    @property
    def all_nav_waypoints(self):
        """All pending navigation waypoints including routing intermediates."""
        return self.wap.all_waypoints

    # ── Main planning step ─────────────────────────────────────────

    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-π, π]."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    def plan_classical(self, state: dict) -> Tuple[float, float]:
        """
        Run the classical 4-stage pipeline (WAP -> SPD -> ARGA -> mHRVO)
        without the final WallProximityGuard correction.

        This is used by the RL environment to observe the pipeline's raw intent.
        
        Parameters
        ----------
        state : dict
            As returned by WheelchairPhysics.get_state().

        Returns
        -------
        (v_cmd, omega_cmd) : tuple of float
            Forward speed (m/s) and yaw rate (rad/s).
        """
        x, y   = state["x"], state["y"]
        theta  = state["theta"]
        vx, vy = state["vx"], state["vy"]
        v_now  = math.hypot(vx, vy)

        # Keep current position updated so add_waypoint can use it
        self._current_pos = (x, y)

        # ── Stage 1: WAP ──
        # Advance waypoint if reached
        self.wap.advance_if_reached(x, y)

        has_target = self.wap.active_waypoint is not None

        # ── Stuck Recovery (priority override) ──
        rv, rw = self._recovery.update(x, y, has_target, theta)
        if rv is not None:
            # Recovery is active — bypass normal planning
            assert rw is not None  # update() always returns both or neither
            return rv, rw

        if not has_target:
            return 0.0, 0.0

        theta_d   = self.wap.get_desired_attitude(x, y)
        dist      = self.wap.get_distance_to_active(x, y)
        theta_err = self._wrap_angle(theta_d - theta)

        # ── Waypoint lookahead ──
        # When close to current waypoint, blend heading toward the NEXT
        # waypoint for smoother transitions (avoids stop-turn-go).
        lookahead_zone = self.wap.capture_radius * 3.0
        if dist < lookahead_zone and len(self.wap.waypoints) > 1:
            next_wp = self.wap.waypoints[1]
            theta_next = math.atan2(next_wp[1] - y, next_wp[0] - x)
            theta_err_next = self._wrap_angle(theta_next - theta)
            # Blend: more weight on next waypoint as we get closer
            blend = 1.0 - (dist / lookahead_zone)
            blend = min(0.6, blend)  # cap blending at 60%
            theta_err = (1.0 - blend) * theta_err + blend * theta_err_next
            theta_err = self._wrap_angle(theta_err)

        # ── Stage 2: SPD ──
        v_desired = self.spd.compute(dist, theta_err, v_now)

        # ── Stage 3: ARGA ──
        omega_desired = self.arga.omega_desired(v_now, theta_err)

        # ── Stage 4: mHRVO ──
        v_algo, omega_algo = self.hrvo.compute(
            pos=(x, y), vel=(vx, vy), heading=theta,
            v_desired=v_desired, omega_desired=omega_desired)

        return v_algo, omega_algo

    def plan(self, state: dict) -> Tuple[float, float]:
        """
        Run the full classical pipeline and apply the WallProximityGuard.
        This provides safe navigation used out-of-the-box by main.py.
        """
        v_cmd, omega_cmd = self.plan_classical(state)
        
        # ── Stage 5: Wall Proximity Guard ──
        v_safe, omega_safe = self._wall_guard.compute(
            state["x"], state["y"], state["theta"], v_cmd, omega_cmd)
            
        return v_safe, omega_safe
