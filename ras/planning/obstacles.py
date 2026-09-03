"""
obstacles.py
============
Configurable obstacle spawner and manager for the wheelchair simulation.

Spawns random circular obstacles with constant velocities.  The user can
configure:
  - Number of obstacles
  - Size range (radius)
  - Speed range (magnitude of velocity)
  - Spawn area (region around the origin)
  - Whether obstacles bounce off boundaries or wrap around

Obstacles update their positions every physics step at constant velocity
and are fed into the mHRVO collision-avoidance module.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
#  Data class for a single moving obstacle
# ---------------------------------------------------------------------------

@dataclass
class MovingObstacle:
    """
    A circular obstacle moving at constant velocity on the ground plane.

    Attributes
    ----------
    x, y : float
        Current position (m).
    vx, vy : float
        Velocity components (m/s) — remain constant unless bounced.
    radius : float
        Collision radius (m).
    color : tuple
        RGBA colour for rendering.
    """
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.3
    color: Tuple[float, float, float, float] = (0.9, 0.6, 0.1, 0.85)


# ---------------------------------------------------------------------------
#  Obstacle Manager
# ---------------------------------------------------------------------------

class ObstacleManager:
    """
    Spawns, updates, and provides obstacle states for the simulation.

    Usage
    -----
    >>> mgr = ObstacleManager(
    ...     num_obstacles=5,
    ...     radius_range=(0.15, 0.5),
    ...     speed_range=(0.2, 1.0),
    ...     spawn_range=(3.0, 10.0),
    ...     boundary=15.0,
    ...     bounce=True,
    ... )
    >>> mgr.spawn()                 # create random obstacles
    >>> mgr.step(dt=0.002)          # call every physics step
    >>> obs_list = mgr.as_planner_obstacles()  # feed to mHRVO
    """

    def __init__(self,
                 num_obstacles: int = 5,
                 radius_range: Tuple[float, float] = (0.15, 0.45),
                 speed_range: Tuple[float, float] = (0.3, 0.8),
                 spawn_range: Tuple[float, float] = (3.0, 10.0),
                 boundary: float = 15.0,
                 bounce: bool = True,
                 seed: Optional[int] = None):
        """
        Parameters
        ----------
        num_obstacles : int
            Number of obstacles to spawn.
        radius_range : (min, max)
            Uniform random range for obstacle radius (m).
        speed_range : (min, max)
            Uniform random range for velocity magnitude (m/s).
        spawn_range : (min, max)
            Distance from origin at which obstacles are placed.
            Obstacles won't spawn right on top of the wheelchair.
        boundary : float
            Half-side of the square arena.  Obstacles that exit
            [-boundary, boundary] will bounce or wrap.
        bounce : bool
            If True, obstacles reflect off boundaries.
            If False, obstacles wrap around (toroidal).
        seed : int or None
            Random seed for reproducibility.
        """
        self.num_obstacles = num_obstacles
        self.radius_range  = radius_range
        self.speed_range   = speed_range
        self.spawn_range   = spawn_range
        self.boundary      = boundary
        self.bounce        = bounce

        self.obstacles: List[MovingObstacle] = []
        self._rng = random.Random(seed)

    # ── Spawning ───────────────────────────────────────────────────

    def spawn(self):
        """Create `num_obstacles` obstacles with random properties."""
        self.obstacles.clear()

        # Colour palette for visual variety
        palette = [
            (0.95, 0.45, 0.15, 0.85),   # orange
            (0.20, 0.65, 0.90, 0.85),   # blue
            (0.85, 0.20, 0.55, 0.85),   # magenta
            (0.15, 0.80, 0.45, 0.85),   # green
            (0.90, 0.85, 0.20, 0.85),   # yellow
            (0.60, 0.30, 0.80, 0.85),   # purple
        ]

        for i in range(self.num_obstacles):
            # Random distance and angle from origin
            dist  = self._rng.uniform(*self.spawn_range)
            angle = self._rng.uniform(0, 2 * math.pi)
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)

            # Random radius
            radius = self._rng.uniform(*self.radius_range)

            # Random velocity direction and magnitude
            speed     = self._rng.uniform(*self.speed_range)
            vel_angle = self._rng.uniform(0, 2 * math.pi)
            vx = speed * math.cos(vel_angle)
            vy = speed * math.sin(vel_angle)

            color = palette[i % len(palette)]

            self.obstacles.append(MovingObstacle(
                x=x, y=y, vx=vx, vy=vy, radius=radius, color=color))

    def add_custom(self, x: float, y: float,
                   vx: float, vy: float, radius: float,
                   color: Tuple[float, float, float, float] = (0.9, 0.6, 0.1, 0.85)):
        """
        Manually add a single obstacle with exact parameters.

        This lets the user choose the exact size, direction, and
        magnitude of velocity for any obstacle.
        """
        self.obstacles.append(MovingObstacle(
            x=x, y=y, vx=vx, vy=vy, radius=radius, color=color))

    # ── Stepping ───────────────────────────────────────────────────

    def step(self, dt: float):
        """
        Advance all obstacle positions by dt seconds.

        Handles boundary bouncing / wrapping.
        """
        b = self.boundary
        for obs in self.obstacles:
            obs.x += obs.vx * dt
            obs.y += obs.vy * dt

            if self.bounce:
                # Reflect off boundaries
                if obs.x - obs.radius < -b:
                    obs.x = -b + obs.radius
                    obs.vx = abs(obs.vx)
                elif obs.x + obs.radius > b:
                    obs.x = b - obs.radius
                    obs.vx = -abs(obs.vx)

                if obs.y - obs.radius < -b:
                    obs.y = -b + obs.radius
                    obs.vy = abs(obs.vy)
                elif obs.y + obs.radius > b:
                    obs.y = b - obs.radius
                    obs.vy = -abs(obs.vy)
            else:
                # Wrap around
                if obs.x < -b: obs.x += 2 * b
                if obs.x >  b: obs.x -= 2 * b
                if obs.y < -b: obs.y += 2 * b
                if obs.y >  b: obs.y -= 2 * b

    # ── Interface to planner ───────────────────────────────────────

    def as_planner_obstacles(self):
        """
        Convert to the Obstacle dataclass list expected by ModifiedHRVO.

        Returns
        -------
        list of path_planner.Obstacle
        """
        from ras.planning.path_planner import Obstacle
        return [
            Obstacle(x=o.x, y=o.y, vx=o.vx, vy=o.vy, radius=o.radius)
            for o in self.obstacles
        ]

    # ── Info ───────────────────────────────────────────────────────

    def __repr__(self):
        lines = [f"ObstacleManager({self.num_obstacles} obstacles):"]
        for i, o in enumerate(self.obstacles):
            speed = math.hypot(o.vx, o.vy)
            angle = math.degrees(math.atan2(o.vy, o.vx))
            lines.append(
                f"  [{i}] pos=({o.x:.1f}, {o.y:.1f})  "
                f"r={o.radius:.2f}m  "
                f"speed={speed:.2f}m/s  dir={angle:.0f}deg")
        return "\n".join(lines)
