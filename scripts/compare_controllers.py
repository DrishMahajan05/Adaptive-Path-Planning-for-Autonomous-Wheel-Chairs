"""
_compare_controllers.py
=======================
Headless comparison: PID vs MPC on two paths with identical dynamic obstacles.
Generates 6 publication-quality PNG figures.

Output files:
  short_path_pid.png       short_path_mpc.png       short_path_overlay.png
  long_path_pid.png        long_path_mpc.png        long_path_overlay.png
"""
import math, copy, os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mujoco

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ras.physics.wheelchair_model import WheelchairPhysics
from ras.control.controller import DifferentialDriveController, MPCController
from ras.planning.path_planner import PathPlanner, WallProximityGuard
from ras.planning.obstacles import ObstacleManager

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.0, "figure.dpi": 150,
})

# ═══════════════════════════════════════════════════════════════════
#  Scenario definitions
# ═══════════════════════════════════════════════════════════════════

SCENARIOS = [
    {
        "name": "short_path",
        "title": "Short Path",
        "start": (32.37, 73.47),
        "goal":  (32.45, 52.92),
        "max_time": 120.0,
        "xlim": (20, 45),
        "ylim": (45, 82),
    },
    {
        "name": "long_path",
        "title": "Long Corridor Path",
        "start": (32.45, 52.92),
        "goal":  (-23.73, -99.49),
        "max_time": 300.0,
        "xlim": (-40, 55),
        "ylim": (-105, 85),
    },
]

# 3 custom obstacles placed IN the corridor path — crossing perpendicular
# so both controllers must dodge them.  Same for both PID and MPC runs.
CUSTOM_OBSTACLES = [
    # (x, y, vx, vy, radius) — crossing the corridor perpendicularly
    #  Obstacle 1: crossing the path around y=65 (short path zone)
    {"x": 30.0, "y": 65.0, "vx": 0.4, "vy": 0.0, "radius": 0.25},
    #  Obstacle 2: crossing around y=10 (mid long corridor)
    {"x": -18.0, "y": 10.0, "vx": 0.35, "vy": -0.15, "radius": 0.28},
    #  Obstacle 3: crossing around y=-50 (deep in long corridor)
    {"x": -22.0, "y": -50.0, "vx": 0.3, "vy": 0.1, "radius": 0.22},
]


# ═══════════════════════════════════════════════════════════════════
#  Simulation runner
# ═══════════════════════════════════════════════════════════════════

def run_simulation(start, goal, controller_type, max_time=300.0):
    """Run a headless simulation. Returns data dict."""
    physics = WheelchairPhysics()
    dt = physics.dt

    if controller_type == "PID":
        ctrl = DifferentialDriveController(
            wheel_base=physics.wheel_base, dt=dt,
            kp=10.0, ki=0.5, kd=1.5, torque_limit=10.0)
    else:
        ctrl = MPCController(
            wheel_base=physics.wheel_base, dt=dt,
            torque_limit=10.0, horizon=10, dt_mpc=0.02,
            Q_track=120.0, R_effort=0.05, Q_jerk=0.8)

    planner = PathPlanner(
        v_max=1.2, a_x_lb=-0.1, a_x_ub=0.1,
        a_y_max=1.5, tau_c=0.1, capture_radius=2.0)

    # Obstacles — identical for both controllers (custom placement)
    obs_mgr = ObstacleManager(num_obstacles=0, boundary=150.0, bounce=True)
    obs_mgr.obstacles.clear()
    for ob in CUSTOM_OBSTACLES:
        obs_mgr.add_custom(ob["x"], ob["y"], ob["vx"], ob["vy"], ob["radius"])
    # Save initial obstacle state so we log positions
    obs_init = [(o.x, o.y, o.vx, o.vy, o.radius) for o in obs_mgr.obstacles]

    # Set start pose
    physics.data.qpos[0] = start[0]
    physics.data.qpos[1] = start[1]
    physics.data.qpos[3] = 1.0
    physics.data.qpos[4:7] = 0.0
    physics.data.qvel[:] = 0
    mujoco.mj_forward(physics.model, physics.data)

    state = physics.get_state()
    planner._current_pos = (state["x"], state["y"])
    planner.add_waypoint(goal[0], goal[1])

    steps = int(max_time / dt)
    sample_every = int(0.05 / dt)  # 20 Hz

    # Acceleration limiter: clamp v_cmd rate to ±0.1 m/s² per second
    A_MAX = 0.1  # m/s²
    v_cmd_prev = 0.0

    D = {"t": [], "x": [], "y": [], "theta": [], "v": [],
         "omega": [], "v_cmd": [], "omega_cmd": [],
         "torque_l": [], "torque_r": []}

    reached = False
    for step in range(steps):
        state = physics.get_state()

        # Update obstacles and feed to planner
        obs_mgr.step(dt)
        planner.hrvo.update_obstacles(obs_mgr.as_planner_obstacles())

        v_cmd, omega_cmd = planner.plan(state)

        # ── Rate-limit v_cmd to enforce ±0.1 m/s² ──
        dv_max = A_MAX * dt
        dv = v_cmd - v_cmd_prev
        if dv > dv_max:
            v_cmd = v_cmd_prev + dv_max
        elif dv < -dv_max:
            v_cmd = v_cmd_prev - dv_max
        v_cmd_prev = v_cmd

        tl, tr = ctrl.compute(v_cmd, omega_cmd,
                              state["v_left"], state["v_right"])
        physics.set_ctrl(tl, tr)
        physics.step()

        if step % sample_every == 0:
            D["t"].append(step * dt)
            D["x"].append(state["x"])
            D["y"].append(state["y"])
            D["theta"].append(state["theta"])
            D["v"].append(math.hypot(state["vx"], state["vy"]))
            D["omega"].append(state["omega"])
            D["v_cmd"].append(v_cmd)
            D["omega_cmd"].append(omega_cmd)
            D["torque_l"].append(tl)
            D["torque_r"].append(tr)

        if math.hypot(goal[0]-state["x"], goal[1]-state["y"]) < 2.0:
            t_end = step * dt
            print(f"  GOAL at t={t_end:.1f}s  pos=({state['x']:.2f}, {state['y']:.2f})")
            reached = True
            # Coast a few more seconds
            for ex in range(int(2.0 / dt)):
                state = physics.get_state()
                physics.step()
                if ex % sample_every == 0:
                    D["t"].append((step + ex) * dt)
                    D["x"].append(state["x"])
                    D["y"].append(state["y"])
                    D["theta"].append(state["theta"])
                    D["v"].append(math.hypot(state["vx"], state["vy"]))
                    D["omega"].append(state["omega"])
                    D["v_cmd"].append(0.0)
                    D["omega_cmd"].append(0.0)
                    D["torque_l"].append(0.0)
                    D["torque_r"].append(0.0)
            break

    if not reached:
        state = physics.get_state()
        fd = math.hypot(goal[0]-state["x"], goal[1]-state["y"])
        print(f"  TIMEOUT t={max_time:.0f}s  dist_remain={fd:.1f}m")

    for k in D:
        D[k] = np.array(D[k])

    D["a_x"] = np.gradient(D["v"], D["t"])
    D["a_y"] = D["v"] * D["omega"]
    D["theta_deg"] = np.degrees(D["theta"])
    D["obs_init"] = obs_init
    D["reached"] = reached
    return D


# ═══════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═══════════════════════════════════════════════════════════════════

walls = None  # lazy-loaded

def _get_walls():
    global walls
    if walls is None:
        walls = WallProximityGuard.WALL_SEGMENTS
        if not walls:
            WallProximityGuard()
            walls = WallProximityGuard.WALL_SEGMENTS
    return walls

def _draw_map(ax, D, start, goal, color, label, xlim, ylim):
    """Draw walls, trajectory, start/goal markers."""
    for (x1, y1, x2, y2) in _get_walls():
        ax.plot([x1, x2], [y1, y2], "k-", linewidth=0.7)
    ax.plot(D["x"], D["y"], color=color, linewidth=1.8, label=label, zorder=3)
    ax.plot(start[0], start[1], "go", markersize=8, zorder=5)
    ax.plot(goal[0], goal[1], "rs", markersize=8, zorder=5)
    ax.annotate("Start", xy=start, xytext=(start[0]+1, start[1]+1),
                fontsize=8, color="green", fontweight="bold")
    ax.annotate("Goal", xy=goal, xytext=(goal[0]+1, goal[1]+1),
                fontsize=8, color="red", fontweight="bold")
    # Obstacle initial positions
    for (ox, oy, _, _, r) in D["obs_init"]:
        circle = plt.Circle((ox, oy), r, color="orange", alpha=0.5, zorder=4)
        ax.add_patch(circle)
    # Direction arrows
    step = max(1, len(D["x"]) // 12)
    for i in range(0, len(D["x"]), step):
        dx = 1.5 * math.cos(D["theta"][i])
        dy = 1.5 * math.sin(D["theta"][i])
        ax.annotate("", xy=(D["x"][i]+dx, D["y"][i]+dy),
                    xytext=(D["x"][i], D["y"][i]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.7))
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc="upper right", fontsize=7)


def plot_single(D, start, goal, ctrl_name, scenario, xlim, ylim, filename):
    """6-subplot figure for a single controller."""
    color = "b" if ctrl_name == "PID" else "r"
    fig = plt.figure(figsize=(12, 8))
    ax_map = fig.add_axes([0.04, 0.08, 0.38, 0.85])
    ax_v   = fig.add_axes([0.50, 0.76, 0.46, 0.17])
    ax_th  = fig.add_axes([0.50, 0.56, 0.46, 0.16], sharex=ax_v)
    ax_ax  = fig.add_axes([0.50, 0.38, 0.46, 0.14], sharex=ax_v)
    ax_ay  = fig.add_axes([0.50, 0.22, 0.46, 0.13], sharex=ax_v)
    ax_tq  = fig.add_axes([0.50, 0.05, 0.46, 0.14], sharex=ax_v)

    _draw_map(ax_map, D, start, goal, color, ctrl_name, xlim, ylim)
    ax_map.set_title(f"(a)  Trajectory — {ctrl_name}", fontsize=10)

    # Speed
    ax_v.plot(D["t"], D["v"], color=color, linewidth=0.8, label="actual")
    ax_v.plot(D["t"], D["v_cmd"], "g:", linewidth=0.7, label="commanded")
    ax_v.set_ylabel("$v$ [m/s]")
    ax_v.legend(loc="upper right", ncol=2, fontsize=6)
    ax_v.set_ylim(-0.05, 1.5)
    ax_v.grid(True, alpha=0.3, linestyle=":")
    plt.setp(ax_v.get_xticklabels(), visible=False)

    # Heading
    ax_th.plot(D["t"], D["theta_deg"], color=color, linewidth=0.8)
    ax_th.set_ylabel(r"$\theta$ [deg]")
    ax_th.grid(True, alpha=0.3, linestyle=":")
    plt.setp(ax_th.get_xticklabels(), visible=False)

    # a_x
    ax_ax.plot(D["t"], D["a_x"], color=color, linewidth=0.6)
    ax_ax.axhline(y=0.1, color="red", ls="--", lw=0.7, label="bound")
    ax_ax.axhline(y=-0.1, color="red", ls="--", lw=0.7)
    ax_ax.set_ylabel("$a_x$ [m/s²]")
    ax_ax.set_ylim(-0.5, 0.5)
    ax_ax.legend(loc="upper right", fontsize=6)
    ax_ax.grid(True, alpha=0.3, linestyle=":")
    plt.setp(ax_ax.get_xticklabels(), visible=False)

    # a_y
    ax_ay.plot(D["t"], D["a_y"], color=color, linewidth=0.6)
    ax_ay.axhline(y=0.4, color="red", ls="--", lw=0.7, label="bound")
    ax_ay.axhline(y=-0.4, color="red", ls="--", lw=0.7)
    ax_ay.set_ylabel("$a_y$ [m/s²]")
    ax_ay.legend(loc="upper right", fontsize=6)
    ax_ay.grid(True, alpha=0.3, linestyle=":")
    plt.setp(ax_ay.get_xticklabels(), visible=False)

    # Torques
    ax_tq.plot(D["t"], D["torque_l"], color=color, linewidth=0.6, label="left", alpha=0.8)
    ax_tq.plot(D["t"], D["torque_r"], color=color, linewidth=0.6, label="right",
               alpha=0.8, linestyle="--")
    ax_tq.set_ylabel("Torque [N·m]")
    ax_tq.set_xlabel("time [s]")
    ax_tq.legend(loc="upper right", fontsize=6)
    ax_tq.grid(True, alpha=0.3, linestyle=":")

    fig.text(0.73, 0.96, f"(b)  {ctrl_name} — {scenario}",
             ha="center", fontsize=10, fontweight="bold")

    plt.savefig(filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_overlay(D_pid, D_mpc, start, goal, scenario, xlim, ylim, filename):
    """6-subplot overlay comparing PID (blue) vs MPC (red)."""
    fig = plt.figure(figsize=(12, 8))
    ax_map = fig.add_axes([0.04, 0.08, 0.38, 0.85])
    ax_v   = fig.add_axes([0.50, 0.76, 0.46, 0.17])
    ax_th  = fig.add_axes([0.50, 0.56, 0.46, 0.16], sharex=ax_v)
    ax_ax  = fig.add_axes([0.50, 0.38, 0.46, 0.14], sharex=ax_v)
    ax_ay  = fig.add_axes([0.50, 0.22, 0.46, 0.13], sharex=ax_v)
    ax_tq  = fig.add_axes([0.50, 0.05, 0.46, 0.14], sharex=ax_v)

    # Map — both trajectories
    for (x1, y1, x2, y2) in _get_walls():
        ax_map.plot([x1, x2], [y1, y2], "k-", linewidth=0.7)
    ax_map.plot(D_pid["x"], D_pid["y"], "b-", linewidth=1.6, label="PID", zorder=3)
    ax_map.plot(D_mpc["x"], D_mpc["y"], "r-", linewidth=1.6, label="MPC", zorder=4)
    ax_map.plot(start[0], start[1], "go", markersize=8, zorder=5)
    ax_map.plot(goal[0], goal[1], "rs", markersize=8, zorder=5)
    ax_map.annotate("Start", xy=start, xytext=(start[0]+1, start[1]+1),
                    fontsize=8, color="green", fontweight="bold")
    ax_map.annotate("Goal", xy=goal, xytext=(goal[0]+1, goal[1]+1),
                    fontsize=8, color="red", fontweight="bold")
    for (ox, oy, _, _, r) in D_pid["obs_init"]:
        ax_map.add_patch(plt.Circle((ox, oy), r, color="orange", alpha=0.5, zorder=4))
    ax_map.set_xlabel("$x$ [m]"); ax_map.set_ylabel("$y$ [m]")
    ax_map.set_aspect("equal"); ax_map.grid(True, alpha=0.2, linestyle=":")
    ax_map.set_xlim(xlim); ax_map.set_ylim(ylim)
    ax_map.legend(loc="upper right", fontsize=7)
    ax_map.set_title("(a)  Trajectory comparison", fontsize=10)

    # Speed
    ax_v.plot(D_pid["t"], D_pid["v"], "b-", lw=0.8, label="PID")
    ax_v.plot(D_mpc["t"], D_mpc["v"], "r-", lw=0.8, label="MPC")
    ax_v.set_ylabel("$v$ [m/s]"); ax_v.set_ylim(-0.05, 1.5)
    ax_v.legend(loc="upper right", ncol=2, fontsize=6)
    ax_v.grid(True, alpha=0.3, linestyle=":"); plt.setp(ax_v.get_xticklabels(), visible=False)

    # Heading
    ax_th.plot(D_pid["t"], D_pid["theta_deg"], "b-", lw=0.8, label="PID")
    ax_th.plot(D_mpc["t"], D_mpc["theta_deg"], "r-", lw=0.8, label="MPC")
    ax_th.set_ylabel(r"$\theta$ [deg]"); ax_th.legend(loc="upper right", fontsize=6)
    ax_th.grid(True, alpha=0.3, linestyle=":"); plt.setp(ax_th.get_xticklabels(), visible=False)

    # a_x
    ax_ax.plot(D_pid["t"], D_pid["a_x"], "b-", lw=0.6, label="PID")
    ax_ax.plot(D_mpc["t"], D_mpc["a_x"], "r-", lw=0.6, label="MPC")
    ax_ax.axhline(y=0.1, color="gray", ls="--", lw=0.7, label="bound")
    ax_ax.axhline(y=-0.1, color="gray", ls="--", lw=0.7)
    ax_ax.set_ylabel("$a_x$ [m/s²]"); ax_ax.set_ylim(-0.5, 0.5)
    ax_ax.legend(loc="upper right", fontsize=6, ncol=3)
    ax_ax.grid(True, alpha=0.3, linestyle=":"); plt.setp(ax_ax.get_xticklabels(), visible=False)

    # a_y
    ax_ay.plot(D_pid["t"], D_pid["a_y"], "b-", lw=0.6, label="PID")
    ax_ay.plot(D_mpc["t"], D_mpc["a_y"], "r-", lw=0.6, label="MPC")
    ax_ay.axhline(y=0.4, color="gray", ls="--", lw=0.7, label="bound")
    ax_ay.axhline(y=-0.4, color="gray", ls="--", lw=0.7)
    ax_ay.set_ylabel("$a_y$ [m/s²]")
    ax_ay.legend(loc="upper right", fontsize=6, ncol=3)
    ax_ay.grid(True, alpha=0.3, linestyle=":"); plt.setp(ax_ay.get_xticklabels(), visible=False)

    # Torques
    ax_tq.plot(D_pid["t"], D_pid["torque_l"], "b-", lw=0.5, label="PID L", alpha=0.7)
    ax_tq.plot(D_pid["t"], D_pid["torque_r"], "b--", lw=0.5, label="PID R", alpha=0.7)
    ax_tq.plot(D_mpc["t"], D_mpc["torque_l"], "r-", lw=0.5, label="MPC L", alpha=0.7)
    ax_tq.plot(D_mpc["t"], D_mpc["torque_r"], "r--", lw=0.5, label="MPC R", alpha=0.7)
    ax_tq.set_ylabel("Torque [N·m]"); ax_tq.set_xlabel("time [s]")
    ax_tq.legend(loc="upper right", fontsize=5, ncol=4)
    ax_tq.grid(True, alpha=0.3, linestyle=":")

    fig.text(0.73, 0.96, f"(b)  PID vs MPC — {scenario}",
             ha="center", fontsize=10, fontweight="bold")

    plt.savefig(filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for sc in SCENARIOS:
        name  = sc["name"]
        title = sc["title"]
        start = sc["start"]
        goal  = sc["goal"]
        mt    = sc["max_time"]
        xl, yl = sc["xlim"], sc["ylim"]

        print(f"\n{'='*60}")
        print(f"  Scenario: {title}")
        print(f"  {start} -> {goal}")
        print(f"{'='*60}")

        print(f"\n  Running PID...")
        D_pid = run_simulation(start, goal, "PID", mt)

        print(f"  Running MPC...")
        D_mpc = run_simulation(start, goal, "MPC", mt)

        # Print comparison stats
        print(f"\n  -- Stats --")
        for label, D in [("PID", D_pid), ("MPC", D_mpc)]:
            rms_ax = float(np.sqrt(np.mean(D["a_x"]**2)))
            rms_ay = float(np.sqrt(np.mean(D["a_y"]**2)))
            rms_tq = float(np.sqrt(np.mean(D["torque_l"]**2 + D["torque_r"]**2)))
            jerk = float(np.sqrt(np.mean(np.diff(D["torque_l"])**2 +
                                         np.diff(D["torque_r"])**2)))
            print(f"  {label}: RMS a_x={rms_ax:.4f}  RMS a_y={rms_ay:.4f}  "
                  f"RMS torque={rms_tq:.3f}  jerk={jerk:.3f}")

        # Generate 3 figures per scenario
        plot_single(D_pid, start, goal, "PID", title, xl, yl, f"{name}_pid.png")
        plot_single(D_mpc, start, goal, "MPC", title, xl, yl, f"{name}_mpc.png")
        plot_overlay(D_pid, D_mpc, start, goal, title, xl, yl, f"{name}_overlay.png")

    print(f"\n{'='*60}")
    print("  All 6 figures generated successfully!")
    print(f"{'='*60}")
