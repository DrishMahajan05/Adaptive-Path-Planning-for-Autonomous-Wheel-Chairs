"""
Research-paper-style plot for long corridor path:
  Start: (32.45, 52.92) → Goal: (-23.73, -99.49)
"""
import math, mujoco, os
os.environ["MPLBACKEND"] = "Agg"  # non-interactive
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wheelchair_model import WheelchairPhysics
from controller import DifferentialDriveController
from path_planner import PathPlanner, WallProximityGuard

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.0, "figure.dpi": 150,
})

# ── Setup ──
physics = WheelchairPhysics()
controller = DifferentialDriveController(
    wheel_base=physics.wheel_base, dt=physics.dt,
    kp=15.0, ki=2.0, kd=2.5, torque_limit=10.0)
planner = PathPlanner(
    v_max=1.2, a_x_lb=-1.0, a_x_ub=0.1,
    a_y_max=1.5, tau_c=0.1, capture_radius=2.0)

start = (32.45, 52.92)
goal  = (-23.73, -99.49)

physics.data.qpos[0] = start[0]
physics.data.qpos[1] = start[1]
physics.data.qpos[3] = 1.0  # identity quaternion
physics.data.qpos[4:7] = 0.0
physics.data.qvel[:] = 0
mujoco.mj_forward(physics.model, physics.data)

state = physics.get_state()
planner._current_pos = (state["x"], state["y"])
planner.add_waypoint(goal[0], goal[1])
wps_list = list(planner.wap.waypoints)

dist_total = math.hypot(goal[0]-start[0], goal[1]-start[1])
print(f"Start: {start}")
print(f"Goal:  {goal}")
print(f"Straight-line distance: {dist_total:.1f}m")
print(f"Waypoints: {len(wps_list)}")
for i, (x, y) in enumerate(wps_list):
    print(f"  [{i}] ({x:.1f}, {y:.1f})")

# ── Simulate ──
dt = physics.dt
max_time = 300.0  # 5 min max for long path
steps = int(max_time / dt)
sample_every = int(0.05 / dt)  # 20 Hz

D = {"t": [], "x": [], "y": [], "theta": [], "v": [],
     "omega": [], "v_cmd": [], "omega_cmd": [], "v_desired": []}

reached = False
for step in range(steps):
    state = physics.get_state()
    v_cmd, omega_cmd = planner.plan(state)
    v_cls, w_cls = planner.plan_classical(state)

    torque_l, torque_r = controller.compute(
        v_cmd, omega_cmd, state["v_left"], state["v_right"])
    physics.set_ctrl(torque_l, torque_r)
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
        D["v_desired"].append(v_cls)

    # Check goal
    if math.hypot(goal[0]-state["x"], goal[1]-state["y"]) < 2.0:
        t = step * dt
        print(f"\nGOAL REACHED at t={t:.1f}s  pos=({state['x']:.2f}, {state['y']:.2f})")
        reached = True
        # Record a few more seconds
        for extra in range(int(3.0 / dt)):
            state = physics.get_state()
            physics.step()
            if extra % sample_every == 0:
                D["t"].append((step + extra) * dt)
                D["x"].append(state["x"])
                D["y"].append(state["y"])
                D["theta"].append(state["theta"])
                D["v"].append(math.hypot(state["vx"], state["vy"]))
                D["omega"].append(state["omega"])
                D["v_cmd"].append(0.0)
                D["omega_cmd"].append(0.0)
                D["v_desired"].append(0.0)
        break

if not reached:
    state = physics.get_state()
    fd = math.hypot(goal[0]-state["x"], goal[1]-state["y"])
    print(f"\nTIMEOUT at t={max_time:.0f}s  pos=({state['x']:.2f}, {state['y']:.2f})  dist={fd:.1f}m")

for k in D:
    D[k] = np.array(D[k])

a_x = np.gradient(D["v"], D["t"])
a_y = D["v"] * D["omega"]
theta_deg = np.degrees(D["theta"])

# ── Load wall segments ──
walls = WallProximityGuard.WALL_SEGMENTS

# ════════════════════════════════════════════
#  FIGURE: Jung et al. style
# ════════════════════════════════════════════
fig = plt.figure(figsize=(12, 8))

ax_map = fig.add_axes([0.04, 0.08, 0.38, 0.85])
ax_v   = fig.add_axes([0.50, 0.76, 0.46, 0.17])
ax_th  = fig.add_axes([0.50, 0.54, 0.46, 0.17], sharex=ax_v)
ax_ax  = fig.add_axes([0.50, 0.32, 0.46, 0.17], sharex=ax_v)
ax_ay  = fig.add_axes([0.50, 0.08, 0.46, 0.19], sharex=ax_v)

# ─── (a) Map + trajectory ───
for (x1, y1, x2, y2) in walls:
    ax_map.plot([x1, x2], [y1, y2], "k-", linewidth=0.7)

ax_map.plot(D["x"], D["y"], "b-", linewidth=1.8, label="wheelchair", zorder=3)
ax_map.plot(start[0], start[1], "go", markersize=8, zorder=5)
ax_map.plot(goal[0], goal[1], "rs", markersize=8, zorder=5)
ax_map.annotate("Start", xy=start, xytext=(start[0]+3, start[1]+3),
                fontsize=8, color="green", fontweight="bold")
ax_map.annotate("Goal", xy=goal, xytext=(goal[0]+3, goal[1]+3),
                fontsize=8, color="red", fontweight="bold")

for i, (wx, wy) in enumerate(wps_list):
    ax_map.plot(wx, wy, "m^", markersize=4, zorder=4)

# Direction arrows along path
arrow_step = max(1, len(D["x"]) // 15)
for i in range(0, len(D["x"]), arrow_step):
    dx = 2.0 * math.cos(D["theta"][i])
    dy = 2.0 * math.sin(D["theta"][i])
    ax_map.annotate("", xy=(D["x"][i]+dx, D["y"][i]+dy),
                    xytext=(D["x"][i], D["y"][i]),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=0.7))

ax_map.set_xlabel("$x$ [m]")
ax_map.set_ylabel("$y$ [m]")
ax_map.set_title("(a)  Trajectory of the wheelchair", fontsize=10)
ax_map.legend(loc="upper right", fontsize=7)
ax_map.set_aspect("equal")
ax_map.grid(True, alpha=0.2, linestyle=":")
# Show full map
ax_map.set_xlim(-40, 55)
ax_map.set_ylim(-105, 85)

# ─── (b) Speed ───
ax_v.plot(D["t"], D["v"], "b-", linewidth=0.8, label="true")
ax_v.plot(D["t"], D["v_desired"], "r--", linewidth=0.7, label="desired")
ax_v.plot(D["t"], D["v_cmd"], "g:", linewidth=0.7, label="wall-guard")
ax_v.set_ylabel("$v$ [m/s]")
ax_v.legend(loc="upper right", ncol=3, fontsize=6)
ax_v.set_ylim(-0.05, 1.5)
ax_v.grid(True, alpha=0.3, linestyle=":")
plt.setp(ax_v.get_xticklabels(), visible=False)

# ─── Heading ───
ax_th.plot(D["t"], theta_deg, "b-", linewidth=0.8)
ax_th.set_ylabel(r"$\theta$ [deg]")
ax_th.grid(True, alpha=0.3, linestyle=":")
plt.setp(ax_th.get_xticklabels(), visible=False)

# ─── a_x ───
ax_ax.plot(D["t"], a_x, "b-", linewidth=0.6, label="true")
ax_ax.axhline(y=0.1, color="red", linestyle="--", linewidth=0.7, label="boundary")
ax_ax.axhline(y=-0.1, color="red", linestyle="--", linewidth=0.7)
ax_ax.set_ylabel("$a_x$ [m/s$^2$]")
ax_ax.legend(loc="upper right", fontsize=6)
ax_ax.set_ylim(-0.5, 0.5)
ax_ax.grid(True, alpha=0.3, linestyle=":")
plt.setp(ax_ax.get_xticklabels(), visible=False)

# ─── a_y ───
ax_ay.plot(D["t"], a_y, "b-", linewidth=0.6)
ax_ay.axhline(y=0.4, color="red", linestyle="--", linewidth=0.7, label="boundary")
ax_ay.axhline(y=-0.4, color="red", linestyle="--", linewidth=0.7)
ax_ay.set_ylabel("$a_y$ [m/s$^2$]")
ax_ay.set_xlabel("time [s]")
ax_ay.legend(loc="upper right", fontsize=6)
ax_ay.grid(True, alpha=0.3, linestyle=":")

fig.text(0.73, 0.96, "(b)  Speed, attitude, and body accelerations",
         ha="center", fontsize=10, fontweight="bold")

out = "navigation_long_path.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {out}")
