"""
wheelchair_model.py
====================
MuJoCo XML model definition and physics wrapper for a differential-drive
electric wheelchair.

The wheelchair has:
  - A rectangular chassis (free body)
  - Two actuated rear drive wheels (hinge joints, motor actuators)
  - Two passive front casters (yaw hinge + spin hinge, no actuators)
  - A textured ground plane

Wall geometry is loaded from `map_dxf.dxf` via the dxf_parser module.

Class WheelchairPhysics provides a clean interface for state readout and
control input, abstracting the raw MuJoCo C-API.
"""

import numpy as np
import mujoco

from ras.map.dxf_parser import (
    load_wall_polyline,
    load_all_wall_polylines,
    get_wall_segments,
    generate_mujoco_wall_geoms,
    get_map_extents,
    get_start_position,
)

# ---------------------------------------------------------------------------
#  Load DXF and compute wall geoms at module load time
#  Uses ALL polylines (outer + inner walls) for complete corridor boundaries
# ---------------------------------------------------------------------------

_wall_segments = get_wall_segments()          # all polylines → all segments
_wall_geom_xml = generate_mujoco_wall_geoms(_wall_segments)
_x_min, _x_max, _y_min, _y_max = get_map_extents()  # broadest extents
_start_x, _start_y = get_start_position()    # midpoint between walls

# Ground plane half-size: cover map with generous margin
_ground_half = max(abs(_x_min), abs(_x_max), abs(_y_min), abs(_y_max)) + 30.0

# Light / camera centre
_light_cx = (_x_min + _x_max) / 2.0
_light_cy = (_y_min + _y_max) / 2.0

print(f"[wheelchair_model] DXF map loaded: {len(_wall_segments)} wall segments")
print(f"  Map extents: X=[{_x_min:.1f}, {_x_max:.1f}], "
      f"Y=[{_y_min:.1f}, {_y_max:.1f}]")
print(f"  Start position: ({_start_x:.2f}, {_start_y:.2f})")

# ---------------------------------------------------------------------------
#  MuJoCo XML Model
# ---------------------------------------------------------------------------

WHEELCHAIR_XML = f"""\
<mujoco model="wheelchair">
  <compiler angle="radian" autolimits="true"/>

  <!-- ───────────── Simulation options ───────────── -->
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicit"/>

  <!-- ───────────── Visual defaults ───────────── -->
  <default>
    <geom condim="3" friction="1.0 0.005 0.001" rgba="0.4 0.4 0.4 1"/>
    <joint damping="0.02"/>
    <motor ctrllimited="true" ctrlrange="-10 10"/>
  </default>

  <!-- ───────────── Assets ───────────── -->
  <asset>
    <!-- Ground texture — hospital floor tile -->
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.92 0.90 0.85" rgb2="0.85 0.83 0.78"
             width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="100 100"
              reflectance="0.15"/>
    <!-- Wheelchair body material -->
    <material name="chassis_mat" rgba="0.15 0.15 0.15 1" specular="0.6"/>
    <material name="wheel_mat"   rgba="0.05 0.05 0.05 1"/>
    <material name="caster_mat"  rgba="0.3 0.3 0.3 1"/>
    <!-- Waypoint marker material -->
    <material name="marker_mat"  rgba="1 0.15 0.15 0.9" emission="0.5"/>
  </asset>

  <!-- ───────────── World ───────────── -->
  <worldbody>
    <!-- Ground plane (covers the full DXF map with margin) -->
    <geom name="ground" type="plane" size="{_ground_half:.0f} {_ground_half:.0f} 0.1"
          material="grid_mat" conaffinity="1" condim="3"/>

    <!-- Ambient light (centred on the map) -->
    <light pos="{_light_cx:.1f} {_light_cy:.1f} 50" dir="0 0 -1"
           diffuse="0.95 0.93 0.88"
           specular="0.3 0.3 0.3" castshadow="true"/>
    <light pos="{_light_cx + 20:.1f} {_light_cy - 30:.1f} 35"
           dir="0.3 0 -1" diffuse="0.4 0.4 0.4"
           specular="0.1 0.1 0.1" castshadow="false"/>

    <!-- ═══════════ HOSPITAL WALLS (from DXF) ═══════════ -->
{_wall_geom_xml}

    <!-- ================ WHEELCHAIR ================ -->
    <body name="chassis" pos="{_start_x:.2f} {_start_y:.2f} 0.25">
      <!-- 6-DOF free joint for the chassis -->
      <joint name="root" type="free"/>
      <inertial pos="0 0 0" mass="80"
                diaginertia="4.0 2.0 3.0"/>

      <!-- Main chassis geometry (seat + frame) -->
      <geom name="chassis_box" type="box" size="0.30 0.25 0.08"
            pos="0 0 0" material="chassis_mat"/>
      <!-- Seat back -->
      <geom name="seat_back" type="box" size="0.04 0.22 0.20"
            pos="-0.28 0 0.28" material="chassis_mat"/>

      <!-- ──── SEATED PERSON ──── -->
      <!-- Torso (blue hospital gown) -->
      <geom name="person_torso" type="capsule" size="0.12 0.18"
            pos="-0.05 0 0.36" euler="0 0 0"
            rgba="0.25 0.45 0.70 1" contype="0" conaffinity="0"/>
      <!-- Head -->
      <geom name="person_head" type="sphere" size="0.10"
            pos="-0.08 0 0.66"
            rgba="0.85 0.72 0.60 1" contype="0" conaffinity="0"/>
      <!-- Left arm -->
      <geom name="person_arm_l" type="capsule" size="0.04 0.14"
            pos="0.05 0.18 0.28" euler="0 1.3 0"
            rgba="0.85 0.72 0.60 1" contype="0" conaffinity="0"/>
      <!-- Right arm -->
      <geom name="person_arm_r" type="capsule" size="0.04 0.14"
            pos="0.05 -0.18 0.28" euler="0 1.3 0"
            rgba="0.85 0.72 0.60 1" contype="0" conaffinity="0"/>

      <!-- ──── LEFT REAR DRIVE WHEEL ──── -->
      <body name="left_wheel" pos="-0.05 0.30 -0.05">
        <joint name="left_wheel_joint" type="hinge"
               axis="0 1 0" damping="0.05"/>
        <geom name="left_wheel_geom" type="cylinder"
              size="0.20 0.04" material="wheel_mat"
              euler="1.5708 0 0" mass="2"/>
      </body>

      <!-- ──── RIGHT REAR DRIVE WHEEL ──── -->
      <body name="right_wheel" pos="-0.05 -0.30 -0.05">
        <joint name="right_wheel_joint" type="hinge"
               axis="0 1 0" damping="0.05"/>
        <geom name="right_wheel_geom" type="cylinder"
              size="0.20 0.04" material="wheel_mat"
              euler="1.5708 0 0" mass="2"/>
      </body>

      <!-- ──── LEFT FRONT CASTER ──── -->
      <body name="left_caster_yaw" pos="0.25 0.20 -0.10">
        <!-- Vertical yaw hinge (free-spinning) -->
        <joint name="left_caster_yaw_joint" type="hinge"
               axis="0 0 1" damping="0.001"/>
        <geom name="left_caster_fork" type="cylinder"
              size="0.01 0.04" rgba="0.5 0.5 0.5 1"/>
        <body name="left_caster_wheel" pos="0.03 0 -0.06">
          <!-- Spin hinge (free-spinning) -->
          <joint name="left_caster_spin_joint" type="hinge"
                 axis="0 1 0" damping="0.001"/>
          <geom name="left_caster_geom" type="sphere"
                size="0.05" material="caster_mat" mass="0.5"/>
        </body>
      </body>

      <!-- ──── RIGHT FRONT CASTER ──── -->
      <body name="right_caster_yaw" pos="0.25 -0.20 -0.10">
        <joint name="right_caster_yaw_joint" type="hinge"
               axis="0 0 1" damping="0.001"/>
        <geom name="right_caster_fork" type="cylinder"
              size="0.01 0.04" rgba="0.5 0.5 0.5 1"/>
        <body name="right_caster_wheel" pos="0.03 0 -0.06">
          <joint name="right_caster_spin_joint" type="hinge"
                 axis="0 1 0" damping="0.001"/>
          <geom name="right_caster_geom" type="sphere"
                size="0.05" material="caster_mat" mass="0.5"/>
        </body>
      </body>

      <!-- Sensor site at chassis center (for heading readout) -->
      <site name="imu_site" pos="0 0 0" size="0.02"/>
    </body>
  </worldbody>

  <!-- ───────────── Actuators ───────────── -->
  <actuator>
    <motor name="left_motor"  joint="left_wheel_joint"  gear="1"/>
    <motor name="right_motor" joint="right_wheel_joint" gear="1"/>
  </actuator>

  <!-- ───────────── Sensors ───────────── -->
  <sensor>
    <framepos  name="chassis_pos"  objtype="site" objname="imu_site"/>
    <framequat name="chassis_quat" objtype="site" objname="imu_site"/>
    <gyro      name="chassis_gyro" site="imu_site"/>
  </sensor>
</mujoco>
"""


# ---------------------------------------------------------------------------
#  WheelchairPhysics Wrapper
# ---------------------------------------------------------------------------

class WheelchairPhysics:
    """
    Wraps the MuJoCo model + data and provides a clean state/control API.

    Key dimensions (matching the XML above):
      wheel_radius = 0.20 m   (rear drive wheel)
      wheel_base   = 0.60 m   (lateral distance between rear wheels)
    """

    # Physical constants matching the XML
    WHEEL_RADIUS = 0.20   # metres
    WHEEL_BASE   = 0.60   # metres (centre-to-centre of rear wheels)
    DT           = 0.002  # physics timestep (s)

    def __init__(self):
        """Load the wheelchair model and allocate data."""
        self.model = mujoco.MjModel.from_xml_string(WHEELCHAIR_XML)
        self.data  = mujoco.MjData(self.model)

        # Cache joint / actuator indices for fast access
        self._left_motor_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
        self._right_motor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")

        # Forward-step once to settle initial contacts
        mujoco.mj_forward(self.model, self.data)

    # ── State readout ──────────────────────────────────────────────

    def get_state(self):
        """
        Return the wheelchair's planar state.

        Returns
        -------
        dict with keys:
            x, y        : position on the ground plane (m)
            theta       : heading angle (rad, 0 = +x axis)
            vx, vy      : linear velocity in world frame (m/s)
            omega       : yaw rate (rad/s)
            v_left      : left wheel tangential speed (m/s)
            v_right     : right wheel tangential speed (m/s)
        """
        # Chassis position (free joint qpos: 3 pos + 4 quat)
        qpos = self.data.qpos
        x, y = qpos[0], qpos[1]

        # Heading from quaternion (qw, qx, qy, qz)
        qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
        # Yaw from quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        theta = np.arctan2(siny_cosp, cosy_cosp)

        # Chassis velocity (free joint qvel: 3 lin + 3 ang)
        qvel = self.data.qvel
        vx, vy = qvel[0], qvel[1]
        omega  = qvel[5]  # yaw rate

        # Wheel angular velocities → tangential speeds
        # Free joint uses 6 dofs in qvel, then hinge joints follow
        left_wheel_qvel_idx  = 6   # first hinge after free joint
        right_wheel_qvel_idx = 7
        v_left  = qvel[left_wheel_qvel_idx]  * self.WHEEL_RADIUS
        v_right = qvel[right_wheel_qvel_idx] * self.WHEEL_RADIUS

        return {
            "x": x, "y": y, "theta": theta,
            "vx": vx, "vy": vy, "omega": omega,
            "v_left": v_left, "v_right": v_right,
        }

    # ── Control input ──────────────────────────────────────────────

    def set_ctrl(self, left_torque: float, right_torque: float):
        """Set motor torques for left and right drive wheels."""
        self.data.ctrl[self._left_motor_id]  = left_torque
        self.data.ctrl[self._right_motor_id] = right_torque

    def step(self):
        """Advance the simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)

    # ── Convenience ────────────────────────────────────────────────

    @property
    def wheel_radius(self):
        return self.WHEEL_RADIUS

    @property
    def wheel_base(self):
        return self.WHEEL_BASE

    @property
    def dt(self):
        return self.DT

    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
