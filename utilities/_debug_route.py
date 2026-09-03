from wheelchair_model import WheelchairPhysics
import numpy as np

p = WheelchairPhysics()
s = p.get_state()
print(f"Default qpos[0:7]: {p.data.qpos[:7]}")
print(f"Default theta: {s['theta']:.3f} rad = {np.degrees(s['theta']):.1f} deg")
print()

# When we place the wheelchair, we only set qpos[0] and qpos[1]
# The quaternion (qpos[3:7]) stays at whatever the default is
print(f"Quaternion: w={p.data.qpos[3]:.3f} x={p.data.qpos[4]:.3f} y={p.data.qpos[5]:.3f} z={p.data.qpos[6]:.3f}")

# The heading to go from (32.37, 73.47) to (32.45, 52.92) is -90 deg
# If default theta is 0 (east), the wheelchair must turn 90 degrees
# During that turn, any forward motion creates an arc
import math
theta_default = s['theta']
theta_target = math.atan2(52.92 - 73.47, 32.45 - 32.37)
err = ((theta_target - theta_default) + math.pi) % (2 * math.pi) - math.pi
print(f"\nHeading to target: {np.degrees(theta_target):.1f} deg")
print(f"Initial heading error: {np.degrees(err):.1f} deg")
print(f"Abs error > 90 deg? {abs(err) > math.radians(90)}")
