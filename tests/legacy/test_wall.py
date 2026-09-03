import sys
sys.path.insert(0, r'C:\Users\vardh\Desktop\RAS')
from path_planner import WallProximityGuard
import math

wg = WallProximityGuard()

# Test 1: Far from walls - no effect
v, w = wg.compute(0.0, 0.0, 0.0, 1.0, 0.0)
assert abs(v - 1.0) < 0.01 and abs(w) < 0.01, f"FAIL: v={v}, w={w}"
print("PASS: Center corridor - no wall effect")

# Test 2: Near north corridor wall (y=1.5), heading toward it
v, w = wg.compute(0.0, 1.1, math.pi/2, 1.0, 0.0)
assert v < 1.0, f"FAIL: expected speed reduction, got v={v}"
assert abs(w) > 0.01, f"FAIL: expected omega correction, got w={w}"
print(f"PASS: Near N wall heading N: v={v:.3f} omega={w:.3f}")

# Test 3: Near west outer wall
v, w = wg.compute(-11.6, 0.0, math.pi, 1.0, 0.0)
assert v < 1.0, f"FAIL: expected speed reduction, got v={v}"
assert abs(w) > 0.01, f"FAIL: expected omega correction, got w={w}"
print(f"PASS: Near W wall heading W: v={v:.3f} omega={w:.3f}")

# Test 4: Near wall but heading AWAY - speed reduction only, minimal omega
v, w = wg.compute(0.0, 1.1, -math.pi/2, 1.0, 0.0)
assert v < 1.0, f"FAIL: expected speed reduction, got v={v}"
print(f"PASS: Near N wall heading S: v={v:.3f} omega={w:.3f}")

# Test 5: Critical distance - emergency brake
v, w = wg.compute(0.0, 1.3, math.pi/2, 1.0, 0.0)
assert v < 0.15, f"FAIL: expected near-zero v at critical distance, got v={v}"
print(f"PASS: Critical proximity: v={v:.3f} omega={w:.3f}")

print("\nAll WallProximityGuard tests passed!")
