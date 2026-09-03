"""Verify which polylines to keep and visualize them."""
import math
import matplotlib.pyplot as plt
from dxf_parser import load_all_wall_polylines, get_wall_segments

all_polys, cx, cy = load_all_wall_polylines()

# User says keep: poly 1 (inner corridor), poly 2 (room outer), poly 4 (box outer)
keep_indices = [1, 2, 4]
labels = {1: "Corridor boundary", 2: "Room obstacle", 4: "Box obstacle"}
colors_map = {1: "blue", 2: "green", 4: "purple"}

fig, ax = plt.subplots(1, 1, figsize=(10, 16))

for i, poly in enumerate(all_polys):
    wx = [v[0] for v in poly] + [poly[0][0]]
    wy = [v[1] for v in poly] + [poly[0][1]]
    if i in keep_indices:
        ax.plot(wx, wy, "-", color=colors_map[i], linewidth=2.5,
                label=f"KEEP: {labels[i]} (poly {i}, {len(poly)} pts)")
    else:
        ax.plot(wx, wy, "--", color="gray", linewidth=0.8, alpha=0.4,
                label=f"REMOVE: poly {i}")

# Print info about kept polylines
print("=== KEPT POLYLINES ===")
for i in keep_indices:
    poly = all_polys[i]
    xs = [v[0] for v in poly]
    ys = [v[1] for v in poly]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    print(f"  Poly {i} ({labels[i]}): {len(poly)} verts, {w:.1f} x {h:.1f}m")

# Count wall segments for kept polylines only
kept_seg_count = sum(len(all_polys[i]) for i in keep_indices)
print(f"\n  Total wall segments (kept only): {kept_seg_count}")
print(f"  Total wall segments (all): {sum(len(p) for p in all_polys)}")

# Check corridor width: distance from poly 1 to nearest interior obstacle
poly1 = all_polys[1]  # corridor boundary
poly2 = all_polys[2]  # room
poly4 = all_polys[4]  # box

def min_dist_between_polys(p1, p2):
    min_d = float('inf')
    for x1, y1 in p1:
        for x2, y2 in p2:
            d = math.hypot(x1-x2, y1-y2)
            min_d = min(min_d, d)
    return min_d

print(f"\n  Corridor->Room gap: {min_dist_between_polys(poly1, poly2):.2f}m")
print(f"  Corridor->Box gap: {min_dist_between_polys(poly1, poly4):.2f}m")
print(f"  Room->Box gap: {min_dist_between_polys(poly2, poly4):.2f}m")

ax.set_aspect("equal")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("New Map — Kept Polylines Only")
ax.grid(True, alpha=0.3)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
plt.tight_layout()
plt.savefig("map_kept_polys.png", dpi=150)
print("\nSaved map_kept_polys.png")
