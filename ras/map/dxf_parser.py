"""
dxf_parser.py
=============
Parses the hospital DXF floor plan (map_dxf.dxf) and produces:
  1. Wall segments for the WallProximityGuard
  2. MuJoCo XML wall geometry strings
  3. Corridor-centreline navigation nodes for the HospitalGraph

The DXF contains 6 polylines, of which only 3 are used:
  - Poly 1: Corridor boundary (main navigable area)
  - Poly 2: Room obstacle (interior structure to navigate around)
  - Poly 4: Box obstacle (interior structure to navigate around)

Coordinates are converted from DXF units to metres using DXF_SCALE.
The map is translated so the centroid sits at the world origin (0, 0).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np


# ── Configuration ─────────────────────────────────────────────────────
DXF_SCALE = 0.3280       # 1 DXF unit = 0.328 m  ->  ~86 x 184 m map
WALL_THICKNESS = 0.30    # metres (half = 0.15)
WALL_HEIGHT = 3.0        # metres (half = 1.5)
RDP_TOLERANCE = 0.8      # DXF units — tighter simplification at larger scale
NAV_NODE_SPACING = 8.0   # metres between corridor-centreline graph nodes

# Polyline indices to keep from the DXF (0-indexed in parse order)
# Poly 1 = corridor boundary, Poly 2 = room obstacle, Poly 4 = box obstacle
KEEP_POLY_INDICES = [1, 2, 4]

from ras.config import DEFAULT_DXF_PATH

# ── DXF file path (assets directory) ──
_DXF_PATH = str(DEFAULT_DXF_PATH)



# ═══════════════════════════════════════════════════════════════════════
#  DXF Parsing
# ═══════════════════════════════════════════════════════════════════════

def _parse_dxf_polylines(filepath: str) -> List[dict]:
    """
    Parse a DXF file and return a list of polylines.

    Each polyline is a dict:
        {'vertices': [(x, y, bulge), ...], 'closed': bool}
    """
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    polylines = []
    current_vertices: list = []
    in_polyline = False
    is_closed = False

    i = 0
    while i < len(lines):
        if lines[i] == "0" and i + 1 < len(lines):
            entity = lines[i + 1]

            if entity == "POLYLINE":
                in_polyline = True
                current_vertices = []
                is_closed = False
                i += 2
                while i < len(lines) and lines[i] != "0":
                    if lines[i] == "70" and i + 1 < len(lines):
                        flag = int(lines[i + 1])
                        is_closed = bool(flag & 1)
                        i += 2
                    else:
                        i += 2
                continue

            elif entity == "VERTEX" and in_polyline:
                x = y = None
                bulge = 0.0
                i += 2
                while i < len(lines):
                    if lines[i] == "0":
                        break
                    gc = lines[i]
                    if i + 1 < len(lines):
                        val = lines[i + 1]
                        if gc == "10":
                            x = float(val)
                        elif gc == "20":
                            y = float(val)
                        elif gc == "42":
                            bulge = float(val)
                    i += 2
                if x is not None and y is not None:
                    current_vertices.append((x, y, bulge))
                continue

            elif entity == "SEQEND" and in_polyline:
                polylines.append({
                    "vertices": current_vertices,
                    "closed": is_closed,
                })
                in_polyline = False
                i += 2
                continue

        i += 1

    return polylines


# ═══════════════════════════════════════════════════════════════════════
#  Ramer-Douglas-Peucker Simplification
# ═══════════════════════════════════════════════════════════════════════

def _perpendicular_distance(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """Perpendicular distance from point (px,py) to line (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - x1, py - y1)
    cross = abs(dx * (y1 - py) - dy * (x1 - px))
    return cross / math.sqrt(length_sq)


def _rdp(points: List[Tuple[float, float]],
         epsilon: float) -> List[Tuple[float, float]]:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) <= 2:
        return list(points)

    # Find point with max distance from start-end line
    dmax = 0.0
    idx = 0
    x1, y1 = points[0]
    x2, y2 = points[-1]
    for i in range(1, len(points) - 1):
        d = _perpendicular_distance(points[i][0], points[i][1],
                                     x1, y1, x2, y2)
        if d > dmax:
            dmax = d
            idx = i

    if dmax > epsilon:
        left = _rdp(points[: idx + 1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


# ═══════════════════════════════════════════════════════════════════════
#  Core: Load, Scale, and Generate
# ═══════════════════════════════════════════════════════════════════════

# Module-level cache so the DXF is only parsed once
_CACHED_RESULT: dict | None = None


def _load_and_cache(
    filepath: str = _DXF_PATH,
    scale: float = DXF_SCALE,
    simplify_tolerance: float = RDP_TOLERANCE,
) -> dict:
    """Parse, simplify, scale, filter and centre polylines.  Cached."""
    global _CACHED_RESULT
    if _CACHED_RESULT is not None:
        return _CACHED_RESULT

    polylines = _parse_dxf_polylines(filepath)
    if not polylines:
        raise RuntimeError(f"No polylines found in {filepath}")

    # Process ALL polylines first (scale + simplify)
    all_processed: List[List[Tuple[float, float]]] = []
    for pl in polylines:
        pts_2d = [(v[0], v[1]) for v in pl["vertices"]]
        simplified = _rdp(pts_2d, simplify_tolerance)
        scaled = [(x * scale, y * scale) for x, y in simplified]
        all_processed.append(scaled)

    # Filter to only the kept polylines
    kept_pts: List[List[Tuple[float, float]]] = []
    for idx in KEEP_POLY_INDICES:
        if idx < len(all_processed):
            kept_pts.append(all_processed[idx])

    if not kept_pts:
        raise RuntimeError("No kept polylines after filtering")

    # Compute centroid from kept polylines only
    flat = [p for poly in kept_pts for p in poly]
    cx = sum(p[0] for p in flat) / len(flat)
    cy = sum(p[1] for p in flat) / len(flat)

    centred = [[(x - cx, y - cy) for x, y in poly] for poly in kept_pts]

    _CACHED_RESULT = {
        "polylines": centred,       # list of list-of-(x,y)
        "cx": cx, "cy": cy,
    }
    return _CACHED_RESULT


def load_wall_polyline(
    filepath: str = _DXF_PATH,
    scale: float = DXF_SCALE,
    simplify_tolerance: float = RDP_TOLERANCE,
) -> Tuple[List[Tuple[float, float]], float, float]:
    """
    Load the corridor boundary polyline (first kept polyline) from the DXF.

    Returns
    -------
    vertices : list of (x, y) in metres, centred at origin
    cx, cy   : world-coordinate centre offsets (for reference)
    """
    data = _load_and_cache(filepath, scale, simplify_tolerance)
    return data["polylines"][0], data["cx"], data["cy"]


def load_all_wall_polylines(
    filepath: str = _DXF_PATH,
    scale: float = DXF_SCALE,
    simplify_tolerance: float = RDP_TOLERANCE,
) -> Tuple[List[List[Tuple[float, float]]], float, float]:
    """
    Load *all* polylines from the DXF (outer + inner boundaries).

    Returns
    -------
    polylines : list of polyline vertex lists  [(x,y), ...]
    cx, cy    : world-coordinate centre offsets
    """
    data = _load_and_cache(filepath, scale, simplify_tolerance)
    return data["polylines"], data["cx"], data["cy"]


def _polyline_to_segments(
    vertices: List[Tuple[float, float]],
    closed: bool = True,
) -> List[Tuple[float, float, float, float]]:
    """Convert a single polyline into (x1,y1,x2,y2) segments."""
    segments: List[Tuple[float, float, float, float]] = []
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        if i + 1 < n:
            x2, y2 = vertices[i + 1]
        elif closed:
            x2, y2 = vertices[0]
        else:
            break
        segments.append((x1, y1, x2, y2))
    return segments


def get_wall_segments(
    vertices: List[Tuple[float, float]] | None = None,
    closed: bool = True,
) -> List[Tuple[float, float, float, float]]:
    """
    Convert polylines into a list of wall segments (x1, y1, x2, y2).

    If *vertices* is None, loads ALL polylines from the DXF file and
    concatenates their segments.
    """
    if vertices is not None:
        return _polyline_to_segments(vertices, closed)

    # Load all polylines and concatenate
    all_polys, _, _ = load_all_wall_polylines()
    segments: List[Tuple[float, float, float, float]] = []
    for poly in all_polys:
        segments.extend(_polyline_to_segments(poly, closed))
    return segments


# ═══════════════════════════════════════════════════════════════════════
#  MuJoCo XML Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_mujoco_wall_geoms(
    segments: List[Tuple[float, float, float, float]] | None = None,
    wall_thickness: float = WALL_THICKNESS,
    wall_height: float = WALL_HEIGHT,
) -> str:
    """
    Generate MuJoCo XML `<geom>` elements for each wall segment.

    Each segment becomes a box geom:
        pos = midpoint
        size = (half_length, half_thickness, half_height)
        euler = (0, 0, angle)

    If *segments* is None, loads ALL polylines from the DXF.
    """
    if segments is None:
        segments = get_wall_segments()

    half_t = wall_thickness / 2.0
    half_h = wall_height / 2.0
    lines = []

    for idx, (x1, y1, x2, y2) in enumerate(segments):
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        dx, dy = x2 - x1, y2 - y1
        half_len = math.hypot(dx, dy) / 2.0
        angle = math.atan2(dy, dx)

        lines.append(
            f'    <geom name="dxf_wall_{idx}" type="box" '
            f'pos="{mx:.4f} {my:.4f} {half_h:.1f}" '
            f'size="{half_len:.4f} {half_t:.2f} {half_h:.1f}" '
            f'euler="0 0 {angle:.6f}" '
            f'rgba="0.93 0.91 0.87 1" conaffinity="1"/>'
        )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Geometry Helpers (segment intersection, point-in-polygon)
# ═══════════════════════════════════════════════════════════════════════

def _segments_intersect(
    p1x: float, p1y: float, p2x: float, p2y: float,
    p3x: float, p3y: float, p4x: float, p4y: float,
) -> bool:
    """Test whether line segments (p1,p2) and (p3,p4) properly intersect."""
    d1x, d1y = p2x - p1x, p2y - p1y
    d2x, d2y = p4x - p3x, p4y - p3y
    cross = d1x * d2y - d1y * d2x
    if abs(cross) < 1e-10:
        return False
    t = ((p3x - p1x) * d2y - (p3y - p1y) * d2x) / cross
    u = ((p3x - p1x) * d1y - (p3y - p1y) * d1x) / cross
    return 0.01 < t < 0.99 and 0.01 < u < 0.99


def _point_inside_polygon(
    px: float, py: float,
    polygon: List[Tuple[float, float]],
) -> bool:
    """Ray-casting test: is (px, py) inside the closed polygon?"""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _line_crosses_any_wall(
    ax: float, ay: float, bx: float, by: float,
    wall_segments: List[Tuple[float, float, float, float]],
) -> bool:
    """Check if line (ax,ay)-(bx,by) crosses any wall segment."""
    for (x1, y1, x2, y2) in wall_segments:
        if _segments_intersect(ax, ay, bx, by, x1, y1, x2, y2):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════
#  Navigation Graph Generation (corridor-centreline nodes)
# ═══════════════════════════════════════════════════════════════════════

def generate_nav_graph(
    vertices: List[Tuple[float, float]] | None = None,
    node_spacing: float = NAV_NODE_SPACING,
) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str]]]:
    """
    Create a navigation graph inside the corridor boundary, avoiding
    interior obstacles (room, box).

    Architecture:
      - Poly 0 (first kept = corridor boundary): navigate INSIDE this
      - Poly 1+ (remaining kept = obstacles): navigate AROUND these

    Algorithm:
      1. Inset the corridor boundary inward by ~inset_distance metres
      2. Place nodes at regular intervals along the inset polyline
      3. Remove nodes that land inside any obstacle polygon
      4. Connect consecutive nodes (sequential edges)
      5. Add visibility-based shortcut edges (no wall crossings)

    Returns
    -------
    nodes : dict  {node_id: (x, y)}
    edges : list  [(node_a, node_b), ...]
    """
    # Load all kept polylines
    all_polys, _, _ = load_all_wall_polylines()

    # Poly 0 (first kept) = corridor boundary
    if vertices is None:
        vertices = all_polys[0]

    # Remaining polys = interior obstacles
    obstacle_polys = all_polys[1:] if len(all_polys) > 1 else []

    # All wall segments (for line-of-sight checks)
    all_wall_segs = get_wall_segments()

    # ── Inset the corridor boundary ──────────────────────────────
    inset_distance = 5.0  # metres inward from the corridor wall

    n = len(vertices)
    inset_pts: List[Tuple[float, float]] = []

    for i in range(n):
        x_prev, y_prev = vertices[(i - 1) % n]
        x_curr, y_curr = vertices[i]
        x_next, y_next = vertices[(i + 1) % n]

        e1x, e1y = x_curr - x_prev, y_curr - y_prev
        e2x, e2y = x_next - x_curr, y_next - y_curr

        len1 = math.hypot(e1x, e1y) + 1e-12
        len2 = math.hypot(e2x, e2y) + 1e-12
        n1x, n1y = -e1y / len1, e1x / len1
        n2x, n2y = -e2y / len2, e2x / len2

        bx = n1x + n2x
        by = n1y + n2y
        blen = math.hypot(bx, by)
        if blen < 1e-6:
            bx, by = n1x, n1y
            blen = 1.0
        bx /= blen
        by /= blen

        inset_pts.append((x_curr + bx * inset_distance,
                          y_curr + by * inset_distance))

    # Determine winding: inset should be smaller
    def polygon_area(pts):
        a = 0.0
        m = len(pts)
        for j in range(m):
            x1, y1 = pts[j]
            x2, y2 = pts[(j + 1) % m]
            a += x1 * y2 - x2 * y1
        return a / 2.0

    orig_area = abs(polygon_area(vertices))
    inset_area = abs(polygon_area(inset_pts))

    if inset_area > orig_area:
        # Went outward — flip
        inset_pts = [(vertices[i][0] - (inset_pts[i][0] - vertices[i][0]),
                      vertices[i][1] - (inset_pts[i][1] - vertices[i][1]))
                     for i in range(n)]

    # ── Sample nodes along the inset polyline ────────────────────
    arc_lengths = [0.0]
    for i in range(1, n):
        dx = inset_pts[i][0] - inset_pts[i - 1][0]
        dy = inset_pts[i][1] - inset_pts[i - 1][1]
        arc_lengths.append(arc_lengths[-1] + math.hypot(dx, dy))
    dx = inset_pts[0][0] - inset_pts[-1][0]
    dy = inset_pts[0][1] - inset_pts[-1][1]
    total_length = arc_lengths[-1] + math.hypot(dx, dy)

    num_nodes = max(4, int(total_length / node_spacing))
    actual_spacing = total_length / num_nodes

    def interp_at(s: float) -> Tuple[float, float]:
        s = s % total_length
        cum = 0.0
        for j in range(n):
            x1, y1 = inset_pts[j]
            x2, y2 = inset_pts[(j + 1) % n]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if cum + seg_len >= s - 1e-9:
                t = (s - cum) / (seg_len + 1e-12)
                t = max(0.0, min(1.0, t))
                return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            cum += seg_len
        return inset_pts[0]

    # Generate candidate nodes
    raw_nodes: List[Tuple[str, float, float]] = []
    for i in range(num_nodes):
        s = i * actual_spacing
        x, y = interp_at(s)
        raw_nodes.append((f"nav_{i}", x, y))

    # ── Filter: remove nodes inside obstacle polygons ────────────
    nodes: Dict[str, Tuple[float, float]] = {}
    for nid, nx, ny in raw_nodes:
        inside_obstacle = False
        for obs_poly in obstacle_polys:
            if _point_inside_polygon(nx, ny, obs_poly):
                inside_obstacle = True
                break
        if not inside_obstacle:
            nodes[nid] = (nx, ny)

    if not nodes:
        # Fallback: use raw nodes without filtering
        for nid, nx, ny in raw_nodes:
            nodes[nid] = (nx, ny)

    # ── Edges: sequential + visibility shortcuts ─────────────────
    edges: List[Tuple[str, str]] = []
    node_ids = list(nodes.keys())
    nn = len(node_ids)

    # Sequential edges (loop)
    for i in range(nn):
        a = node_ids[i]
        b = node_ids[(i + 1) % nn]
        ax, ay = nodes[a]
        bx, by = nodes[b]
        if not _line_crosses_any_wall(ax, ay, bx, by, all_wall_segs):
            edges.append((a, b))

    # Visibility-based shortcut edges
    max_shortcut_dist = 60.0  # metres
    for i in range(nn):
        for j in range(i + 2, nn):
            if j == (i + nn - 1) % nn:
                continue  # skip the wrap-around sequential neighbour
            a = node_ids[i]
            b = node_ids[j]
            ax, ay = nodes[a]
            bx, by = nodes[b]
            dist = math.hypot(bx - ax, by - ay)
            if dist > max_shortcut_dist:
                continue
            if not _line_crosses_any_wall(ax, ay, bx, by, all_wall_segs):
                edges.append((a, b))

    return nodes, edges


# ═══════════════════════════════════════════════════════════════════════
#  Convenience: compute map extents
# ═══════════════════════════════════════════════════════════════════════

def get_map_extents(
    vertices: List[Tuple[float, float]] | None = None,
) -> Tuple[float, float, float, float]:
    """
    Return (x_min, x_max, y_min, y_max) of the wall boundary in metres.

    If *vertices* is None, uses all polylines for the broadest extents.
    """
    if vertices is None:
        all_polys, _, _ = load_all_wall_polylines()
        vertices = [v for poly in all_polys for v in poly]
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return min(xs), max(xs), min(ys), max(ys)


def get_start_position(
    vertices: List[Tuple[float, float]] | None = None,
) -> Tuple[float, float]:
    """
    Return a suitable starting position for the wheelchair.

    If two polylines exist (outer + inner), the start is placed at the
    midpoint between the outer and inner boundaries near the bottom of
    the map.  Otherwise falls back to the single-polyline approach.
    """
    all_polys, _, _ = load_all_wall_polylines()

    if vertices is not None:
        # Caller provided explicit vertices — use single-poly logic
        all_verts = vertices
    elif len(all_polys) >= 2:
        # Two walls — place between them
        outer, inner = all_polys[0], all_polys[1]
        all_verts_combined = outer + inner
        min_y = min(v[1] for v in all_verts_combined)
        # Collect bottom vertices from BOTH walls
        bottom_outer = [(x, y) for x, y in outer if y < min_y + 5.0]
        bottom_inner = [(x, y) for x, y in inner if y < min_y + 5.0]
        bottom = bottom_outer + bottom_inner
        if bottom:
            sx = sum(v[0] for v in bottom) / len(bottom)
            sy = sum(v[1] for v in bottom) / len(bottom)
            return (sx, sy + 3.0)
        return (0.0, 0.0)
    else:
        all_verts = all_polys[0]

    # Single-polyline fallback
    min_y = min(v[1] for v in all_verts)
    bottom_verts = [(x, y) for x, y in all_verts
                    if y < min_y + 5.0]  # within 5m of bottom

    if bottom_verts:
        sx = sum(v[0] for v in bottom_verts) / len(bottom_verts)
        sy = sum(v[1] for v in bottom_verts) / len(bottom_verts)
        return (sx, sy + 3.0)

    # Fallback: centroid
    return (0.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    all_polys, cx, cy = load_all_wall_polylines()
    print(f"Loaded {len(all_polys)} polyline(s)")
    for i, poly in enumerate(all_polys):
        print(f"  Polyline {i}: {len(poly)} simplified vertices")
    print(f"  Centre offset: ({cx:.2f}, {cy:.2f}) m")

    segs = get_wall_segments()  # all polylines
    print(f"  Total wall segments: {len(segs)}")

    x_min, x_max, y_min, y_max = get_map_extents()
    print(f"  Map extents: X=[{x_min:.1f}, {x_max:.1f}], "
          f"Y=[{y_min:.1f}, {y_max:.1f}]")
    print(f"  Map size: {x_max - x_min:.1f} m × {y_max - y_min:.1f} m")

    start = get_start_position()
    print(f"  Start position: ({start[0]:.2f}, {start[1]:.2f})")

    nodes, edges = generate_nav_graph()
    print(f"  Nav nodes: {len(nodes)}")
    print(f"  Nav edges: {len(edges)}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 16))

    # Wall boundaries
    colors = ["blue", "red", "green", "orange"]
    for i, poly in enumerate(all_polys):
        wx = [v[0] for v in poly] + [poly[0][0]]
        wy = [v[1] for v in poly] + [poly[0][1]]
        ax.plot(wx, wy, "-", color=colors[i % len(colors)],
                linewidth=2, label=f"Wall {i} ({len(poly)} pts)")

    # Nav nodes
    for nid, (nx, ny) in nodes.items():
        ax.plot(nx, ny, "go", markersize=4)
    # Nav edges
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]],
                [nodes[a][1], nodes[b][1]],
                "g-", linewidth=0.8, alpha=0.5)

    # Start
    ax.plot(start[0], start[1], "r*", markersize=15, label="Start")

    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("DXF Hospital Map (metres, centred)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("dxf_map_processed.png", dpi=150)
    print("\nSaved dxf_map_processed.png")

    # Print MuJoCo XML snippet
    xml = generate_mujoco_wall_geoms(segs)
    print(f"\nMuJoCo wall geoms ({xml.count('geom')} geoms):")
    print(xml[:500], "...")
