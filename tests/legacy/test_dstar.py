"""
Quick test: verify D* Lite in HospitalGraph produces correct paths
and that update_edge_cost() triggers rerouting.
"""
import sys
sys.path.insert(0, r"C:\Users\vardh\Desktop\RAS")

from path_planner import HospitalGraph

def test_basic_routing():
    """D* Lite should find same optimal paths as A* did."""
    g = HospitalGraph()
    
    # Test 1: Same-node query
    path = g.dstar_lite("mc_W", "mc_W")
    assert path == ["mc_W"], f"Same-node failed: {path}"
    print("[PASS] Same-node query")
    
    # Test 2: Adjacent nodes
    path = g.dstar_lite("mc_W", "mc_CW")
    assert path == ["mc_W", "mc_CW"], f"Adjacent failed: {path}"
    print("[PASS] Adjacent nodes")
    
    # Test 3: Corridor traversal
    path = g.dstar_lite("mc_W", "mc_E")
    assert path[0] == "mc_W" and path[-1] == "mc_E"
    assert len(path) == 5  # mc_W -> mc_CW -> mc_C -> mc_CE -> mc_E
    print(f"[PASS] Corridor traversal: {' -> '.join(path)}")
    
    # Test 4: Cross-zone routing (corridor to room)
    path = g.dstar_lite("mc_W", "room_NW_S")
    assert path[0] == "mc_W" and path[-1] == "room_NW_S"
    assert "d_cNW_in" in path and "d_cNW_out" in path  # must go through door
    print(f"[PASS] Cross-zone route: {' -> '.join(path)}")
    
    # Test 5: Full route() API
    waypoints = g.route((-10.0, 0.0), (-6.0, 3.2))  # corridor -> room_NW_S
    assert len(waypoints) > 0
    assert waypoints[-1] == (-6.0, 3.2)  # user goal is appended
    print(f"[PASS] route() API: {len(waypoints)} waypoints")
    
    # Test 6: Same-zone route (direct path, no D* Lite needed)
    waypoints = g.route((0.0, 0.0), (5.0, 0.0))  # both in corridor
    assert waypoints == [(5.0, 0.0)]
    print("[PASS] Same-zone direct path")

def test_incremental_replan():
    """update_edge_cost() should trigger rerouting."""
    g = HospitalGraph()
    
    # Normal path: mc_W -> mc_CW -> mc_C -> mc_CE -> mc_E
    path1 = g.dstar_lite("mc_W", "mc_E")
    assert "mc_CW" in path1
    print(f"[PASS] Original path: {' -> '.join(path1)}")
    
    # Block mc_W <-> mc_CW edge
    g2 = HospitalGraph()
    path_before = g2.dstar_lite("mc_W", "mc_E")
    g2.update_edge_cost("mc_W", "mc_CW", float("inf"))
    
    # Recompute — should find alternate route
    path2 = g2.dstar_lite("mc_W", "mc_E")
    # The path should still reach mc_E but avoid the blocked edge
    assert path2[0] == "mc_W" and path2[-1] == "mc_E"
    assert not ("mc_CW" in path2 and path2.index("mc_CW") == path2.index("mc_W") + 1), \
        f"Should not use blocked edge directly: {path2}"
    print(f"[PASS] Rerouted path: {' -> '.join(path2)}")

def test_path_costs():
    """Verify that D* Lite finds optimal-cost paths on the static graph."""
    import math
    g = HospitalGraph()
    
    # mc_W(-10,0) -> mc_CW(-6,0) -> mc_C(0,0)
    path = g.dstar_lite("mc_W", "mc_C")
    # Optimal is the straight corridor: mc_W -> mc_CW -> mc_C
    # Total distance = 4.0 + 6.0 = 10.0
    expected_cost = math.hypot(-6 - (-10), 0) + math.hypot(0 - (-6), 0)
    
    # Calculate actual path cost
    actual_cost = 0.0
    for i in range(len(path) - 1):
        n1, n2 = path[i], path[i+1]
        x1, y1 = g.NODES[n1]
        x2, y2 = g.NODES[n2]
        actual_cost += math.hypot(x2 - x1, y2 - y1)
    
    assert abs(actual_cost - expected_cost) < 0.01, \
        f"Cost mismatch: {actual_cost} vs {expected_cost}"
    print(f"[PASS] Path cost optimal: {actual_cost:.2f}m")

if __name__ == "__main__":
    print("=" * 60)
    print("D* Lite Verification Tests")
    print("=" * 60)
    
    test_basic_routing()
    print()
    test_incremental_replan()
    print()
    test_path_costs()
    
    print()
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
