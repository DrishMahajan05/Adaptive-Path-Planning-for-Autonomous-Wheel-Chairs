import sys
import re

file_path = r'c:\Users\vardh\Desktop\RAS\path_planner.py'
with open(file_path, 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Remove D* state
state_pattern = r'        # --- D\* Lite persistent state ---+.*?self\._goal:\s*Optional\[str\] = None\n'
code = re.sub(state_pattern, '', code, flags=re.DOTALL)

# 2. Replace D* Lite methods with astar
dstar_pattern = r'    # -- D\* Lite ---+.*?def dstar_lite.*?return path\n'
astar_code = '''    # -- A* Search -------------------------------------------------------------

    def astar(self, src: str, dst: str) -> List[str]:
        if src == dst:
            return [src]

        open_set = []
        heapq.heappush(open_set, (0.0, src))
        came_from: Dict[str, str] = {}
        
        g_score: Dict[str, float] = {n: float("inf") for n in self.NODES}
        g_score[src] = 0.0
        
        f_score: Dict[str, float] = {n: float("inf") for n in self.NODES}
        f_score[src] = self._dist(src, dst)

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == dst:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor, cost in self._adj.get(current, []):
                tentative_g = g_score[current] + cost
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._dist(neighbor, dst)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return [src, dst]
'''
code = re.sub(dstar_pattern, astar_code, code, flags=re.DOTALL)

# 3. Update update_edge_cost
edge_cost_pattern = r'    def update_edge_cost\(self, a: str, b: str,.*?self\._compute_shortest_path\(\)\n'
edge_cost_code = '''    def update_edge_cost(self, a: str, b: str,
                         new_cost: float) -> None:
        """
        Dynamically update the cost of edge (a, b)
        """
        for i, (nb, _old_cost) in enumerate(self._adj.get(a, [])):
            if nb == b:
                self._adj[a][i] = (b, new_cost)
                break
        for i, (nb, _old_cost) in enumerate(self._adj.get(b, [])):
            if nb == a:
                self._adj[b][i] = (a, new_cost)
                break
'''
code = re.sub(edge_cost_pattern, edge_cost_code, code, flags=re.DOTALL)

# 4. Replace string references
code = code.replace('node_path = self.dstar_lite(src_node, dst_node)', 'node_path = self.astar(src_node, dst_node)')
code = code.replace('runs D* Lite through', 'runs A* through')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(code)

print("Replacement done.")
