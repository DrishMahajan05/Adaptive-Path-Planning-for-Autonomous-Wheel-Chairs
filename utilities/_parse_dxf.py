"""
Parse the DXF map file and extract polyline vertices.
"""
import matplotlib.pyplot as plt
import numpy as np

def parse_dxf_polylines(filepath):
    """Extract polyline vertices from a DXF file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split into lines, stripping whitespace
    lines = [l.strip() for l in content.split('\n')]
    
    polylines = []
    current_vertices = []
    in_polyline = False
    is_closed = False
    
    i = 0
    while i < len(lines):
        # Look for group code 0 followed by entity type
        if lines[i] == '0' and i + 1 < len(lines):
            entity = lines[i + 1]
            
            if entity == 'POLYLINE':
                in_polyline = True
                current_vertices = []
                is_closed = False
                i += 2
                # Read polyline attributes
                while i < len(lines) and not (lines[i] == '0'):
                    if lines[i] == '70' and i + 1 < len(lines):
                        flag = int(lines[i + 1])
                        is_closed = bool(flag & 1)
                        i += 2
                    else:
                        i += 2  # skip group code + value pairs
                continue
                
            elif entity == 'VERTEX' and in_polyline:
                x = None
                y = None
                bulge = 0.0
                i += 2
                # Read vertex attributes until next group code 0
                while i < len(lines):
                    if lines[i] == '0':
                        break
                    gc = lines[i]
                    if i + 1 < len(lines):
                        val = lines[i + 1]
                        if gc == '10':
                            x = float(val)
                        elif gc == '20':
                            y = float(val)
                        elif gc == '42':
                            bulge = float(val)
                    i += 2
                if x is not None and y is not None:
                    current_vertices.append((x, y, bulge))
                continue
                
            elif entity == 'SEQEND' and in_polyline:
                polylines.append({
                    'vertices': current_vertices,
                    'closed': is_closed
                })
                in_polyline = False
                i += 2
                continue
        
        i += 1
    
    return polylines

def main():
    polylines = parse_dxf_polylines('map_dxf.dxf')
    
    print(f"Found {len(polylines)} polylines")
    for idx, pl in enumerate(polylines):
        verts = pl['vertices']
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        print(f"\nPolyline {idx}: {len(verts)} vertices, closed={pl['closed']}")
        if verts:
            print(f"  X range: {min(xs):.2f} - {max(xs):.2f}")
            print(f"  Y range: {min(ys):.2f} - {max(ys):.2f}")
            print(f"  X span: {max(xs)-min(xs):.2f}")
            print(f"  Y span: {max(ys)-min(ys):.2f}")
            # Print first/last few vertices
            print(f"  First 3 vertices:")
            for v in verts[:3]:
                print(f"    ({v[0]:.2f}, {v[1]:.2f}) bulge={v[2]:.4f}")
            print(f"  Last 3 vertices:")
            for v in verts[-3:]:
                print(f"    ({v[0]:.2f}, {v[1]:.2f}) bulge={v[2]:.4f}")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 16))
    colors = ['blue', 'red', 'green', 'orange']
    
    for idx, pl in enumerate(polylines):
        verts = pl['vertices']
        if not verts:
            continue
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        if pl['closed']:
            xs.append(xs[0])
            ys.append(ys[0])
        ax.plot(xs, ys, '-', color=colors[idx % len(colors)], 
                linewidth=1.5, label=f'Polyline {idx} ({len(verts)} pts)')
    
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('DXF Hospital Map')
    ax.set_xlabel('X (DXF units)')
    ax.set_ylabel('Y (DXF units)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dxf_map_plot.png', dpi=150)
    print("\nSaved plot to dxf_map_plot.png")
    
    # Stats
    all_xs = []
    all_ys = []
    for pl in polylines:
        for v in pl['vertices']:
            all_xs.append(v[0])
            all_ys.append(v[1])
    
    if all_xs:
        print(f"\nOverall X range: {min(all_xs):.2f} - {max(all_xs):.2f} (span: {max(all_xs)-min(all_xs):.2f})")
        print(f"Overall Y range: {min(all_ys):.2f} - {max(all_ys):.2f} (span: {max(all_ys)-min(all_ys):.2f})")
        
        # Try different scale interpretations
        for label, scale in [("mm (0.001 m/unit)", 0.001), 
                             ("cm (0.01 m/unit)", 0.01),
                             ("10cm (0.1 m/unit)", 0.1)]:
            xspan = (max(all_xs)-min(all_xs)) * scale
            yspan = (max(all_ys)-min(all_ys)) * scale
            print(f"\n  If {label}:")
            print(f"    X span: {xspan:.2f} m")
            print(f"    Y span: {yspan:.2f} m")

if __name__ == '__main__':
    main()
