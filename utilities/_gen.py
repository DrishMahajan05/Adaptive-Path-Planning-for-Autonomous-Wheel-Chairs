import math
OUTER = [
    (19.3, 7.7), (23.9, 7.7), (27.2, 28.5), (37.1, 72.8),
    (28.3, 78.4), (22.3, 73.1), (4.5, 73.6), (18.1, 107.0),
    (32.2, 107.0), (32.1, 126.8), (51.5, 98.4), (61.7, 120.9),
    (39.5, 147.2), (83.0, 199.1), (92.5, 195.4), (86.1, 183.1),
    (92.9, 175.7)
]
INNER = [
    (25.0, 12.0), (29.0, 12.0), (33.0, 33.0), (42.0, 70.0),
    (34.0, 76.0), (27.0, 72.0), (10.0, 72.0), (22.0, 104.0),
    (37.0, 104.0), (37.0, 122.0), (48.0, 103.0), (57.0, 118.0),
    (44.0, 142.0), (79.0, 193.0), (86.0, 190.0), (81.0, 179.0),
    (87.0, 172.0)
]
def w(x1,y1,x2,y2,n,h=3.0,t=0.15):
    mx,my=(x1+x2)/2,(y1+y2)/2
    l=math.hypot(x2-x1,y2-y1)
    a=math.atan2(y2-y1,x2-x1)
    return f'    <geom name="{n}" type="box" pos="{mx:.2f} {my:.2f} {h/2:.1f}" size="{l/2:.2f} {t:.2f} {h/2:.1f}" euler="0 0 {a:.4f}" rgba="0.85 0.85 0.80 1" conaffinity="1"/>'

lines = []
for i in range(len(OUTER)-1):
    lines.append(w(*OUTER[i],*OUTER[i+1],f'ow{i}'))
lines.append("")
for i in range(len(INNER)-1):
    lines.append(w(*INNER[i],*INNER[i+1],f'iw{i}'))
lines.append("")
lines.append(w(*OUTER[0],*INNER[0],'cap_bot'))
lines.append(w(*OUTER[-1],*INNER[-1],'cap_top'))
lines.append("")
# Obstacles (angles in radians)
obs = [(58.8,148.5,6.9,4.5,47),(36.1,112.8,4.3,3.4,0),(24.6,130.6,1.7,1.6,0),(31.4,120.8,1.4,1.2,0),(38.6,120.8,1.2,1.0,0)]
for i,(cx,cy,hx,hy,ad) in enumerate(obs):
    ar = math.radians(ad)
    lines.append(f'    <geom name="sobs{i+1}" type="box" pos="{cx} {cy} 1.5" size="{hx} {hy} 1.5" euler="0 0 {ar:.4f}" rgba="0.70 0.25 0.25 1" conaffinity="1"/>')

for l in lines: print(l)
