from wheelchair_model import WheelchairPhysics
p = WheelchairPhysics()
s = p.get_state()
print(f"Model loaded OK. Start pos: ({s['x']:.1f}, {s['y']:.1f})")
print(f"Num geoms: {p.model.ngeom}")
