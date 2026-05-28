"""Typed read/write for POI (patches of interest)."""
import sys
from bvbabel.poi import POI

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.poi"

poi = POI.read(path)
print(f"{poi.nr_of_pois} patches  mesh={poi.from_mesh_file}  vertices={poi.nr_of_mesh_vertices}")
for p in poi.pois:
    verts = p.vertices[:5] if p.vertices is not None else []
    print(f"  {p.name}: {p.nr_of_vertices} vertices  first={verts}  color={p.color}")
