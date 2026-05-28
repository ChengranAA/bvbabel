"""Typed read/write for VMP (volumetric statistical map)."""
import sys
from bvbabel.vmp import VMP

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.vmp"

vmp = VMP.read(path)
print(f"{vmp.nr_of_sub_maps} maps, data={vmp.data.shape}")
for i, m in enumerate(vmp.maps):
    print(f"  [{i}] {m.map_name}  type={m.type_of_map}  threshold={m.map_threshold}  df=({m.df1},{m.df2})")

# Header-only
meta = VMP.read(path, load_data=False)
print(f"header-only: data is None={meta.data is None}")

# Create + write
new = VMP.create_default(dim_x=64, dim_y=64, dim_z=32, n_maps=2)
new.maps[0].map_name = "t-map"
new.maps[1].map_name = "F-map"
new.write("/tmp/example_out.vmp")
