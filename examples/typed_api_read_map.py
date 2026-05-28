"""Typed read/write for MAP (statistical map stack)."""
import sys
from bvbabel.map import MAP

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.map"

mp = MAP.read(path)
print(f"{mp.nr_of_slices} slices  {mp.dim_x}×{mp.dim_y}  data={mp.data.shape}  version={mp.file_version}")
print(f"map_type_code={mp.map_type_code}  cluster={mp.cluster_size}")
print(f"threshold=[{mp.stat_threshold_min}, {mp.stat_threshold_max}]  df=({mp.df1},{mp.df2})")

new = MAP.create_default(dim_x=64, dim_y=64, nr_slices=12)
new.write("/tmp/example_out.map")
