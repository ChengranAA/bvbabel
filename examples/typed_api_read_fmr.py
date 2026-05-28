"""Typed read/write for FMR (functional MR)."""
import sys
from bvbabel.fmr import FMR

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.fmr"

fmr = FMR.read(path)
print(f"{fmr.nr_of_volumes} vols × {fmr.nr_of_slices} slices × {fmr.resolution_x}×{fmr.resolution_y}")
print(f"data={fmr.data.shape}  dtype={fmr.data.dtype}  data_type={fmr.data_type}")
print(f"TR={fmr.tr}  TE={fmr.te}  slice_order={fmr.slice_acquisition_order}")

# Position sub-object
p = fmr.position
print(f"position: slice1=({p.slice_1_center_x},{p.slice_1_center_y},{p.slice_1_center_z})")
print(f"  row_dir=({p.row_dir_x},{p.row_dir_y},{p.row_dir_z})")
print(f"  col_dir=({p.col_dir_x},{p.col_dir_y},{p.col_dir_z})")

# Header-only
meta = FMR.read(path, load_data=False)
print(f"header-only: data is None={meta.data is None}")

new = FMR.create_default(nr_volumes=100, nr_slices=16, res_x=64, res_y=64)
new.write("/tmp/example_out.fmr")
