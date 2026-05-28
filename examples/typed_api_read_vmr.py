"""Typed read/write for VMR (anatomical MRI)."""
import sys
from bvbabel.vmr import VMR

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.vmr"

vmr = VMR.read(path)
print(f"{vmr.dim_x}×{vmr.dim_y}×{vmr.dim_z}, dtype={vmr.data.dtype}, version={vmr.file_version}")
print(f"offset=({vmr.offset_x},{vmr.offset_y},{vmr.offset_z}), voxel=({vmr.voxel_size_x},{vmr.voxel_size_y},{vmr.voxel_size_z})")

# Header-only
meta = VMR.read(path, load_data=False)
print(f"header-only OK: data is None={meta.data is None}")

# Create + write
new = VMR.create_default(dim_x=64, dim_y=64, dim_z=32)
new.write("/tmp/example_out.vmr")
print("wrote /tmp/example_out.vmr")
