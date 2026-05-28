"""Typed read/write for MSK (volume mask)."""
import sys, numpy as np
from bvbabel.msk import MSK

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.msk"

msk = MSK.read(path)
print(f"computed dims: {msk.dim_x}×{msk.dim_y}×{msk.dim_z}, bounds: x[{msk.x_start}:{msk.x_end}] y[{msk.y_start}:{msk.y_end}] z[{msk.z_start}:{msk.z_end}]")

# Create + write
m = MSK()
m.vtc_resolution = 1
m.x_start = 0; m.x_end = 32
m.y_start = 0; m.y_end = 32
m.z_start = 0; m.z_end = 16
m.data = np.random.randint(0, 2, (16, 32, 32), dtype=np.uint8)
m.write("/tmp/example_out.msk")
