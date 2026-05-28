"""Typed read/write for V16 (16-bit anatomical)."""
import sys, numpy as np
from bvbabel.v16 import V16

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.v16"

v16 = V16.read(path)
print(f"{v16.dim_x}×{v16.dim_y}×{v16.dim_z}, dtype={v16.data.dtype}, range=[{v16.data.min()},{v16.data.max()}]")

new = V16.create_default(dim_x=64, dim_y=64, dim_z=32)
new.write("/tmp/example_out.v16")
