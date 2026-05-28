"""Typed read/write for VTC (volume time course)."""
import sys
from bvbabel.vtc import VTC

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.vtc"

vtc = VTC.read(path)
print(f"{vtc.nr_time_points} TP, data={vtc.data.shape}, dtype={vtc.data.dtype}, TR={vtc.tr_ms}ms")
print(f"source FMR: {vtc.source_fmr_name}")

new = VTC.create_default(dim_x=40, dim_y=30, dim_z=20, nr_time_points=10)
new.write("/tmp/example_out.vtc")
