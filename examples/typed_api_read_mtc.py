"""Typed read/write for MTC (mesh time course)."""
import sys
from bvbabel.mtc import MTC

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.mtc"

mtc = MTC.read(path)
print(f"{mtc.nr_vertices} vertices × {mtc.nr_time_points} TP  data={mtc.data.shape}")
print(f"TR={mtc.tr}  delta={mtc.delta}  tau={mtc.tau}  delay={mtc.hemodynamic_delay}")

new = MTC.create_default(nr_vertices=100, nr_time_points=50)
new.write("/tmp/example_out.mtc")
