"""Typed read/write for VOI (voxels of interest)."""
import sys
from bvbabel.voi import VOI

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.voi"

voi = VOI.read(path)
print(f"{voi.nr_of_vois} VOIs  space={voi.reference_space}")
for vv in voi.vois:
    coords = vv.coordinates[:3] if vv.coordinates is not None else []
    print(f"  {vv.name}: {vv.nr_of_voxels} voxels  first={coords}  color={vv.color}")
