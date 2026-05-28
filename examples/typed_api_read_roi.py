"""Typed read/write for ROI (regions of interest)."""
import sys
from bvbabel.roi import ROI

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.roi"

roi = ROI.read(path)
print(f"{roi.nr_of_rois} regions")
for r in roi.rois:
    coords = r.coordinates[:3] if r.coordinates is not None else []
    print(f"  {r.nr_of_voxels} voxels  slice={r.from_slice}  rect=[{r.left},{r.right},{r.top},{r.bottom}]  first={coords}")
