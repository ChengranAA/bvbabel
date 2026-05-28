"""Typed read/write for TRF (transformation matrix)."""
import sys
from bvbabel.trf import TRF

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.trf"

trf = TRF.read(path)
print(f"type={trf.transformation_type}  version={trf.file_version}  coord_sys={trf.coordinate_system}")
print(f"source={trf.source_file}")
print(f"target={trf.target_file}")
if trf.matrix is not None:
    print(f"matrix:\n{trf.matrix}")

new = TRF.create_default()
new.write("/tmp/example_out.trf")
