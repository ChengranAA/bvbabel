"""Typed read/write for SSM (surface-to-surface mapping)."""
import sys
from bvbabel.ssm import SSM

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.ssm"

ssm = SSM.read(path)
print(f"{ssm.nr_vertices_1} → {ssm.nr_vertices_2} vertices  version={ssm.file_version}")
print(f"first indices: {ssm.data[:5]}")

new = SSM.create_default(nr_vertices=1000)
new.write("/tmp/example_out.ssm")
