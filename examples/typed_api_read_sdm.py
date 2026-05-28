"""Typed read/write for SDM (design matrix)."""
import sys
from bvbabel.sdm import SDM

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.sdm"

sdm = SDM.read(path)
print(f"{sdm.nr_of_predictors} predictors × {sdm.nr_of_data_points} points")
for p in sdm.predictors:
    print(f"  {p.name}: color={p.color}  range=[{p.values.min():.3f}, {p.values.max():.3f}]")

new = SDM.create_default(nr_predictors=3, nr_data_points=100)
new.write("/tmp/example_out.sdm")
