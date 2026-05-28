"""Typed read/write for PRT (stimulation protocol)."""
import sys
from bvbabel.prt import PRT

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.prt"

prt = PRT.read(path)
print(f"{prt.nr_of_conditions} conditions  resolution={prt.resolution_of_time}")
for c in prt.conditions:
    onset = c.time_start[0] if c.nr_of_occurrences > 0 else "N/A"
    print(f"  {c.name}: {c.nr_of_occurrences} occurrences  first onset={onset}  color={c.color}")
