"""Typed read/write for GTC (depth-grid time course)."""
import sys
from bvbabel.gtc import GTC

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.gtc"

gtc = GTC.read(path)
print(f"D={gtc.dim_d} X={gtc.dim_x} Y={gtc.dim_y} T={gtc.dim_t}  data={gtc.data.shape}")

new = GTC.create_default(dim_d=5, dim_x=32, dim_y=32, dim_t=50)
new.write("/tmp/example_out.gtc")
