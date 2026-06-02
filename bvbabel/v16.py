"""Read, write, create BrainVoyager V16 (16-bit anatomical) file format.

Provides a typed-object API (``V16`` class) and backward-compatible
procedural shims (``read_v16``, ``write_v16``, ``create_v16``).

Typed API
---------
    v16 = V16.read("anatomy.v16")
    print(v16.dim_x, v16.data.shape, v16.data.dtype)
    v16.write("output.v16")

Procedural API (backward-compatible)
------------------------------------
    header, data = bvbabel.v16.read_v16("anatomy.v16")
    bvbabel.v16.write_v16("output.v16", header, data)
    header, data = bvbabel.v16.create_v16()
"""

import numpy as np
from bvbabel._binary_format import (
    Field, DataField, BinaryFormat, register_format
)



# ---------------------------------------------------------------------------
# Typed V16 format
# ---------------------------------------------------------------------------

@register_format(".v16")
class V16(BinaryFormat):
    """Typed BrainVoyager V16 (16-bit unsigned anatomical dataset).

    Unlike VMR, V16 has no file-version field, no conditional fields,
    and no post-data header — just three dimension fields followed by
    the raw uint16 voxel data.
    """

    dim_x = Field("<H")
    dim_y = Field("<H")
    dim_z = Field("<H")

    data = DataField(
        dtype="<H",
        shape_fields=("dim_z", "dim_y", "dim_x"),
    )

    # -- Factory ---------------------------------------------------------

    @classmethod
    def create_default(cls, dim_x=256, dim_y=256, dim_z=256):
        v16 = cls()
        v16.dim_x = dim_x
        v16.dim_y = dim_y
        v16.dim_z = dim_z
        shape = (dim_z, dim_y, dim_x)
        v16.data = np.random.randint(
            0, 65535, size=shape, dtype=np.uint16
        )
        return v16

    # -- Legacy key mapping ----------------------------------------------

    _LEGACY_MAP = {
        "dim_x": "DimX",
        "dim_y": "DimY",
        "dim_z": "DimZ",
    }
    _LEGACY_REVERSE = {v: k for k, v in _LEGACY_MAP.items()}

    def to_legacy_dict(self):
        return {
            legacy: getattr(self, py_name)
            for py_name, legacy in self._LEGACY_MAP.items()
        }

    @classmethod
    def from_legacy_dict(cls, d, data=None):
        kwargs = {}
        for legacy_name, py_name in cls._LEGACY_REVERSE.items():
            if legacy_name in d:
                kwargs[py_name] = d[legacy_name]
        instance = cls(**kwargs)
        if data is not None:
            instance.data = data
        return instance


# =============================================================================
# Backward-compatible procedural shims
# =============================================================================

def read_v16(filename):
    """Read BrainVoyager V16 file (legacy API)."""
    v16 = V16.read(filename)
    return v16.to_legacy_dict(), v16.data


def write_v16(filename, header, data_img):
    """Write BrainVoyager V16 file (legacy API)."""
    v16 = V16.from_legacy_dict(header, data=data_img)
    v16.write(filename)
    print("V16 saved.")


def create_v16():
    """Create BrainVoyager V16 file with default values (legacy API)."""
    v16 = V16.create_default()
    return v16.to_legacy_dict(), v16.data
