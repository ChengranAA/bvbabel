"""Read, write, create BrainVoyager MSK (mask) file format.

Provides a typed-object API (``MSK`` class) and backward-compatible
procedural shims (``read_msk``, ``write_msk``).

Typed API
---------
    msk = MSK.read("mask.msk")
    print(msk.x_start, msk.x_end, msk.data.shape)
    msk.write("output.msk")

Procedural API (backward-compatible)
------------------------------------
    header, data = bvbabel.msk.read_msk("mask.msk")
    bvbabel.msk.write_msk("output.msk", header, data)
"""

import numpy as np
from bvbabel._binary_format import (
    Field, DataField, BinaryFormat, register_format
)


# ---------------------------------------------------------------------------
# Axis transforms
# ---------------------------------------------------------------------------

def _bv_to_tal(data):
    data = np.transpose(data, (0, 2, 1))
    data = data[::-1, ::-1, ::-1]
    return data


_bv_to_tal_inv = _bv_to_tal


# ---------------------------------------------------------------------------
# Typed MSK format
# ---------------------------------------------------------------------------

@register_format(".msk")
class MSK(BinaryFormat):
    """Typed BrainVoyager MSK (volume mask)."""

    vtc_resolution = Field("<h", default=1)
    x_start = Field("<h")
    x_end = Field("<h")
    y_start = Field("<h")
    y_end = Field("<h")
    z_start = Field("<h")
    z_end = Field("<h")

    data = DataField(
        dtype="<B",
        shape_fields=("dim_z", "dim_y", "dim_x"),
        transform=_bv_to_tal,
        inverse_transform=_bv_to_tal_inv,
    )

    # -- Derived properties ----------------------------------------------

    @property
    def dim_x(self):
        return (self.x_end - self.x_start) // self.vtc_resolution

    @property
    def dim_y(self):
        return (self.y_end - self.y_start) // self.vtc_resolution

    @property
    def dim_z(self):
        return (self.z_end - self.z_start) // self.vtc_resolution

    # -- Legacy key mapping ----------------------------------------------

    _LEGACY_MAP = {
        "vtc_resolution": "VTC resolution relative to VMR (1, 2, or 3)",
        "x_start": "XStart",
        "x_end": "XEnd",
        "y_start": "YStart",
        "y_end": "YEnd",
        "z_start": "ZStart",
        "z_end": "ZEnd",
    }
    _LEGACY_REVERSE = {v: k for k, v in _LEGACY_MAP.items()}

    def to_legacy_dict(self):
        result = {}
        for py_name, legacy_name in self._LEGACY_MAP.items():
            result[legacy_name] = getattr(self, py_name)
        return result

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

def read_msk(filename):
    """Read BrainVoyager MSK file (legacy API)."""
    msk = MSK.read(filename)
    return msk.to_legacy_dict(), msk.data


def write_msk(filename, header, data_img):
    """Write BrainVoyager MSK file (legacy API)."""
    msk = MSK.from_legacy_dict(header, data=data_img)
    msk.write(filename)
