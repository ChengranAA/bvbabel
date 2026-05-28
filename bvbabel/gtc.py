"""Read, write, create BrainVoyager GTC (depth-grid time course) file format.

Typed API
---------
    gtc = GTC.read("data.gtc")
    print(gtc.dim_d, gtc.data.shape)
    gtc.write("output.gtc")

Procedural API (backward-compatible)
------------------------------------
    header, data = bvbabel.gtc.read_gtc("data.gtc")
    bvbabel.gtc.write_gtc("output.gtc", header, data)
"""

import numpy as np
from bvbabel._binary_format import (
    Field, DataField, BinaryFormat, register_format,
)


@register_format(".gtc")
class GTC(BinaryFormat):
    """Typed BrainVoyager GTC (depth-grid time course).

    Data is 4-D ``(X, Y, D, T)`` after axis rearrangement.
    """

    file_version = Field("<i")
    dim_d = Field("<i")
    dim_x = Field("<i")
    dim_y = Field("<i")
    dim_t = Field("<i")

    data = DataField(
        dtype="<i",
        shape_fields=("dim_d", "dim_y", "dim_x", "dim_t"),
    )

    # -- Hooks: rearrange axes (D, Y, X, T) → (X, Y, D, T) ---------------

    def _post_read(self):
        if self.data is None:
            return
        # On-disk: (D, Y, X, T) → logical: (X, Y, D, T)
        self.data = np.transpose(self.data, (2, 1, 0, 3))

    def _pre_write(self):
        if self.data is None:
            return
        # Logical: (X, Y, D, T) → on-disk: (D, Y, X, T)
        self.data = np.transpose(self.data, (2, 1, 0, 3))

    # -- Factory ---------------------------------------------------------

    @classmethod
    def create_default(cls, dim_d=10, dim_x=64, dim_y=64, dim_t=100):
        gtc = cls()
        gtc.file_version = 1
        gtc.dim_d = dim_d
        gtc.dim_x = dim_x
        gtc.dim_y = dim_y
        gtc.dim_t = dim_t
        # Logical shape: (X, Y, D, T)
        gtc.data = np.random.randint(
            0, 1000, size=(dim_x, dim_y, dim_d, dim_t), dtype=np.int32
        )
        return gtc

    # -- Legacy key mapping ----------------------------------------------

    _LEGACY_MAP = {
        "file_version": "File version",
        "dim_d": "DimD",
        "dim_x": "DimX",
        "dim_y": "DimY",
        "dim_t": "DimT",
    }
    _LEGACY_REVERSE = {v: k for k, v in _LEGACY_MAP.items()}

    def to_legacy_dict(self):
        return {l: getattr(self, p) for p, l in self._LEGACY_MAP.items()}

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
# Backward-compatible shims
# =============================================================================

def read_gtc(filename):
    gtc = GTC.read(filename)
    return gtc.to_legacy_dict(), gtc.data


def write_gtc(filename, header, data_img):
    gtc = GTC.from_legacy_dict(header, data=data_img)
    gtc.write(filename)
