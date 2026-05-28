"""Read, write, create BrainVoyager MAP (statistical map) file format.

MAP is a binary format with interspersed slice markers: each 2-D float32
slice is preceded by a short-int slice index.  The typed API overrides
``read()`` / ``write()`` to handle this non-contiguous data layout.

Typed API
---------
    mp = MAP.read("stats.map")
    print(mp.nr_of_slices, mp.data.shape)
    mp.write("output.map")
"""

import struct
import numpy as np
from bvbabel._binary_format import (
    Field, StringField, DataField, BinaryFormat, register_format,
)
from bvbabel.utils import read_variable_length_string, write_variable_length_string


# =============================================================================
# MAP format
# =============================================================================

@register_format(".map")
class MAP(BinaryFormat):
    """Typed BrainVoyager MAP (2-D statistical map stack).

    Data is 3-D ``(Y, X, Z)`` after BV→Tal axis conversion.
    """

    # -- Header fields ---------------------------------------------------
    map_type_code = Field("<h")   # also serves as nr_of_slices
    nr_of_maps = Field("<h")
    dim_x = Field("<h")
    dim_y = Field("<h")
    cluster_size = Field("<h")

    stat_threshold_min = Field("<f")
    stat_threshold_max = Field("<f")

    # Conditional: only when map_type_code == 3 (cross-correlation)
    nr_of_lags = Field(
        "<h", condition=lambda s: s.map_type_code == 3
    )

    _reserved = Field("<h")         # always 9999
    file_version = Field("<h")

    df1 = Field("<i")
    df2 = Field("<i")

    rtc_name = StringField()

    # -- Data (populated manually in read/write) -------------------------
    data = DataField(dtype="<f", shape_fields=())

    # -- Derived ---------------------------------------------------------

    @property
    def nr_of_slices(self):
        return self.map_type_code

    # -- I/O override ----------------------------------------------------

    @classmethod
    def read(cls, filename, load_data=True):
        instance = cls()
        with open(filename, "rb") as f:
            # Header fields (standard binary iteration)
            for field in cls._fields:
                if field.name == "data":
                    break  # stop before data — we handle it manually
                field.read(f, instance, load_data=load_data)

            if not load_data:
                # Skip data: each slice has 2-byte index + float32 data
                slice_bytes = (2 + instance.dim_y * instance.dim_x * 4)
                total = instance.nr_of_slices * slice_bytes
                f.seek(total, 1)
                instance._values["data"] = None
                return instance

            # Read slice-structured data
            nr_slices = instance.nr_of_slices
            slices = []
            for _ in range(nr_slices):
                slice_idx, = struct.unpack("<h", f.read(2))
                slc = np.fromfile(
                    f, dtype="<f", count=instance.dim_y * instance.dim_x
                )
                slc = slc.reshape(instance.dim_y, instance.dim_x)
                slices.append(slc)

        data = np.stack(slices, axis=-1)  # (Y, X, Z)
        # BV → Tal: transpose + flip
        data = np.transpose(data, (1, 0, 2))  # (X, Y, Z)
        data = data[::-1, ::-1, :]            # flip X, Y
        instance._values["data"] = data
        return instance

    def write(self, filename):
        with open(filename, "wb") as f:
            for field in self._fields:
                if field.name == "data":
                    break
                field.write(f, self)

            # Write slice-structured data
            data = self._values.get("data")
            if data is not None:
                # Tal → BV: flip + transpose back
                data = data[::-1, ::-1, :]
                data = np.transpose(data, (1, 0, 2))  # (Y, X, Z)
                nr_slices = data.shape[-1]
                for s in range(nr_slices):
                    f.write(struct.pack("<h", s))
                    f.write(
                        data[:, :, s].astype("<f").tobytes(order="C")
                    )

    # -- Factory ---------------------------------------------------------

    @classmethod
    def create_default(cls, dim_x=80, dim_y=80, nr_slices=16):
        mp = cls()
        mp.map_type_code = nr_slices  # doubles as nr_of_slices for type 1
        mp.nr_of_maps = nr_slices
        mp.dim_x = dim_x
        mp.dim_y = dim_y
        mp.cluster_size = 8
        mp.stat_threshold_min = 0.5
        mp.stat_threshold_max = 1.0
        mp._reserved = 9999
        mp.file_version = 3
        mp.df1 = 0
        mp.df2 = 0
        mp.rtc_name = ""
        mp.data = (
            np.random.random((dim_y, dim_x, nr_slices)) * 2 - 1
        ).astype(np.float32)
        return mp

    # -- Legacy key mapping ----------------------------------------------

    _LEGACY_MAP = {
        "map_type_code": "NrOfSlices",
        "nr_of_maps": "NrOfMaps",
        "dim_x": "DimX",
        "dim_y": "DimY",
        "cluster_size": "ClusterSize",
        "stat_threshold_min": "Min",
        "stat_threshold_max": "Max",
        "nr_of_lags": "NrOfLags",
        "_reserved": "Reserved",
        "file_version": "FileVersion",
        "df1": "df1",
        "df2": "df2",
        "rtc_name": "RTCName",
    }
    _LEGACY_REVERSE = {v: k for k, v in _LEGACY_MAP.items()}

    def to_legacy_dict(self):
        result = {}
        for py_name, legacy_name in self._LEGACY_MAP.items():
            result[legacy_name] = getattr(self, py_name)
        result["MapType"] = "t-values"  # original hardcodes this
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
# Backward-compatible shims
# =============================================================================

def read_map(filename):
    mp = MAP.read(filename)
    return mp.to_legacy_dict(), mp.data


def write_map(filename, header, data_img):
    mp = MAP.from_legacy_dict(header, data=data_img)
    mp.write(filename)
