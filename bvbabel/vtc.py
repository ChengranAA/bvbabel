"""Read, write, create BrainVoyager VTC (volume time course) file format.

Typed API
---------
    vtc = VTC.read("data.vtc")
    print(vtc.nr_time_points, vtc.data.shape, vtc.data.dtype)
    vtc.write("output.vtc")

Procedural API (backward-compatible)
------------------------------------
    header, data = bvbabel.vtc.read_vtc("data.vtc")
    bvbabel.vtc.write_vtc("output.vtc", header, data)
    header, data = bvbabel.vtc.create_vtc()
"""

import struct
import numpy as np
from bvbabel._binary_format import (
    Field, StringField, DataField, BinaryFormat, register_format,
)
from bvbabel.utils import read_variable_length_string, write_variable_length_string


# =============================================================================
# Typed VTC format
# =============================================================================

@register_format(".vtc")
class VTC(BinaryFormat):
    """Typed BrainVoyager VTC (4-D volume time course).

    Data is always 4-D ``(Z, X, Y, T)`` in RAS-like axis order.  The
    on-disk BV layout ``(Z, Y, X, T)`` is transposed and flipped on
    read.
    """

    # -- Header ----------------------------------------------------------
    file_version = Field("<h")
    source_fmr_name = StringField()

    protocol_attached = Field("<h")

    # Conditional: only present when protocol_attached > 0
    protocol_name = StringField(
        condition=lambda s: s.protocol_attached > 0
    )

    current_protocol_index = Field("<h")
    data_type = Field("<h")   # 1 = int16, 2 = float32
    nr_time_points = Field("<h")
    vtc_resolution = Field("<h")

    x_start = Field("<h")
    x_end = Field("<h")
    y_start = Field("<h")
    y_end = Field("<h")
    z_start = Field("<h")
    z_end = Field("<h")

    lr_convention = Field("<B")
    reference_space = Field("<B")
    tr_ms = Field("<f")

    # -- Data (raw bytes; dtype resolved in _post_read) ------------------
    data = DataField(dtype="<u1", shape_fields=("_data_byte_count",))

    # -- Computed --------------------------------------------------------

    @property
    def dim_x(self):
        return (self.x_end - self.x_start) // self.vtc_resolution

    @property
    def dim_y(self):
        return (self.y_end - self.y_start) // self.vtc_resolution

    @property
    def dim_z(self):
        return (self.z_end - self.z_start) // self.vtc_resolution

    @property
    def _itemsize(self):
        return 2 if self.data_type == 1 else 4

    @property
    def _data_byte_count(self):
        return (self.dim_z * self.dim_y * self.dim_x
                * self.nr_time_points * self._itemsize)

    # -- Hooks -----------------------------------------------------------

    def _post_read(self):
        if self.data is None:
            return
        dt = np.dtype("<h") if self.data_type == 1 else np.dtype("<f")
        self.data = np.frombuffer(self.data.tobytes(), dtype=dt)
        z, y, x, t = self.dim_z, self.dim_y, self.dim_x, self.nr_time_points
        self.data = self.data.reshape(z, y, x, t)
        # BV (Z, Y, X, T) → RAS-like (Z, X, Y, T)
        self.data = np.transpose(self.data, (0, 2, 1, 3))
        self.data = self.data[::-1, ::-1, ::-1, :]

    def _pre_write(self):
        if self.data is None:
            return
        data = self.data[::-1, ::-1, ::-1, :]
        data = np.transpose(data, (0, 2, 1, 3))
        self.data = data.ravel().view("<u1")

    # -- Factory ---------------------------------------------------------

    @classmethod
    def create_default(cls, dim_x=120, dim_y=100, dim_z=80, nr_time_points=10):
        vtc = cls()
        vtc.file_version = 3
        vtc.source_fmr_name = ""
        vtc.protocol_attached = 0
        vtc.current_protocol_index = 0
        vtc.data_type = 1          # int16
        vtc.nr_time_points = nr_time_points
        vtc.vtc_resolution = 1
        vtc.x_start = 90
        vtc.x_end = 90 + dim_x
        vtc.y_start = 100
        vtc.y_end = 100 + dim_y
        vtc.z_start = 110
        vtc.z_end = 110 + dim_z
        vtc.lr_convention = 1
        vtc.reference_space = 1
        vtc.tr_ms = 1000.0
        # Logical shape: (Z, X, Y, T)
        vtc.data = (np.random.random(
            (dim_z, dim_x, dim_y, nr_time_points)) * 225
        ).astype(np.int16)
        return vtc

    # -- Legacy key mapping ----------------------------------------------

    _LEGACY_MAP = {
        "file_version": "File version",
        "source_fmr_name": "Source FMR name",
        "protocol_attached": "Protocol attached",
        "protocol_name": "Protocol name",
        "current_protocol_index": "Current protocol index",
        "data_type": "Data type (1:short int, 2:float)",
        "nr_time_points": "Nr time points",
        "vtc_resolution": "VTC resolution relative to VMR (1, 2, or 3)",
        "x_start": "XStart",
        "x_end": "XEnd",
        "y_start": "YStart",
        "y_end": "YEnd",
        "z_start": "ZStart",
        "z_end": "ZEnd",
        "lr_convention": "L-R convention (0:unknown, 1:radiological, 2:neurological)",
        "reference_space": "Reference space (0:unknown, 1:native, 2:ACPC, 3:Tal, 4:MNI)",
        "tr_ms": "TR (ms)",
    }
    _LEGACY_REVERSE = {v: k for k, v in _LEGACY_MAP.items()}

    def to_legacy_dict(self):
        result = {}
        for py_name, legacy_name in self._LEGACY_MAP.items():
            val = getattr(self, py_name)
            # Only include protocol_name if it was actually present
            if py_name == "protocol_name" and self.protocol_attached == 0:
                result[legacy_name] = ""
            else:
                result[legacy_name] = val
        return result

    @classmethod
    def from_legacy_dict(cls, d, data=None):
        kwargs = {}
        for legacy_name, py_name in cls._LEGACY_REVERSE.items():
            if legacy_name in d:
                kwargs[py_name] = d[legacy_name]
        # protocol_name: always set from dict (even empty string)
        if "Protocol name" in d:
            kwargs["protocol_name"] = d["Protocol name"]
        instance = cls(**kwargs)
        if data is not None:
            instance.data = data
        return instance


# =============================================================================
# Backward-compatible shims
# =============================================================================

def read_vtc(filename, rearrange_data_axes=True):
    """Read BrainVoyager VTC file (legacy API).

    The *rearrange_data_axes* flag is accepted for backward compatibility
    but has no effect — the typed API always returns RAS-like axes.
    """
    vtc = VTC.read(filename)
    return vtc.to_legacy_dict(), vtc.data


def write_vtc(filename, header, data_img, rearrange_data_axes=True):
    """Write BrainVoyager VTC file (legacy API).

    The *rearrange_data_axes* flag is accepted for backward compatibility
    but the data is always assumed to be in RAS-like ``(Z, X, Y, T)``
    layout (the output of ``read_vtc``).
    """
    vtc = VTC.from_legacy_dict(header, data=data_img)
    vtc.write(filename)


def create_vtc(rearrange_data_axes=True):
    """Create BrainVoyager VTC file with default values (legacy API)."""
    vtc = VTC.create_default()
    return vtc.to_legacy_dict(), vtc.data
