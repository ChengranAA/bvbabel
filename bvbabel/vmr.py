"""Read, write, create BrainVoyager VMR file format.

Provides both a modern typed-object API (``VMR`` class) and backward-
compatible procedural shims (``read_vmr``, ``write_vmr``, ``create_vmr``).

Typed API
---------
    vmr = VMR.read("subject01.vmr")
    print(vmr.dim_x, vmr.data.shape)
    vmr.write("output.vmr")

Procedural API (backward-compatible)
------------------------------------
    header, data = bvbabel.vmr.read_vmr("subject01.vmr")
    bvbabel.vmr.write_vmr("output.vmr", header, data)
    header, data = bvbabel.vmr.create_vmr()
"""

import struct
import numpy as np
from bvbabel._binary_format import (
    Field, DataField, SubRecordListField, BinaryFormat, register_format
)
from bvbabel.utils import (
    read_variable_length_string, write_variable_length_string
)


# ---------------------------------------------------------------------------
# Axis transforms (BV internal ↔ Talairach / NIfTI RAS-like)
# ---------------------------------------------------------------------------

def _bv_to_tal(data):
    """Convert on-disk BV axis layout to logical (Talairach-like) layout."""
    data = np.transpose(data, (0, 2, 1))
    data = data[::-1, ::-1, ::-1]
    return data


_bv_to_tal_inv = _bv_to_tal  # transpose + flip is its own inverse


# ---------------------------------------------------------------------------
# Past-transformation sub-record helpers
# ---------------------------------------------------------------------------

def _read_past_transformation(f):
    """Read a single past-spatial-transformation record."""
    name = read_variable_length_string(f)
    tr_type, = struct.unpack("<i", f.read(4))
    source = read_variable_length_string(f)
    nr_values, = struct.unpack("<i", f.read(4))
    values = []
    for _ in range(nr_values):
        v, = struct.unpack("<f", f.read(4))
        values.append(v)
    return {
        "Name": name,
        "Type": tr_type,
        "SourceFileName": source,
        "NrOfValues": nr_values,
        "Values": values,
    }


def _write_past_transformation(f, record):
    """Write a single past-spatial-transformation record."""
    write_variable_length_string(f, record["Name"])
    f.write(struct.pack("<i", record["Type"]))
    write_variable_length_string(f, record["SourceFileName"])
    f.write(struct.pack("<i", record["NrOfValues"]))
    for v in record["Values"]:
        f.write(struct.pack("<f", v))


# ---------------------------------------------------------------------------
# Typed VMR format
# ---------------------------------------------------------------------------

@register_format(".vmr")
class VMR(BinaryFormat):
    """Typed BrainVoyager VMR (volumetric anatomical dataset).

    Field order matches the on-disk layout exactly: pre-data header,
    raw voxel data, post-data header with conditional (version-gated)
    fields, and optional past-spatial-transformation records.
    """

    # -- Pre-data header -------------------------------------------------
    file_version = Field("<H")
    dim_x = Field("<H")
    dim_y = Field("<H")
    dim_z = Field("<H")

    # -- Voxel data (BV internal layout → Talairach layout on read) ------
    data = DataField(
        dtype="<B",
        shape_fields=("dim_z", "dim_y", "dim_x"),
        transform=_bv_to_tal,
        inverse_transform=_bv_to_tal_inv,
    )

    # -- Post-data header ------------------------------------------------
    offset_x = Field(
        "<h", condition=lambda s: s.file_version >= 3, default=0
    )
    offset_y = Field(
        "<h", condition=lambda s: s.file_version >= 3, default=0
    )
    offset_z = Field(
        "<h", condition=lambda s: s.file_version >= 3, default=0
    )
    framing_cube_dim = Field(
        "<h", condition=lambda s: s.file_version >= 3, default=256
    )

    pos_infos_verified = Field("<i", default=1)
    coordinate_system = Field("<i", default=0)

    slice_1_center_x = Field("<f", default=-87.5)
    slice_1_center_y = Field("<f", default=0.0)
    slice_1_center_z = Field("<f", default=0.0)
    slice_n_center_x = Field("<f", default=87.5)
    slice_n_center_y = Field("<f", default=0.0)
    slice_n_center_z = Field("<f", default=0.0)

    row_dir_x = Field("<f", default=0.0)
    row_dir_y = Field("<f", default=1.0)
    row_dir_z = Field("<f", default=0.0)
    col_dir_x = Field("<f", default=0.0)
    col_dir_y = Field("<f", default=0.0)
    col_dir_z = Field("<f", default=-1.0)

    n_rows = Field("<i", default=256)
    n_cols = Field("<i", default=256)

    fov_rows = Field("<f", default=256.0)
    fov_cols = Field("<f", default=256.0)
    slice_thickness = Field("<f", default=1.0)
    gap_thickness = Field("<f", default=0.0)

    nr_of_past_spatial_transformations = Field("<i", default=0)

    past_transformations = SubRecordListField(
        count_field="nr_of_past_spatial_transformations",
        read_record=_read_past_transformation,
        write_record=_write_past_transformation,
    )

    left_right_convention = Field("<B", default=1)

    reference_space_vmr = Field(
        "<B",
        condition=lambda s: s.file_version >= 4,
        default=0,
    )

    voxel_size_x = Field("<f", default=1.0)
    voxel_size_y = Field("<f", default=1.0)
    voxel_size_z = Field("<f", default=1.0)

    voxel_resolution_verified = Field("<B", default=1)
    voxel_resolution_in_tal_mm = Field("<B", default=1)

    vmr_orig_v16_min_value = Field("<i", default=-1)
    vmr_orig_v16_mean_value = Field("<i", default=-1)
    vmr_orig_v16_max_value = Field("<i", default=-1)

    # ------------------------------------------------------------------
    # Legacy-key mapping (Python attr name → original dict key)
    # ------------------------------------------------------------------
    _LEGACY_MAP = {
        "file_version": "File version",
        "dim_x": "DimX",
        "dim_y": "DimY",
        "dim_z": "DimZ",
        "offset_x": "OffsetX",
        "offset_y": "OffsetY",
        "offset_z": "OffsetZ",
        "framing_cube_dim": "FramingCubeDim",
        "pos_infos_verified": "PosInfosVerified",
        "coordinate_system": "CoordinateSystem",
        "slice_1_center_x": "Slice1CenterX",
        "slice_1_center_y": "Slice1CenterY",
        "slice_1_center_z": "Slice1CenterZ",
        "slice_n_center_x": "SliceNCenterX",
        "slice_n_center_y": "SliceNCenterY",
        "slice_n_center_z": "SliceNCenterZ",
        "row_dir_x": "RowDirX",
        "row_dir_y": "RowDirY",
        "row_dir_z": "RowDirZ",
        "col_dir_x": "ColDirX",
        "col_dir_y": "ColDirY",
        "col_dir_z": "ColDirZ",
        "n_rows": "NRows",
        "n_cols": "NCols",
        "fov_rows": "FoVRows",
        "fov_cols": "FoVCols",
        "slice_thickness": "SliceThickness",
        "gap_thickness": "GapThickness",
        "nr_of_past_spatial_transformations": "NrOfPastSpatialTransformations",
        "past_transformations": "PastTransformation",
        "left_right_convention": "LeftRightConvention",
        "reference_space_vmr": "ReferenceSpaceVMR",
        "voxel_size_x": "VoxelSizeX",
        "voxel_size_y": "VoxelSizeY",
        "voxel_size_z": "VoxelSizeZ",
        "voxel_resolution_verified": "VoxelResolutionVerified",
        "voxel_resolution_in_tal_mm": "VoxelResolutionInTALmm",
        "vmr_orig_v16_min_value": "VMROrigV16MinValue",
        "vmr_orig_v16_mean_value": "VMROrigV16MeanValue",
        "vmr_orig_v16_max_value": "VMROrigV16MaxValue",
    }
    # Reverse map for reading legacy dicts
    _LEGACY_REVERSE = {v: k for k, v in _LEGACY_MAP.items()}

    def to_legacy_dict(self):
        """Export header as a dict with original BrainVoyager key names.

        Uses the descriptor protocol so that unset fields return their
        declared defaults, matching the original ``create_vmr()`` behaviour.
        """
        result = {}
        for py_name, legacy_name in self._LEGACY_MAP.items():
            val = getattr(self, py_name, None)
            result[legacy_name] = val
        return result

    @classmethod
    def from_legacy_dict(cls, d, data=None):
        """Create a VMR instance from a legacy-format dict."""
        kwargs = {}
        for legacy_name, py_name in cls._LEGACY_REVERSE.items():
            if legacy_name in d:
                kwargs[py_name] = d[legacy_name]
        instance = cls(**kwargs)
        if data is not None:
            instance.data = data
        return instance

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create_default(cls, dim_x=256, dim_y=256, dim_z=256):
        """Return a new VMR instance populated with sensible defaults.

        The voxel data is initialised with random values in [0, 225] as
        unsigned 8-bit integers.
        """
        vmr = cls()
        vmr.file_version = 4
        vmr.dim_x = dim_x
        vmr.dim_y = dim_y
        vmr.dim_z = dim_z
        # Post-data header defaults are already set by Field defaults
        # Generate random data
        shape = (dim_z, dim_y, dim_x)
        vmr.data = (np.random.random(np.prod(shape)) * 225).astype(
            np.uint8
        ).reshape(shape)
        return vmr


# =============================================================================
# Backward-compatible procedural shims
# =============================================================================

def read_vmr(filename):
    """Read BrainVoyager VMR file (legacy API).

    Returns (header_dict, data_ndarray).
    """
    vmr = VMR.read(filename)
    header = vmr.to_legacy_dict()
    data = vmr.data
    return header, data


def write_vmr(filename, header, data_img):
    """Write BrainVoyager VMR file (legacy API)."""
    vmr = VMR.from_legacy_dict(header, data=data_img)
    vmr.write(filename)
    print("VMR saved.")


def create_vmr():
    """Create BrainVoyager VMR file with default values (legacy API).

    Returns (header_dict, data_ndarray).
    """
    vmr = VMR.create_default()
    header = vmr.to_legacy_dict()
    data = vmr.data
    return header, data
