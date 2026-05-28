"""Read, write, create BrainVoyager VMP (volumetric map) file format.

Provides a typed-object API (``VMP`` / ``MapRecord`` classes) and
backward-compatible procedural shims (``read_vmp``, ``write_vmp``,
``create_vmp``).

Typed API
---------
    vmp = VMP.read("results.vmp")
    print(vmp.nr_of_sub_maps, vmp.data.shape)
    for m in vmp.maps:
        print(m.map_name, m.type_of_map)
    vmp.write("output.vmp")

    # Header-only
    meta = VMP.read("results.vmp", load_data=False)

Procedural API (backward-compatible)
------------------------------------
    header, data = bvbabel.vmp.read_vmp("results.vmp")
    bvbabel.vmp.write_vmp("output.vmp", header, data)
    header, data = bvbabel.vmp.create_vmp()
"""

import struct
import numpy as np
from bvbabel._binary_format import (
    Field, StringField, RGBField, DataField,
    TypedSubRecordListField, BinaryFormat, register_format,
)
from bvbabel.utils import read_variable_length_string, write_variable_length_string


# =============================================================================
# Map sub-record
# =============================================================================

class MapRecord(BinaryFormat):
    """A single statistical sub-map within a VMP file."""

    type_of_map = Field("<i")
    map_threshold = Field("<f")
    upper_threshold = Field("<f")

    map_name = StringField()

    rgb_positive_min = RGBField()
    rgb_positive_max = RGBField()
    rgb_negative_min = RGBField()
    rgb_negative_max = RGBField()

    use_vmp_color = Field("<B")
    lut_file_name = StringField(default="<default>")
    transparent_color_factor = Field("<f", default=1.0)

    # -- Lag fields (only when TypeOfMap == 3) ---------------------------
    nr_of_lags = Field("<i", condition=lambda s: s.type_of_map == 3)
    display_min_lag = Field("<i", condition=lambda s: s.type_of_map == 3)
    display_max_lag = Field("<i", condition=lambda s: s.type_of_map == 3)
    show_correlation_or_lag = Field("<i", condition=lambda s: s.type_of_map == 3)

    # -- Shared tail fields ----------------------------------------------
    cluster_size_threshold = Field("<i", default=1)
    enable_cluster_size_threshold = Field("<b", default=0)
    show_values_above_upper_threshold = Field("<i", default=1)
    df1 = Field("<i")
    df2 = Field("<i")
    show_pos_neg_values = Field("<b", default=3)

    nr_of_used_voxels = Field("<i")
    size_of_fdr_table = Field("<i")

    # FDR table:  (size_of_fdr_table × 3) float32
    fdr_table_info = DataField(
        dtype="<f",
        shape_fields=("size_of_fdr_table", "_fdr_cols"),
    )

    use_fdr_table_index = Field("<i")

    @property
    def _fdr_cols(self):
        return 3


# =============================================================================
# Trailing-section helpers (interleaved after each map record)
# =============================================================================

def _read_trailing(f, vmp, _map_record):
    """Read ComponentTimeCourseValues / ComponentTimeCourseParams once."""
    # Time course values
    if vmp.nr_of_time_points > 0 and "_tc_values" not in vmp._values:
        tcs = []
        for _ in range(vmp.nr_of_sub_maps):
            tc = np.zeros(vmp.nr_of_time_points, dtype=np.float32)
            for j in range(vmp.nr_of_time_points):
                tc[j], = struct.unpack("<f", f.read(4))
            tcs.append(tc)
        vmp._values["_tc_values"] = tcs

    # Component parameters
    if vmp.nr_of_component_params > 0 and "_tc_params" not in vmp._values:
        params = []
        for _ in range(vmp.nr_of_component_params):
            name = read_variable_length_string(f)
            values = np.zeros(vmp.nr_of_sub_maps, dtype=np.float32)
            for j in range(vmp.nr_of_sub_maps):
                values[j], = struct.unpack("<f", f.read(4))
            params.append({"Name": name, "Values": values})
        vmp._values["_tc_params"] = params


def _write_trailing(f, vmp, _map_record):
    """Write ComponentTimeCourseValues / ComponentTimeCourseParams (repeated per map)."""
    if vmp.nr_of_time_points > 0:
        for tc in vmp._values.get("_tc_values", []):
            for j in range(vmp.nr_of_time_points):
                f.write(struct.pack("<f", tc[j]))

    if vmp.nr_of_component_params > 0:
        for param in vmp._values.get("_tc_params", []):
            write_variable_length_string(f, param["Name"])
            for v in param["Values"]:
                f.write(struct.pack("<f", v))


# =============================================================================
# VMP format
# =============================================================================

@register_format(".vmp")
class VMP(BinaryFormat):
    """Typed BrainVoyager VMP (volumetric statistical map).

    Data is always 4-D ``(n_maps, Z, X, Y)`` matching VMR's Talairach axis order.
    leading dimension of 1 is prepended.  For cross-correlation lag maps
    (``TypeOfMap == 3``) a trailing dimension of 2 is appended::

        data[0, :, :, :, 0]  → lag values
        data[0, :, :, :, 1]  → correlation values
    """

    # -- Main header -----------------------------------------------------
    nr_vmp_identifier = Field("<i")
    version_number = Field("<h")
    document_type = Field("<h")

    nr_of_sub_maps = Field("<i")
    nr_of_time_points = Field("<i")
    nr_of_component_params = Field("<i")

    show_params_range_from = Field("<i")
    show_params_range_to = Field("<i")
    use_for_fingerprint_params_range_from = Field("<i")
    use_for_fingerprint_params_range_to = Field("<i")

    x_start = Field("<i")
    x_end = Field("<i")
    y_start = Field("<i")
    y_end = Field("<i")
    z_start = Field("<i")
    z_end = Field("<i")

    resolution = Field("<i")
    header_dim_x = Field("<i")   # stored explicitly; use dim_x property
    header_dim_y = Field("<i")
    header_dim_z = Field("<i")

    name_of_vtc_file = StringField()
    name_of_protocol_file = StringField()
    name_of_voi_file = StringField()

    # -- Map records -----------------------------------------------------
    maps = TypedSubRecordListField(
        count_field="nr_of_sub_maps",
        record_cls=MapRecord,
        post_record_read=_read_trailing,
        post_record_write=_write_trailing,
    )

    # -- Voxel data (flat on disk; reshaped in _post_read) ---------------
    data = DataField(dtype="<f", shape_fields=("_data_raw_count",))

    # -- Computed dimensions ---------------------------------------------

    @property
    def dim_x(self):
        return (self.x_end - self.x_start) // self.resolution

    @property
    def dim_y(self):
        return (self.y_end - self.y_start) // self.resolution

    @property
    def dim_z(self):
        return (self.z_end - self.z_start) // self.resolution

    @property
    def _data_raw_count(self):
        return self.nr_of_sub_maps * self.dim_z * self.dim_y * self.dim_x

    # -- Time-course accessors (set by _read_trailing) -------------------

    @property
    def component_time_course_values(self):
        return self._values.get("_tc_values", [])

    @property
    def component_time_course_params(self):
        return self._values.get("_tc_params", [])

    # -- Hooks: reshape data ↔ flat --------------------------------------

    def _post_read(self):
        """Reshape flat float32 data → 4-D (maps, Z, Y, X)."""
        if self.data is None:
            return
        n_maps = self.nr_of_sub_maps
        z, y, x = self.dim_z, self.dim_y, self.dim_x

        if n_maps > 1:
            self.data = self.data.reshape(n_maps, z, y, x)
            # BV → Tal: transpose + flip spatial axes
            self.data = np.transpose(self.data, (1, 3, 2, 0))
            self.data = self.data[::-1, ::-1, ::-1, :]
            self.data = np.moveaxis(self.data, -1, 0)  # (T, Z, X, Y)
        else:
            self.data = self.data.reshape(z, y, x)
            self.data = np.transpose(self.data, (0, 2, 1))
            self.data = self.data[::-1, ::-1, ::-1]
            # Lag maps: integer part = lag, fractional = correlation
            if self.maps and self.maps[0].type_of_map == 3:
                lag = np.floor(self.data).astype(np.float32)
                corr = (self.data - lag).astype(np.float32)
                self.data = np.stack([lag, corr], axis=-1)
            self.data = self.data[np.newaxis, ...]  # (1, Z, Y, X[, 2])

    def _pre_write(self):
        """Flatten 4-D data back to on-disk layout."""
        if self.data is None:
            return
        n_maps = self.nr_of_sub_maps

        if n_maps == 1:
            data = self.data[0]
            if data.ndim == 4 and data.shape[-1] == 2:
                data = data[..., 0] + data[..., 1]
            data = data[::-1, ::-1, ::-1]
            data = np.transpose(data, (0, 2, 1))
            self.data = data.ravel().astype(np.float32)
        else:
            data = np.moveaxis(self.data, 0, -1)
            data = data[::-1, ::-1, ::-1, :]
            data = np.transpose(data, (3, 0, 2, 1))
            self.data = data.ravel().astype(np.float32)

    # -- Factory ---------------------------------------------------------

    @classmethod
    def create_default(cls, dim_x=256, dim_y=256, dim_z=256, n_maps=1):
        vmp = cls()
        vmp.nr_vmp_identifier = np.int32(-1582119980)
        vmp.version_number = 6
        vmp.document_type = 1
        vmp.nr_of_sub_maps = n_maps
        vmp.nr_of_time_points = 0
        vmp.nr_of_component_params = 0
        vmp.show_params_range_from = 0
        vmp.show_params_range_to = 0
        vmp.use_for_fingerprint_params_range_from = 0
        vmp.use_for_fingerprint_params_range_to = 0
        vmp.x_start = 0
        vmp.x_end = dim_x
        vmp.y_start = 0
        vmp.y_end = dim_y
        vmp.z_start = 0
        vmp.z_end = dim_z
        vmp.resolution = 1
        vmp.header_dim_x = dim_x
        vmp.header_dim_y = dim_y
        vmp.header_dim_z = dim_z
        vmp.name_of_vtc_file = ""
        vmp.name_of_protocol_file = ""
        vmp.name_of_voi_file = ""

        maps = []
        for _ in range(n_maps):
            m = MapRecord()
            m.type_of_map = 1
            m.map_threshold = 0.5
            m.upper_threshold = 1.0
            m.map_name = "bvbabel default"
            m.rgb_positive_min = np.array([224, 243, 248], dtype=np.ubyte)
            m.rgb_positive_max = np.array([40, 51, 144], dtype=np.ubyte)
            m.rgb_negative_min = np.array([254, 236, 153], dtype=np.ubyte)
            m.rgb_negative_max = np.array([145, 0, 37], dtype=np.ubyte)
            m.use_vmp_color = 0
            m.lut_file_name = "<default>"
            m.transparent_color_factor = 1.0
            m.cluster_size_threshold = 1
            m.enable_cluster_size_threshold = 0
            m.show_values_above_upper_threshold = 1
            m.df1 = 0
            m.df2 = 0
            m.show_pos_neg_values = 3
            m.nr_of_used_voxels = 0
            m.size_of_fdr_table = 0
            m.fdr_table_info = np.zeros((0, 3), dtype=np.float32)
            m.use_fdr_table_index = 0
            maps.append(m)
        vmp.maps = maps

        shape = (n_maps, dim_z, dim_y, dim_x)
        vmp.data = (np.random.random(np.prod(shape)) * 2 - 1).astype(
            np.float32
        ).reshape(shape)
        return vmp

    # -- Legacy key mapping ----------------------------------------------

    _LEGACY_MAP = {
        "nr_vmp_identifier": "NR-VMP identifier",
        "version_number": "VersionNumber",
        "document_type": "DocumentType",
        "nr_of_sub_maps": "NrOfSubMaps",
        "nr_of_time_points": "NrOfTimePoints",
        "nr_of_component_params": "NrOfComponentParams",
        "show_params_range_from": "ShowParamsRangeFrom",
        "show_params_range_to": "ShowParamsRangeTo",
        "use_for_fingerprint_params_range_from": "UseForFingerprintParamsRangeFrom",
        "use_for_fingerprint_params_range_to": "UseForFingerprintParamsRangeTo",
        "x_start": "XStart",
        "x_end": "XEnd",
        "y_start": "YStart",
        "y_end": "YEnd",
        "z_start": "ZStart",
        "z_end": "ZEnd",
        "resolution": "Resolution",
        "header_dim_x": "DimX",
        "header_dim_y": "DimY",
        "header_dim_z": "DimZ",
        "name_of_vtc_file": "NameOfVTCFile",
        "name_of_protocol_file": "NameOfProtocolFile",
        "name_of_voi_file": "NameOfVOIFile",
    }
    _LEGACY_REVERSE = {v: k for k, v in _LEGACY_MAP.items()}

    _MAP_LEGACY_MAP = {
        "type_of_map": "TypeOfMap",
        "map_threshold": "MapThreshold",
        "upper_threshold": "UpperThreshold",
        "map_name": "MapName",
        "rgb_positive_min": "RGB positive min",
        "rgb_positive_max": "RGB positive max",
        "rgb_negative_min": "RGB negative min",
        "rgb_negative_max": "RGB negative max",
        "use_vmp_color": "UseVMPColor",
        "lut_file_name": "LUTFileName",
        "transparent_color_factor": "TransparentColorFactor",
        "nr_of_lags": "NrOfLags",
        "display_min_lag": "DisplayMinLag",
        "display_max_lag": "DisplayMaxLag",
        "show_correlation_or_lag": "ShowCorrelationOrLag",
        "cluster_size_threshold": "ClusterSizeThreshold",
        "enable_cluster_size_threshold": "EnableClusterSizeThreshold",
        "show_values_above_upper_threshold": "ShowValuesAboveUpperThreshold",
        "df1": "DF1",
        "df2": "DF2",
        "show_pos_neg_values": "ShowPosNegValues",
        "nr_of_used_voxels": "NrOfUsedVoxels",
        "size_of_fdr_table": "SizeOfFDRTable",
        "fdr_table_info": "FDRTableInfo",
        "use_fdr_table_index": "UseFDRTableIndex",
    }

    def to_legacy_dict(self):
        result = {}
        for py_name, legacy_name in self._LEGACY_MAP.items():
            result[legacy_name] = getattr(self, py_name)
        # Maps
        result["Map"] = []
        for m in self.maps:
            md = {}
            for py_name, legacy_name in self._MAP_LEGACY_MAP.items():
                md[legacy_name] = getattr(m, py_name)
            result["Map"].append(md)
        # Trailing sections
        if self.nr_of_time_points > 0:
            result["ComponentTimeCourseValues"] = list(
                self.component_time_course_values
            )
        if self.nr_of_component_params > 0:
            result["ComponentTimeCourseParams"] = list(
                self.component_time_course_params
            )
        return result

    @classmethod
    def from_legacy_dict(cls, d, data=None):
        kwargs = {}
        for legacy_name, py_name in cls._LEGACY_REVERSE.items():
            if legacy_name in d:
                kwargs[py_name] = d[legacy_name]
        instance = cls(**kwargs)

        # Maps
        maps = []
        for md in d.get("Map", []):
            m = MapRecord()
            for py_name, legacy_name in cls._MAP_LEGACY_MAP.items():
                if legacy_name in md:
                    setattr(m, py_name, md[legacy_name])
            maps.append(m)
        instance.maps = maps

        # Trailing sections
        if "ComponentTimeCourseValues" in d:
            instance._values["_tc_values"] = list(
                d["ComponentTimeCourseValues"]
            )
        if "ComponentTimeCourseParams" in d:
            instance._values["_tc_params"] = list(
                d["ComponentTimeCourseParams"]
            )

        if data is not None:
            instance.data = data
        return instance


# =============================================================================
# Backward-compatible procedural shims
# =============================================================================

def read_vmp(filename):
    """Read BrainVoyager VMP file (legacy API)."""
    vmp = VMP.read(filename)
    return vmp.to_legacy_dict(), vmp.data


def write_vmp(filename, header, data_img):
    """Write BrainVoyager VMP file (legacy API)."""
    vmp = VMP.from_legacy_dict(header, data=data_img)
    vmp.write(filename)


def create_vmp():
    """Create BrainVoyager VMP file with default values (legacy API)."""
    vmp = VMP.create_default()
    return vmp.to_legacy_dict(), vmp.data
