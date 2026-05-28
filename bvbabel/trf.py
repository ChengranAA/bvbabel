"""Read, write, create BrainVoyager TRF (transformation) file format.

TRF is a text format describing spatial transformations (rigid-body,
affine, Talairach, MNI, etc.) between BrainVoyager datasets.

Typed API
---------
    trf = TRF.read("alignment.trf")
    print(trf.transformation_type, trf.matrix.shape)
    trf.write("output.trf")
"""

import numpy as np
from bvbabel._binary_format import (
    Field, ObjectField, Section, BinaryFormat, register_format,
)


# =============================================================================
# Section sub-objects
# =============================================================================

class TrfMatrixData(Section):
    """Transformation matrix container."""

    matrix = Field(default=None)          # 4×4 float64
    extra_vmr_transf = Field(default=None)  # optional 4×4


# =============================================================================
# TRF format
# =============================================================================

@register_format(".trf")
class TRF(BinaryFormat):
    """Typed BrainVoyager TRF (spatial transformation)."""

    # -- Header fields (populated by text parsing) -----------------------
    file_version = Field(default=8)
    data_format = Field(default="Matrix")
    transformation_type = Field(default=0)
    coordinate_system = Field(default=0)

    n_slices_fmr_vmr = Field(default="")
    sl_thick_fmr_vmr = Field(default="")
    sl_gap_fmr_vmr = Field(default="")
    create_fmr3d_method = Field(default="")
    alignment_step = Field(default=0)
    extra_vmr_transf_flag = Field(default=0)
    to_vmr_framing_cube = Field(default="")
    to_vmr_voxel_res = Field(default="")
    acpc_vmr_framing_cube = Field(default="")
    acpc_vmr_voxel_res = Field(default="")

    x_scales_mni = Field(default=None)
    y_scales_mni = Field(default=None)
    z_scales_mni = Field(default=None)

    source_file = Field(default="")
    target_file = Field(default="")

    # -- Sub-object ------------------------------------------------------
    matrices = ObjectField(default_factory=TrfMatrixData)

    # -- Data accessor (alias) -------------------------------------------

    @property
    def matrix(self):
        return self.matrices.matrix if self.matrices else None

    # -- I/O: text parsing -----------------------------------------------

    @classmethod
    def read(cls, filename, load_data=True):
        instance = cls()
        header = {}
        has_vmr_trf = False

        with open(filename, "r") as f:
            lines = [r for r in (line.strip() for line in f) if r]

        for line in lines:
            parts = line.split(":", 1)
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            value = parts[1].strip()

            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                header[key] = int(value)
            else:
                header[key] = value

        # MNI scale lists
        for key in ("xScalesMNI", "yScalesMNI", "zScalesMNI"):
            if key in header and isinstance(header[key], str):
                header[key] = [float(x) for x in header[key].split() if x]

        if header.get("ExtraVMRTransf", 0) > 0:
            has_vmr_trf = True

        # Parse 4×4 matrices (follow the line containing "DataFormat" + "Matrix")
        data = {}
        for i, line in enumerate(lines):
            if "DataFormat" in line and "Matrix" in line:
                m44 = np.zeros((4, 4))
                for j in range(1, 5):
                    vals = lines[i + j].split()
                    for k, v in enumerate(vals):
                        m44[j - 1, k] = float(v)
                data["Matrix"] = m44.copy()

            if "ExtraVMRTransf" in line and has_vmr_trf and ":" in line:
                m44b = np.zeros((4, 4))
                for j in range(1, 5):
                    vals = lines[i + j].split()
                    for k, v in enumerate(vals):
                        m44b[j - 1, k] = float(v)
                data["ExtraVMRTransf"] = m44b.copy()

        # Populate instance
        for key, val in header.items():
            py_name = {
                "FileVersion": "file_version",
                "DataFormat": "data_format",
                "TransformationType": "transformation_type",
                "CoordinateSystem": "coordinate_system",
                "NSlicesFMRVMR": "n_slices_fmr_vmr",
                "SlThickFMRVMR": "sl_thick_fmr_vmr",
                "SlGapFMRVMR": "sl_gap_fmr_vmr",
                "CreateFMR3DMethod": "create_fmr3d_method",
                "AlignmentStep": "alignment_step",
                "ExtraVMRTransf": "extra_vmr_transf_flag",
                "ToVMRFramingCube": "to_vmr_framing_cube",
                "ToVMRVoxelRes": "to_vmr_voxel_res",
                "ACPCVMRFramingCube": "acpc_vmr_framing_cube",
                "ACPCVMRVoxelRes": "acpc_vmr_voxel_res",
                "xScalesMNI": "x_scales_mni",
                "yScalesMNI": "y_scales_mni",
                "zScalesMNI": "z_scales_mni",
                "SourceFile": "source_file",
                "TargetFile": "target_file",
            }.get(key, key.lower())
            if hasattr(instance, py_name):
                setattr(instance, py_name, val)

        instance.matrices = TrfMatrixData(
            matrix=data.get("Matrix"),
            extra_vmr_transf=data.get("ExtraVMRTransf"),
        )
        return instance

    def write(self, filename):
        h = self
        m = self.matrices

        with open(filename, "w") as f:
            f.write(f"\nFileVersion:\t{h.file_version}\n\n")
            f.write("DataFormat: \tMatrix\n\n")
            if m.matrix is not None:
                for i in range(4):
                    f.write(
                        " {:20.16f} {:20.16f} {:20.16f} {:20.16f}\n".format(
                            m.matrix[i, 0], m.matrix[i, 1],
                            m.matrix[i, 2], m.matrix[i, 3],
                        )
                    )
            f.write(f"\nTransformationType: \t{h.transformation_type}\n")
            f.write(f"CoordinateSystem: \t{h.coordinate_system}\n\n")

            if h.transformation_type == 1:
                f.write(f"NSlicesFMRVMR:\t\t{h.n_slices_fmr_vmr}\n")
                f.write(f"SlThickFMRVMR:\t\t{h.sl_thick_fmr_vmr}\n")
                f.write(f"SlGapFMRVMR:\t\t{h.sl_gap_fmr_vmr}\n")
                f.write(f"CreateFMR3DMethod:\t{h.create_fmr3d_method}\n")
                f.write(f"AlignmentStep:\t\t{h.alignment_step}\n\n")
                if h.file_version > 4:
                    f.write(f"ExtraVMRTransf:\t\t{h.extra_vmr_transf_flag}\n\n")
                    if h.extra_vmr_transf_flag > 0 and m.extra_vmr_transf is not None:
                        for i in range(4):
                            f.write(
                                " {:20.16f} {:20.16f} {:20.16f} {:20.16f}\n".format(
                                    m.extra_vmr_transf[i, 0],
                                    m.extra_vmr_transf[i, 1],
                                    m.extra_vmr_transf[i, 2],
                                    m.extra_vmr_transf[i, 3],
                                )
                            )
                        f.write("\n")
                if h.alignment_step == 1 and h.file_version > 5:
                    f.write(f"ToVMRFramingCube:\t{h.to_vmr_framing_cube}\n")
                    f.write(f"ToVMRVoxelRes:\t\t{h.to_vmr_voxel_res}\n\n")
            elif h.transformation_type == 3:
                for scale, name in [
                    (h.x_scales_mni, "xScalesMNI"),
                    (h.y_scales_mni, "yScalesMNI"),
                    (h.z_scales_mni, "zScalesMNI"),
                ]:
                    if scale is not None and len(scale) >= 2:
                        f.write(
                            f"{name}:\t\t{scale[0]:>10.5f}\t"
                            f"{scale[1]:>10.5f}\n"
                        )
                f.write("\n")

            f.write(f"SourceFile:\t\t{h.source_file}\n")
            f.write(f"TargetFile:\t\t{h.target_file}\n\n")

            if h.transformation_type == 2 and "ACPC" in filename:
                f.write(f"ACPCVMRFramingCube:\t{h.acpc_vmr_framing_cube}\n")
                f.write(f"ACPCVMRVoxelRes:\t{h.acpc_vmr_voxel_res}\n\n")

    # -- Factory ---------------------------------------------------------

    @classmethod
    def create_default(cls):
        trf = cls()
        trf.file_version = 8
        trf.transformation_type = 2
        trf.coordinate_system = 0
        trf.source_file = ""
        trf.target_file = ""
        trf.matrices = TrfMatrixData(matrix=np.eye(4))
        return trf

    # -- Legacy key mapping ----------------------------------------------

    _LEGACY_MAP = {
        "file_version": "FileVersion",
        "data_format": "DataFormat",
        "transformation_type": "TransformationType",
        "coordinate_system": "CoordinateSystem",
        "n_slices_fmr_vmr": "NSlicesFMRVMR",
        "sl_thick_fmr_vmr": "SlThickFMRVMR",
        "sl_gap_fmr_vmr": "SlGapFMRVMR",
        "create_fmr3d_method": "CreateFMR3DMethod",
        "alignment_step": "AlignmentStep",
        "extra_vmr_transf_flag": "ExtraVMRTransf",
        "to_vmr_framing_cube": "ToVMRFramingCube",
        "to_vmr_voxel_res": "ToVMRVoxelRes",
        "acpc_vmr_framing_cube": "ACPCVMRFramingCube",
        "acpc_vmr_voxel_res": "ACPCVMRVoxelRes",
        "x_scales_mni": "xScalesMNI",
        "y_scales_mni": "yScalesMNI",
        "z_scales_mni": "zScalesMNI",
        "source_file": "SourceFile",
        "target_file": "TargetFile",
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
            instance.matrices = TrfMatrixData(
                matrix=data.get("Matrix"),
                extra_vmr_transf=data.get("ExtraVMRTransf"),
            )
        return instance


# =============================================================================
# Backward-compatible shims
# =============================================================================

def read_trf(filename):
    trf = TRF.read(filename)
    h = trf.to_legacy_dict()
    m = trf.matrices
    data = {}
    if m.matrix is not None:
        data["Matrix"] = m.matrix
    if m.extra_vmr_transf is not None:
        data["ExtraVMRTransf"] = m.extra_vmr_transf
    return h, data


def write_trf(filename, header, data):
    trf = TRF.from_legacy_dict(header, data=data)
    trf.write(filename)
