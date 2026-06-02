"""Read, write, create BrainVoyager FMR (functional MR) file format.

FMR is a text header file paired with a binary STC data file.  This
module provides a typed-object API (``FMR`` class) and backward-
compatible procedural shims.

Typed API
---------
    fmr = FMR.read("project.fmr")
    print(fmr.nr_of_volumes, fmr.data.shape)
    print(fmr.position.slice_1_center_x)
    fmr.write("output.fmr")

    # Header-only (no STC data loaded)
    meta = FMR.read("project.fmr", load_data=False)
"""

import os
import numpy as np
from bvbabel._binary_format import (
    Field, DataField, ObjectField, Section, BinaryFormat, register_format,
)
from bvbabel.stc import read_stc, write_stc


# =============================================================================
# Section sub-objects
# =============================================================================

class FmrPositionInfo(Section):
    """Slice position / orientation information."""

    pos_infos_verified = Field()
    coordinate_system = Field()
    slice_1_center_x = Field()
    slice_1_center_y = Field()
    slice_1_center_z = Field()
    slice_n_center_x = Field()
    slice_n_center_y = Field()
    slice_n_center_z = Field()
    row_dir_x = Field()
    row_dir_y = Field()
    row_dir_z = Field()
    col_dir_x = Field()
    col_dir_y = Field()
    col_dir_z = Field()
    n_rows = Field()
    n_cols = Field()
    fov_rows = Field()
    fov_cols = Field()
    slice_thickness = Field()
    gap_thickness = Field()


class FmrTransformInfo(Section):
    """Spatial transformation records."""

    nr_of_past_spatial_transformations = Field(default=0)
    name_of_spatial_transformation = Field(default="")
    type_of_spatial_transformation = Field(default="")
    applied_to_file_name = Field(default="")
    nr_of_transformation_values = Field(default="")
    transformation_matrix = Field(default=None)


class FmrMultibandInfo(Section):
    """Multiband acquisition parameters."""

    first_data_source_file = Field(default="")
    multiband_sequence = Field(default="")
    multiband_factor = Field(default="")
    slice_timing_table_size = Field(default=0)
    slice_timings = Field(default=None)
    acquisition_time = Field(default="")


# =============================================================================
# FMR format
# =============================================================================

@register_format(".fmr")
class FMR(BinaryFormat):
    """Typed BrainVoyager FMR (functional MR dataset).

    The header is stored as a text ``.fmr`` file; the 4-D voxel data is
    stored in a paired ``.stc`` binary file.  Data is always
    ``(Z, X, Y, T)`` in RAS-like axis order.
    """

    # -- Main header fields ----------------------------------------------
    file_version = Field(default="")
    nr_of_volumes = Field(default=0)
    nr_of_slices = Field(default=0)
    nr_of_skipped_volumes = Field(default="")
    prefix = Field(default="")
    data_storage_format = Field(default=0)
    data_type = Field(default=0)          # 1=int16, 2=float32
    tr = Field(default="")
    inter_slice_time = Field(default="")
    time_resolution_verified = Field(default="")
    te = Field(default="")
    slice_acquisition_order = Field(default="")
    slice_acquisition_order_verified = Field(default="")
    resolution_x = Field(default=0)
    resolution_y = Field(default=0)
    load_amr_file = Field(default="")
    show_amr_file = Field(default="")
    image_index = Field(default="")
    layout_n_columns = Field(default="")
    layout_n_rows = Field(default="")
    layout_zoom_level = Field(default="")
    segment_size = Field(default="")
    segment_offset = Field(default="")
    nr_of_linked_protocols = Field(default="")
    protocol_file = Field(default="")
    inplane_resolution_x = Field(default="")
    inplane_resolution_y = Field(default="")
    slice_thickness = Field(default="")
    slice_gap = Field(default="")
    voxel_resolution_verified = Field(default="")
    left_right_convention = Field(default="")

    # -- Sub-objects (populated by parsing, skipped by binary loop) ------
    position = ObjectField(default_factory=FmrPositionInfo)
    transform = ObjectField(default_factory=FmrTransformInfo)
    multiband = ObjectField(default_factory=FmrMultibandInfo)

    # -- I/O: override for text parsing ----------------------------------

    @classmethod
    def read(cls, filename, load_data=True):
        instance = cls()

        # --- Parse text header ---
        info_pos = {}
        info_tra = {}
        info_multiband = {}
        slice_thickness_count = 0
        header_raw = {}  # raw string values from file

        with open(filename, "r") as f:
            lines = f.readlines()

        j = 0
        while j < len(lines):
            line = lines[j]
            line = line.strip()
            if not line:
                j += 1
                continue

            parts = line.split(":", 1)
            key = parts[0].strip()

            if len(parts) < 2:
                j += 1
                continue
            value = parts[1].strip()

            # Skip numeric-only keys (transformation/multiband data rows)
            if key.isdigit():
                j += 1
                continue

            # --- Main header ---
            if key == "FileVersion":
                instance.file_version = value
            elif key == "NrOfVolumes":
                instance.nr_of_volumes = int(value)
            elif key == "NrOfSlices":
                instance.nr_of_slices = int(value)
            elif key == "NrOfSkippedVolumes":
                instance.nr_of_skipped_volumes = value
            elif key == "Prefix":
                instance.prefix = value.strip('"')
            elif key == "DataStorageFormat":
                instance.data_storage_format = int(value)
            elif key == "DataType":
                instance.data_type = int(value)
            elif key == "TR":
                instance.tr = value
            elif key == "InterSliceTime":
                instance.inter_slice_time = value
            elif key == "TimeResolutionVerified":
                instance.time_resolution_verified = value
            elif key == "TE":
                instance.te = value
            elif key == "SliceAcquisitionOrder":
                instance.slice_acquisition_order = value
            elif key == "SliceAcquisitionOrderVerified":
                instance.slice_acquisition_order_verified = value
            elif key in ("ResolutionX", "NrOfColumns"):
                instance.resolution_x = float(value)
            elif key in ("ResolutionY", "NrOfRows"):
                instance.resolution_y = float(value)
            elif key == "LoadAMRFile":
                instance.load_amr_file = value.strip('"')
            elif key == "ShowAMRFile":
                instance.show_amr_file = value
            elif key == "ImageIndex":
                instance.image_index = value
            elif key == "LayoutNColumns":
                instance.layout_n_columns = value
            elif key == "LayoutNRows":
                instance.layout_n_rows = value
            elif key == "LayoutZoomLevel":
                instance.layout_zoom_level = value
            elif key == "SegmentSize":
                instance.segment_size = value
            elif key == "SegmentOffset":
                instance.segment_offset = value
            elif key == "NrOfLinkedProtocols":
                instance.nr_of_linked_protocols = value
            elif key == "ProtocolFile":
                instance.protocol_file = value.strip('"')
            elif key == "InplaneResolutionX":
                instance.inplane_resolution_x = value
            elif key == "InplaneResolutionY":
                instance.inplane_resolution_y = value
            elif key == "SliceThickness" and slice_thickness_count == 0:
                instance.slice_thickness = value
                slice_thickness_count += 1
            elif key == "SliceGap":
                instance.slice_gap = value
            elif key == "VoxelResolutionVerified":
                instance.voxel_resolution_verified = value

            # --- Position information ---
            elif key == "PosInfosVerified":
                info_pos[key] = value
            elif key == "CoordinateSystem":
                info_pos[key] = value
            elif key == "Slice1CenterX":
                info_pos[key] = value
            elif key == "Slice1CenterY":
                info_pos[key] = value
            elif key == "Slice1CenterZ":
                info_pos[key] = value
            elif key == "SliceNCenterX":
                info_pos[key] = value
            elif key == "SliceNCenterY":
                info_pos[key] = value
            elif key == "SliceNCenterZ":
                info_pos[key] = value
            elif key == "RowDirX":
                info_pos[key] = value
            elif key == "RowDirY":
                info_pos[key] = value
            elif key == "RowDirZ":
                info_pos[key] = value
            elif key == "ColDirX":
                info_pos[key] = value
            elif key == "ColDirY":
                info_pos[key] = value
            elif key == "ColDirZ":
                info_pos[key] = value
            elif key == "NRows":
                info_pos[key] = value
            elif key == "NCols":
                info_pos[key] = value
            elif key == "FoVRows":
                info_pos[key] = value
            elif key == "FoVCols":
                info_pos[key] = value
            elif key == "SliceThickness":
                info_pos[key] = value
            elif key == "GapThickness":
                info_pos[key] = value
            elif key == "PositionInformationFromImageHeaders":
                pass

            # --- Transformation ---
            elif key == "NrOfPastSpatialTransformations":
                info_tra[key] = int(value)
            elif key == "NameOfSpatialTransformation":
                info_tra[key] = value
            elif key == "TypeOfSpatialTransformation":
                info_tra[key] = value
            elif key == "AppliedToFileName":
                info_tra[key] = value
            elif key == "NrOfTransformationValues":
                info_tra[key] = value
                # Multi-line affine matrix
                nr_values = int(value)
                affine = []
                v = 0
                n = 1
                while v < nr_values:
                    row_line = lines[j + n].strip().split()
                    for val in row_line:
                        affine.append(float(val))
                    v += len(row_line)
                    n += 1
                j += n - 1  # skip consumed lines
                info_tra["Transformation matrix"] = np.array(
                    affine, dtype=np.float64
                ).reshape(4, 4)

            # --- Left-right convention ---
            elif key == "LeftRightConvention":
                instance.left_right_convention = value

            # --- Multiband ---
            elif key == "FirstDataSourceFile":
                info_multiband[key] = value
            elif key == "MultibandSequence":
                info_multiband[key] = value
            elif key == "MultibandFactor":
                info_multiband[key] = value
            elif key == "SliceTimingTableSize":
                info_multiband[key] = int(value)
                nr_values = int(value)
                timings = []
                for n in range(1, nr_values + 1):
                    timings.append(float(lines[j + n].strip()))
                j += nr_values
                info_multiband["Slice timings"] = timings
            elif key == "AcqusitionTime":
                info_multiband[key] = value

            j += 1

        # Populate section objects
        instance.position = FmrPositionInfo(
            pos_infos_verified=info_pos.get("PosInfosVerified", ""),
            coordinate_system=info_pos.get("CoordinateSystem", ""),
            slice_1_center_x=info_pos.get("Slice1CenterX", ""),
            slice_1_center_y=info_pos.get("Slice1CenterY", ""),
            slice_1_center_z=info_pos.get("Slice1CenterZ", ""),
            slice_n_center_x=info_pos.get("SliceNCenterX", ""),
            slice_n_center_y=info_pos.get("SliceNCenterY", ""),
            slice_n_center_z=info_pos.get("SliceNCenterZ", ""),
            row_dir_x=info_pos.get("RowDirX", ""),
            row_dir_y=info_pos.get("RowDirY", ""),
            row_dir_z=info_pos.get("RowDirZ", ""),
            col_dir_x=info_pos.get("ColDirX", ""),
            col_dir_y=info_pos.get("ColDirY", ""),
            col_dir_z=info_pos.get("ColDirZ", ""),
            n_rows=info_pos.get("NRows", ""),
            n_cols=info_pos.get("NCols", ""),
            fov_rows=info_pos.get("FoVRows", ""),
            fov_cols=info_pos.get("FoVCols", ""),
            slice_thickness=info_pos.get("SliceThickness", ""),
            gap_thickness=info_pos.get("GapThickness", ""),
        )

        instance.transform = FmrTransformInfo(
            nr_of_past_spatial_transformations=info_tra.get(
                "NrOfPastSpatialTransformations", 0
            ),
            name_of_spatial_transformation=info_tra.get(
                "NameOfSpatialTransformation", ""
            ),
            type_of_spatial_transformation=info_tra.get(
                "TypeOfSpatialTransformation", ""
            ),
            applied_to_file_name=info_tra.get("AppliedToFileName", ""),
            nr_of_transformation_values=info_tra.get(
                "NrOfTransformationValues", ""
            ),
            transformation_matrix=info_tra.get(
                "Transformation matrix", None
            ),
        )

        instance.multiband = FmrMultibandInfo(
            first_data_source_file=info_multiband.get(
                "FirstDataSourceFile", ""
            ),
            multiband_sequence=info_multiband.get(
                "MultibandSequence", ""
            ),
            multiband_factor=info_multiband.get("MultibandFactor", ""),
            slice_timing_table_size=info_multiband.get(
                "SliceTimingTableSize", 0
            ),
            slice_timings=info_multiband.get("Slice timings", None),
            acquisition_time=info_multiband.get("AcqusitionTime", ""),
        )

        # --- Load STC data ---
        if load_data and instance.prefix:
            dirname = os.path.dirname(filename)
            stc_path = os.path.join(
                dirname, "{}.stc".format(instance.prefix)
            )
            instance._values["data"] = read_stc(
                stc_path,
                nr_slices=instance.nr_of_slices,
                nr_volumes=instance.nr_of_volumes,
                res_x=int(instance.resolution_x),
                res_y=int(instance.resolution_y),
                data_type=instance.data_type,
                rearrange_data_axes=True,
            )
        else:
            instance._values["data"] = None

        return instance

    def write(self, filename):
        """Write FMR text header and paired STC data file."""
        info_pos = self.position
        info_tra = self.transform
        info_multiband = self.multiband
        basepath = filename.split(os.extsep, 1)[0]
        basename = os.path.basename(basepath)

        with open(filename, "w") as f:
            f.write("\n")
            f.write(f"FileVersion:                   {self.file_version}\n")
            f.write(f"NrOfVolumes:                   {self.nr_of_volumes}\n")
            f.write(f"NrOfSlices:                    {self.nr_of_slices}\n")
            f.write(f"NrOfSkippedVolumes:            {self.nr_of_skipped_volumes}\n")
            f.write(f'Prefix:                        "{basename}"\n')
            f.write(f"DataStorageFormat:             {self.data_storage_format}\n")
            f.write(f"DataType:                      {self.data_type}\n")
            f.write(f"TR:                            {self.tr}\n")
            f.write(f"InterSliceTime:                {self.inter_slice_time}\n")
            f.write(f"TimeResolutionVerified:        {self.time_resolution_verified}\n")
            f.write(f"TE:                            {self.te}\n")
            f.write(f"SliceAcquisitionOrder:         {self.slice_acquisition_order}\n")
            f.write(f"SliceAcquisitionOrderVerified: {self.slice_acquisition_order_verified}\n")
            f.write(f"ResolutionX:                   {self.resolution_x}\n")
            f.write(f"ResolutionY:                   {self.resolution_y}\n")
            load_amr = getattr(self, "load_amr_file", "")
            f.write(f'LoadAMRFile:                   "{load_amr}"\n')
            f.write(f"ShowAMRFile:                   {self.show_amr_file}\n")
            f.write(f"ImageIndex:                    {self.image_index}\n")
            f.write(f"LayoutNColumns:                {self.layout_n_columns}\n")
            f.write(f"LayoutNRows:                   {self.layout_n_rows}\n")
            f.write(f"LayoutZoomLevel:               {self.layout_zoom_level}\n")
            f.write(f"SegmentSize:                   {self.segment_size}\n")
            f.write(f"SegmentOffset:                 {self.segment_offset}\n")
            f.write(f"NrOfLinkedProtocols:           {self.nr_of_linked_protocols}\n")
            protocol = getattr(self, "protocol_file", "")
            f.write(f'ProtocolFile:                  "{protocol}"\n')
            f.write(f"InplaneResolutionX:            {self.inplane_resolution_x}\n")
            f.write(f"InplaneResolutionY:            {self.inplane_resolution_y}\n")
            f.write(f"SliceThickness:                {self.slice_thickness}\n")
            f.write(f"SliceGap:                      {self.slice_gap}\n")
            f.write(f"VoxelResolutionVerified:       {self.voxel_resolution_verified}\n")
            f.write("\n")

            # Position info
            f.write("\n")
            f.write("PositionInformationFromImageHeaders\n")
            f.write("\n")
            f.write(f"PosInfosVerified: {info_pos.pos_infos_verified}\n")
            f.write(f"CoordinateSystem: {info_pos.coordinate_system}\n")
            f.write(f"Slice1CenterX:    {info_pos.slice_1_center_x}\n")
            f.write(f"Slice1CenterY:    {info_pos.slice_1_center_y}\n")
            f.write(f"Slice1CenterZ:    {info_pos.slice_1_center_z}\n")
            f.write(f"SliceNCenterX:    {info_pos.slice_n_center_x}\n")
            f.write(f"SliceNCenterY:    {info_pos.slice_n_center_y}\n")
            f.write(f"SliceNCenterZ:    {info_pos.slice_n_center_z}\n")
            f.write(f"RowDirX:          {info_pos.row_dir_x}\n")
            f.write(f"RowDirY:          {info_pos.row_dir_y}\n")
            f.write(f"RowDirZ:          {info_pos.row_dir_z}\n")
            f.write(f"ColDirX:          {info_pos.col_dir_x}\n")
            f.write(f"ColDirY:          {info_pos.col_dir_y}\n")
            f.write(f"ColDirZ:          {info_pos.col_dir_z}\n")
            f.write(f"NRows:            {info_pos.n_rows}\n")
            f.write(f"NCols:            {info_pos.n_cols}\n")
            f.write(f"FoVRows:          {info_pos.fov_rows}\n")
            f.write(f"FoVCols:          {info_pos.fov_cols}\n")
            f.write(f"SliceThickness:   {info_pos.slice_thickness}\n")
            f.write(f"GapThickness:     {info_pos.gap_thickness}\n")
            f.write("\n")

            # Transformation
            if info_tra.nr_of_past_spatial_transformations > 0:
                f.write("\n")
                f.write(f"NrOfPastSpatialTransformations: {info_tra.nr_of_past_spatial_transformations}\n")
                f.write("\n")
                f.write(f"NameOfSpatialTransformation: {info_tra.name_of_spatial_transformation}\n")
                f.write(f"TypeOfSpatialTransformation: {info_tra.type_of_spatial_transformation}\n")
                f.write(f"AppliedToFileName:           {info_tra.applied_to_file_name}\n")
                f.write(f"NrOfTransformationValues:    {info_tra.nr_of_transformation_values}\n")
                if info_tra.transformation_matrix is not None:
                    aff = info_tra.transformation_matrix
                    for i in range(4):
                        f.write(" {:8.5f}  {:8.5f}  {:8.5f}  {:8.5f}  \n".format(
                            aff[i, 0], aff[i, 1], aff[i, 2], aff[i, 3]))
                f.write("\n")

            # Left-right convention
            f.write("\n")
            f.write(f"LeftRightConvention: {self.left_right_convention}\n")
            f.write("\n")

            # Multiband
            if info_multiband._values:
                f.write("\n")
                if info_multiband.first_data_source_file:
                    f.write(f"FirstDataSourceFile: {info_multiband.first_data_source_file}\n")
                if info_multiband.multiband_sequence:
                    f.write(f"MultibandSequence: {info_multiband.multiband_sequence}\n")
                if info_multiband.multiband_factor:
                    f.write(f"MultibandFactor:   {info_multiband.multiband_factor}\n")
                if info_multiband.slice_timing_table_size:
                    f.write(f"SliceTimingTableSize: {info_multiband.slice_timing_table_size}\n")
                    if info_multiband.slice_timings:
                        for t in info_multiband.slice_timings:
                            f.write(f"{t}\n")
                if info_multiband.acquisition_time:
                    f.write("\n")
                    f.write(f"AcqusitionTime: {info_multiband.acquisition_time}\n")
                    f.write("\n")

        # Write STC data
        dirname = os.path.dirname(filename)
        stc_path = os.path.join(dirname, "{}.stc".format(basename))
        data = self._values.get("data")
        if data is not None:
            write_stc(
                stc_path, data, data_type=self.data_type,
                rearrange_data_axes=True,
            )

    # -- Data (populated from paired STC file) ---------------------------
    data = DataField(dtype="<f", shape_fields=())

    # -- Factory ---------------------------------------------------------

    @classmethod
    def create_default(cls, nr_volumes=100, nr_slices=16,
                       res_x=80, res_y=80):
        fmr = cls()
        fmr.file_version = "7"
        fmr.nr_of_volumes = nr_volumes
        fmr.nr_of_slices = nr_slices
        fmr.nr_of_skipped_volumes = "0"
        fmr.prefix = "bvbabel_default_fmr"
        fmr.data_storage_format = 2
        fmr.data_type = 2
        fmr.tr = "2000"
        fmr.inter_slice_time = "31"
        fmr.time_resolution_verified = "1"
        fmr.te = "30"
        fmr.slice_acquisition_order = "5"
        fmr.slice_acquisition_order_verified = "1"
        fmr.resolution_x = res_x
        fmr.resolution_y = res_y
        fmr.load_amr_file = ""
        fmr.show_amr_file = "1"
        fmr.image_index = "0"
        fmr.layout_n_columns = str(int(np.ceil(np.sqrt(nr_slices))))
        fmr.layout_n_rows = str(int(np.ceil(np.sqrt(nr_slices))))
        fmr.layout_zoom_level = "1"
        fmr.segment_size = "10"
        fmr.segment_offset = "0"
        fmr.nr_of_linked_protocols = "0"
        fmr.protocol_file = ""
        fmr.inplane_resolution_x = "2"
        fmr.inplane_resolution_y = "2"
        fmr.slice_thickness = "2"
        fmr.slice_gap = "0"
        fmr.voxel_resolution_verified = "1"
        fmr.left_right_convention = "0"

        fmr.position = FmrPositionInfo(
            pos_infos_verified="1",
            coordinate_system="1",
            slice_1_center_x="-8.34283",
            slice_1_center_y="-13.0168",
            slice_1_center_z="-12.9074",
            slice_n_center_x="-8.34283",
            slice_n_center_y="23.4012",
            slice_n_center_z="107.715",
            row_dir_x="1.0",
            row_dir_y="0.0",
            row_dir_z="0.0",
            col_dir_x="0.0",
            col_dir_y="0.957319",
            col_dir_z="-0.289032",
            n_rows="100",
            n_cols="100",
            fov_rows="200",
            fov_cols="200",
            slice_thickness="2",
            gap_thickness="0",
        )
        fmr.transform = FmrTransformInfo()
        fmr.multiband = FmrMultibandInfo()

        dims = (res_y, res_x, nr_slices, nr_volumes)
        fmr.data = (np.random.random(np.prod(dims)) * (2**16)).astype(
            np.uint16
        ).reshape(dims)
        return fmr

    @classmethod
    def from_nifti(cls, nifti_path, prefix="bv_fmr"):
        """Create an FMR from a 4D NIfTI file.

        Passes RAS+ NIfTI data directly to ``FMR.data`` — the
        ``write_stc(rearrange_data_axes=True)`` call inside ``write()``
        handles the BV-oriented STC layout automatically.

        Parameters
        ----------
        nifti_path : str
            Path to a ``.nii`` or ``.nii.gz`` 4D NIfTI file.
        prefix : str
            Base name for the ``.fmr`` / ``.stc`` file pair.

        Returns
        -------
        FMR instance with header and data populated.
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required for NIfTI import")

        img = nib.load(nifti_path)
        data = img.get_fdata().astype(np.float32)
        zooms = img.header.get_zooms()

        if len(data.shape) != 4:
            raise ValueError(f"Expected 4D NIfTI, got shape {data.shape}")

        nr_volumes = data.shape[3]
        nr_slices = data.shape[2]
        res_x = data.shape[0]
        res_y = data.shape[1]

        fmr = cls()
        fmr.file_version = "7"
        fmr.nr_of_volumes = nr_volumes
        fmr.nr_of_slices = nr_slices
        fmr.prefix = prefix
        fmr.data_storage_format = 2
        fmr.data_type = 2  # float32
        fmr.tr = str(int(zooms[3] * 1000))
        fmr.inter_slice_time = "0"
        fmr.time_resolution_verified = "1"
        fmr.te = "30"
        fmr.slice_acquisition_order = "5"
        fmr.slice_acquisition_order_verified = "1"
        fmr.resolution_x = res_x
        fmr.resolution_y = res_y
        fmr.inplane_resolution_x = str(zooms[0])
        fmr.inplane_resolution_y = str(zooms[1])
        fmr.slice_thickness = str(zooms[2])
        fmr.slice_gap = "0"
        fmr.voxel_resolution_verified = "1"
        fmr.left_right_convention = "0"
        fmr.layout_n_columns = str(int(np.ceil(np.sqrt(nr_slices))))
        fmr.layout_n_rows = str(int(np.ceil(np.sqrt(nr_slices))))
        fmr.layout_zoom_level = "1"
        fmr.segment_size = "10"
        fmr.segment_offset = "0"

        # Pass RAS+ data — write_stc(rearrange=True) converts to BV layout
        fmr.data = data

        # Position info matching the data dimensions
        fmr.position = FmrPositionInfo(
            pos_infos_verified="1",
            coordinate_system="1",
            slice_1_center_x=str(-zooms[0] * (res_x - 1) / 2),
            slice_1_center_y=str(-zooms[1] * (res_y - 1) / 2),
            slice_1_center_z=str(-zooms[2] * (nr_slices - 1) / 2),
            slice_n_center_x=str(zooms[0] * (res_x - 1) / 2),
            slice_n_center_y=str(zooms[1] * (res_y - 1) / 2),
            slice_n_center_z=str(zooms[2] * (nr_slices - 1) / 2),
            row_dir_x="1.0", row_dir_y="0.0", row_dir_z="0.0",
            col_dir_x="0.0", col_dir_y="1.0", col_dir_z="0.0",
            n_rows=str(res_y), n_cols=str(res_x),
            fov_rows=str(res_y * zooms[1]),
            fov_cols=str(res_x * zooms[0]),
            slice_thickness=str(zooms[2]),
            gap_thickness="0",
        )
        fmr.transform = FmrTransformInfo()
        fmr.multiband = FmrMultibandInfo()

        return fmr

    # -- Legacy key mapping ----------------------------------------------

    _LEGACY_MAP = {
        "file_version": "FileVersion",
        "nr_of_volumes": "NrOfVolumes",
        "nr_of_slices": "NrOfSlices",
        "nr_of_skipped_volumes": "NrOfSkippedVolumes",
        "prefix": "Prefix",
        "data_storage_format": "DataStorageFormat",
        "data_type": "DataType",
        "tr": "TR",
        "inter_slice_time": "InterSliceTime",
        "time_resolution_verified": "TimeResolutionVerified",
        "te": "TE",
        "slice_acquisition_order": "SliceAcquisitionOrder",
        "slice_acquisition_order_verified": "SliceAcquisitionOrderVerified",
        "resolution_x": "ResolutionX",
        "resolution_y": "ResolutionY",
        "load_amr_file": "LoadAMRFile",
        "show_amr_file": "ShowAMRFile",
        "image_index": "ImageIndex",
        "layout_n_columns": "LayoutNColumns",
        "layout_n_rows": "LayoutNRows",
        "layout_zoom_level": "LayoutZoomLevel",
        "segment_size": "SegmentSize",
        "segment_offset": "SegmentOffset",
        "nr_of_linked_protocols": "NrOfLinkedProtocols",
        "protocol_file": "ProtocolFile",
        "inplane_resolution_x": "InplaneResolutionX",
        "inplane_resolution_y": "InplaneResolutionY",
        "slice_thickness": "SliceThickness",
        "slice_gap": "SliceGap",
        "voxel_resolution_verified": "VoxelResolutionVerified",
        "left_right_convention": "LeftRightConvention",
    }
    _LEGACY_REVERSE = {v: k for k, v in _LEGACY_MAP.items()}

    _POS_LEGACY_MAP = {
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
    }

    _TRA_LEGACY_MAP = {
        "nr_of_past_spatial_transformations": "NrOfPastSpatialTransformations",
        "name_of_spatial_transformation": "NameOfSpatialTransformation",
        "type_of_spatial_transformation": "TypeOfSpatialTransformation",
        "applied_to_file_name": "AppliedToFileName",
        "nr_of_transformation_values": "NrOfTransformationValues",
        "transformation_matrix": "Transformation matrix",
    }

    _MULTI_LEGACY_MAP = {
        "first_data_source_file": "FirstDataSourceFile",
        "multiband_sequence": "MultibandSequence",
        "multiband_factor": "MultibandFactor",
        "slice_timing_table_size": "SliceTimingTableSize",
        "slice_timings": "Slice timings",
        "acquisition_time": "AcqusitionTime",
    }

    def to_legacy_dict(self):
        result = {}
        for py_name, legacy_name in self._LEGACY_MAP.items():
            result[legacy_name] = getattr(self, py_name)

        pos = {}
        for py_name, legacy_name in self._POS_LEGACY_MAP.items():
            pos[legacy_name] = getattr(self.position, py_name, "")
        result["Position information"] = pos

        tra = {}
        for py_name, legacy_name in self._TRA_LEGACY_MAP.items():
            tra[legacy_name] = getattr(self.transform, py_name, "")
        result["Transformation information"] = tra

        multi = {}
        for py_name, legacy_name in self._MULTI_LEGACY_MAP.items():
            multi[legacy_name] = getattr(self.multiband, py_name, "")
        result["Multiband information"] = multi

        return result

    @classmethod
    def from_legacy_dict(cls, d, data=None):
        kwargs = {}
        for legacy_name, py_name in cls._LEGACY_REVERSE.items():
            if legacy_name in d:
                kwargs[py_name] = d[legacy_name]
        instance = cls(**kwargs)

        pos_d = d.get("Position information", {})
        pos_kwargs = {}
        for py_name, legacy_name in cls._POS_LEGACY_MAP.items():
            if legacy_name in pos_d:
                pos_kwargs[py_name] = pos_d[legacy_name]
        instance.position = FmrPositionInfo(**pos_kwargs)

        tra_d = d.get("Transformation information", {})
        tra_kwargs = {}
        for py_name, legacy_name in cls._TRA_LEGACY_MAP.items():
            if legacy_name in tra_d:
                tra_kwargs[py_name] = tra_d[legacy_name]
        instance.transform = FmrTransformInfo(**tra_kwargs)

        multi_d = d.get("Multiband information", {})
        multi_kwargs = {}
        for py_name, legacy_name in cls._MULTI_LEGACY_MAP.items():
            if legacy_name in multi_d:
                multi_kwargs[py_name] = multi_d[legacy_name]
        instance.multiband = FmrMultibandInfo(**multi_kwargs)

        if data is not None:
            instance.data = data
        return instance


# =============================================================================
# Backward-compatible shims
# =============================================================================

def read_fmr(filename, rearrange_data_axes=True):
    """Read BrainVoyager FMR file (legacy API).

    The *rearrange_data_axes* flag is accepted for compatibility but
    has no effect — data is always returned in RAS-like layout.
    """
    fmr = FMR.read(filename)
    return fmr.to_legacy_dict(), fmr.data


def write_fmr(filename, header, data_img, rearrange_data_axes=True):
    """Write BrainVoyager FMR file (legacy API)."""
    fmr = FMR.from_legacy_dict(header, data=data_img)
    fmr.write(filename)


def create_fmr():
    """Create BrainVoyager FMR file with default values (legacy API)."""
    fmr = FMR.create_default()
    return fmr.to_legacy_dict(), fmr.data
