"""Read, write, create BrainVoyager ROI (regions of interest) file format."""

import numpy as np
from bvbabel._binary_format import (
    Field, ObjectField, Section, BinaryFormat, register_format,
)


class RoiRegion(Section):
    """A single region of interest."""

    nr_of_rects = Field(default=0)
    from_slice = Field(default=0)
    left = Field(default=0)
    right = Field(default=0)
    top = Field(default=0)
    bottom = Field(default=0)
    nr_of_voxels = Field(default=0)
    coordinates = Field(default=None)  # (N, 3) int array


@register_format(".roi")
class ROI(BinaryFormat):
    """Typed BrainVoyager ROI (regions of interest)."""

    file_version = Field(default=0)
    reference_space = Field(default="")
    original_vmr_resolution_x = Field(default=0)
    original_vmr_resolution_y = Field(default=0)
    original_vmr_resolution_z = Field(default=0)
    original_vmr_offset_x = Field(default=0)
    original_vmr_offset_y = Field(default=0)
    original_vmr_offset_z = Field(default=0)
    original_vmr_framing_cube_dim = Field(default=0)
    left_right_convention = Field(default=0)
    subject_roi_naming_convention = Field(default="")
    nr_of_rois = Field(default=0)

    rois = ObjectField(default_factory=list)

    @classmethod
    def read(cls, filename, load_data=True):
        with open(filename, "r") as f:
            lines = [r for r in (line.strip() for line in f) if r]

        instance = cls()
        roi_starts = [i for i, l in enumerate(lines) if l.startswith("NrOfRects")]
        header_rows = roi_starts[0] if roi_starts else len(lines)

        header = {}
        for line in lines[:header_rows]:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            header[k] = int(v) if v.isdigit() else v

        mapping = {
            "FileVersion": "file_version",
            "ReferenceSpace": "reference_space",
            "OriginalVMRResolutionX": "original_vmr_resolution_x",
            "OriginalVMRResolutionY": "original_vmr_resolution_y",
            "OriginalVMRResolutionZ": "original_vmr_resolution_z",
            "OriginalVMROffsetX": "original_vmr_offset_x",
            "OriginalVMROffsetY": "original_vmr_offset_y",
            "OriginalVMROffsetZ": "original_vmr_offset_z",
            "OriginalVMRFramingCubeDim": "original_vmr_framing_cube_dim",
            "LeftRightConvention": "left_right_convention",
            "SubjectROINamingConvention": "subject_roi_naming_convention",
            "NrOfROIs": "nr_of_rois",
        }
        for k, v in header.items():
            py = mapping.get(k)
            if py:
                setattr(instance, py, v)

        rois = []
        idx = -1
        for line in lines[header_rows:]:
            if ":" not in line:
                continue
            parts = line.split(":", 1)
            key = parts[0].strip()
            val = parts[1].strip() if len(parts) > 1 else ""

            if key == "NrOfRects":
                idx += 1
                r = RoiRegion()
                r.nr_of_rects = int(val) if val.isdigit() else 0
                r.coordinates = []
                rois.append(r)
            elif key == "FromSlice" and idx >= 0:
                rois[idx].from_slice = int(val)
            elif key == "Left" and idx >= 0:
                rois[idx].left = int(val)
            elif key == "Right" and idx >= 0:
                rois[idx].right = int(val)
            elif key == "Top" and idx >= 0:
                rois[idx].top = int(val)
            elif key == "Bottom" and idx >= 0:
                rois[idx].bottom = int(val)
            elif key == "NrOfVoxels" and idx >= 0:
                rois[idx].nr_of_voxels = int(val)
            elif idx >= 0 and key.split()[0].isdigit():
                coords = [int(x) for x in key.split()]
                rois[idx].coordinates.append(coords)

        for r in rois:
            r.coordinates = np.array(r.coordinates) if r.coordinates else np.zeros((0, 3))
        instance.rois = rois
        return instance

    def write(self, filename):
        with open(filename, "w") as f:
            f.write(f"\nFileVersion:                   {self.file_version}\n\n")
            f.write(f"ReferenceSpace:                {self.reference_space}\n\n")
            f.write(f"OriginalVMRResolutionX:        {self.original_vmr_resolution_x}\n")
            f.write(f"OriginalVMRResolutionY:        {self.original_vmr_resolution_y}\n")
            f.write(f"OriginalVMRResolutionZ:        {self.original_vmr_resolution_z}\n")
            f.write(f"OriginalVMROffsetX:            {self.original_vmr_offset_x}\n")
            f.write(f"OriginalVMROffsetY:            {self.original_vmr_offset_y}\n")
            f.write(f"OriginalVMROffsetZ:            {self.original_vmr_offset_z}\n")
            f.write(f"OriginalVMRFramingCubeDim:     {self.original_vmr_framing_cube_dim}\n\n")
            f.write(f"LeftRightConvention:           {self.left_right_convention}\n\n")
            f.write(f"SubjectROINamingConvention:    {self.subject_roi_naming_convention}\n\n\n")
            f.write(f"NrOfROIs:                      {self.nr_of_rois}\n\n")

            for r in (self.rois or []):
                f.write(f"NrOfRects: {r.nr_of_rects}\n")
                f.write(f"FromSlice: {r.from_slice}\n")
                f.write(f"Left: {r.left}\n")
                f.write(f"Right: {r.right}\n")
                f.write(f"Top: {r.top}\n")
                f.write(f"Bottom: {r.bottom}\n")
                f.write(f"NrOfVoxels: {r.nr_of_voxels}\n")
                if r.coordinates is not None:
                    for c in r.coordinates:
                        f.write(f"{c[0]} {c[1]} {c[2]}\n")
                f.write("\n")


def read_roi(filename):
    roi = ROI.read(filename)
    h = {
        "FileVersion": roi.file_version,
        "ReferenceSpace": roi.reference_space,
        "OriginalVMRResolutionX": roi.original_vmr_resolution_x,
        "OriginalVMRResolutionY": roi.original_vmr_resolution_y,
        "OriginalVMRResolutionZ": roi.original_vmr_resolution_z,
        "OriginalVMROffsetX": roi.original_vmr_offset_x,
        "OriginalVMROffsetY": roi.original_vmr_offset_y,
        "OriginalVMROffsetZ": roi.original_vmr_offset_z,
        "OriginalVMRFramingCubeDim": roi.original_vmr_framing_cube_dim,
        "LeftRightConvention": roi.left_right_convention,
        "SubjectROINamingConvention": roi.subject_roi_naming_convention,
        "NrOfROIs": roi.nr_of_rois,
    }
    data = []
    for r in (roi.rois or []):
        data.append({
            "NrOfRects": r.nr_of_rects,
            "FromSlice": r.from_slice,
            "Left": r.left,
            "Right": r.right,
            "Top": r.top,
            "Bottom": r.bottom,
            "NrOfVoxels": r.nr_of_voxels,
            "Coordinates": r.coordinates,
        })
    return h, data
