"""Read, write, create BrainVoyager VOI (voxels of interest) file format."""

import numpy as np
from bvbabel._binary_format import (
    Field, ObjectField, Section, BinaryFormat, register_format,
)


class VoiVolume(Section):
    """A single volume of interest."""

    name = Field(default="")
    color = Field(default=None)
    nr_of_voxels = Field(default=0)
    coordinates = Field(default=None)  # (N, 3) int array


@register_format(".voi")
class VOI(BinaryFormat):
    """Typed BrainVoyager VOI (volume of interest)."""

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
    subject_voi_naming_convention = Field(default="")
    nr_of_vois = Field(default=0)
    nr_of_voi_vtcs = Field(default="")

    vois = ObjectField(default_factory=list)

    @classmethod
    def read(cls, filename, load_data=True):
        with open(filename, "r") as f:
            lines = [r for r in (line.strip() for line in f) if r]

        instance = cls()
        header = {}
        header_rows = 12
        for line in lines[:header_rows]:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            header[k] = int(v) if v.isdigit() else v

        for k, v in header.items():
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
                "SubjectVOINamingConvention": "subject_voi_naming_convention",
                "NrOfVOIs": "nr_of_vois",
                "NrOfVOIVTCs": "nr_of_voi_vtcs",
            }
            py = mapping.get(k)
            if py and hasattr(instance, py):
                setattr(instance, py, v)

        vois = []
        idx = -1
        for line in lines[header_rows:]:
            if ":" not in line:
                continue
            parts = line.split(":", 1)
            key = parts[0].strip()
            val = parts[1].strip() if len(parts) > 1 else ""

            if key == "NameOfVOI":
                idx += 1
                vv = VoiVolume()
                vv.name = val
                vv.coordinates = []
                vois.append(vv)
            elif key == "ColorOfVOI" and idx >= 0:
                vois[idx].color = [int(x) for x in val.split()]
            elif key == "NrOfVoxels" and idx >= 0:
                vois[idx].nr_of_voxels = int(val)
            elif key == "NrOfVOIVTCs":
                instance.nr_of_voi_vtcs = val
            elif idx >= 0 and len(vois[idx].coordinates) < vois[idx].nr_of_voxels:
                # Coordinate line
                coords = [int(x) for x in key.split() if x.lstrip("-").isdigit()]
                if coords:
                    vois[idx].coordinates.append(coords)

        for vv in vois:
            vv.coordinates = np.array(vv.coordinates) if vv.coordinates else np.zeros((0, 3))
        instance.vois = vois
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
            f.write(f"SubjectVOINamingConvention:    {self.subject_voi_naming_convention}\n\n\n")
            f.write(f"NrOfVOIs:                      {self.nr_of_vois}\n\n")

            for vv in (self.vois or []):
                f.write(f"NameOfVOI:  {vv.name}\n")
                c = vv.color or [0, 0, 0]
                f.write(f"ColorOfVOI: {c[0]} {c[1]} {c[2]}\n\n")
                f.write(f"NrOfVoxels: {vv.nr_of_voxels}\n")
                if vv.coordinates is not None:
                    for coord in vv.coordinates:
                        f.write(f"{coord[0]} {coord[1]} {coord[2]}\n")
                f.write("\n")

            f.write(f"\nNrOfVOIVTCs: {self.nr_of_voi_vtcs}\n")
            f.write(f"{self.nr_of_voi_vtcs}")


def read_voi(filename):
    voi = VOI.read(filename)
    h = {
        "FileVersion": voi.file_version,
        "ReferenceSpace": voi.reference_space,
        "OriginalVMRResolutionX": voi.original_vmr_resolution_x,
        "OriginalVMRResolutionY": voi.original_vmr_resolution_y,
        "OriginalVMRResolutionZ": voi.original_vmr_resolution_z,
        "OriginalVMROffsetX": voi.original_vmr_offset_x,
        "OriginalVMROffsetY": voi.original_vmr_offset_y,
        "OriginalVMROffsetZ": voi.original_vmr_offset_z,
        "OriginalVMRFramingCubeDim": voi.original_vmr_framing_cube_dim,
        "LeftRightConvention": voi.left_right_convention,
        "SubjectVOINamingConvention": voi.subject_voi_naming_convention,
        "NrOfVOIs": voi.nr_of_vois,
        "NrOfVOIVTCs": voi.nr_of_voi_vtcs,
    }
    data = []
    for vv in (voi.vois or []):
        data.append({
            "NameOfVOI": vv.name,
            "ColorOfVOI": vv.color,
            "NrOfVoxels": vv.nr_of_voxels,
            "Coordinates": vv.coordinates,
        })
    return h, data


def write_voi(filename, header, data_voi):
    mapping = {
        "FileVersion": "file_version", "ReferenceSpace": "reference_space",
        "OriginalVMRResolutionX": "original_vmr_resolution_x",
        "OriginalVMRResolutionY": "original_vmr_resolution_y",
        "OriginalVMRResolutionZ": "original_vmr_resolution_z",
        "OriginalVMROffsetX": "original_vmr_offset_x",
        "OriginalVMROffsetY": "original_vmr_offset_y",
        "OriginalVMROffsetZ": "original_vmr_offset_z",
        "OriginalVMRFramingCubeDim": "original_vmr_framing_cube_dim",
        "LeftRightConvention": "left_right_convention",
        "SubjectVOINamingConvention": "subject_voi_naming_convention",
        "NrOfVOIs": "nr_of_vois", "NrOfVOIVTCs": "nr_of_voi_vtcs",
    }
    voi = VOI()
    for k, v in header.items():
        py = mapping.get(k)
        if py and hasattr(voi, py):
            setattr(voi, py, v)
    vois = []
    for d in data_voi:
        vv = VoiVolume()
        vv.name = d.get("NameOfVOI", "")
        vv.color = d.get("ColorOfVOI")
        vv.nr_of_voxels = d.get("NrOfVoxels", 0)
        vv.coordinates = d.get("Coordinates")
        vois.append(vv)
    voi.vois = vois
    voi.write(filename)
