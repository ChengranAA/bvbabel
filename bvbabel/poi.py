"""Read, write, create BrainVoyager POI (patches of interest) file format."""

import numpy as np
from bvbabel._binary_format import (
    Field, ObjectField, Section, BinaryFormat, register_format,
)


class PoiPatch(Section):
    """A single surface patch of interest."""

    name = Field(default="")
    info_text_file = Field(default="")
    color = Field(default=None)
    label_vertex = Field(default=0)
    nr_of_vertices = Field(default=0)
    vertices = Field(default=None)  # 1D int array of vertex indices


@register_format(".poi")
class POI(BinaryFormat):
    """Typed BrainVoyager POI (surface patches of interest)."""

    file_version = Field(default=0)
    from_mesh_file = Field(default="")
    nr_of_mesh_vertices = Field(default=0)
    nr_of_pois = Field(default=0)
    nr_of_poi_mtcs = Field(default=0)

    pois = ObjectField(default_factory=list)

    @classmethod
    def read(cls, filename, load_data=True):
        with open(filename, "r") as f:
            lines = [r for r in (line.strip() for line in f) if r]

        instance = cls()
        header_rows = 4
        header = {}
        for line in lines[:header_rows]:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            header[k] = int(v) if v.isdigit() else v

        instance.file_version = header.get("FileVersion", 0)
        instance.from_mesh_file = header.get("FromMeshFile", "").strip('"')
        instance.nr_of_mesh_vertices = header.get("NrOfMeshVertices", 0)
        instance.nr_of_pois = header.get("NrOfPOIs", 0)

        pois = []
        idx = -1
        for line in lines[header_rows:]:
            if ":" not in line:
                continue
            parts = line.split(":", 1)
            key = parts[0].strip()
            val = parts[1].strip() if len(parts) > 1 else ""

            if key == "NameOfPOI":
                idx += 1
                p = PoiPatch()
                p.name = val.strip('"')
                p.vertices = []
                pois.append(p)
            elif key == "InfoTextFile" and idx >= 0:
                pois[idx].info_text_file = val
            elif key == "ColorOfPOI" and idx >= 0:
                pois[idx].color = [int(x) for x in val.split()]
            elif key == "LabelVertex" and idx >= 0:
                pois[idx].label_vertex = int(val)
            elif key == "NrOfVertices" and idx >= 0:
                pois[idx].nr_of_vertices = int(val)
            elif key == "NrOfPOIMTCs":
                instance.nr_of_poi_mtcs = int(val)
            elif idx >= 0 and key.split()[0].isdigit():
                pois[idx].vertices.append(int(key))

        for p in pois:
            p.vertices = np.array(p.vertices) if p.vertices else np.array([])
        instance.pois = pois
        return instance

    def write(self, filename):
        with open(filename, "w") as f:
            f.write(f"\nFileVersion:                   {self.file_version}\n\n")
            f.write(f'FromMeshFile:                  "{self.from_mesh_file}"\n\n')
            f.write(f"NrOfMeshVertices:              {self.nr_of_mesh_vertices}\n\n")
            f.write(f"NrOfPOIs:                      {self.nr_of_pois}\n\n\n")

            for p in (self.pois or []):
                f.write(f'NameOfPOI:  "{p.name}"\n')
                f.write(f"InfoTextFile:  {p.info_text_file}\n")
                c = p.color or [0, 0, 0]
                f.write(f"ColorOfPOI: {c[0]} {c[1]} {c[2]}\n")
                f.write(f"LabelVertex:  {p.label_vertex}\n")
                f.write(f"NrOfVertices: {p.nr_of_vertices}\n")
                if p.vertices is not None:
                    for v in p.vertices:
                        f.write(f"{v}\n")
                f.write("\n")

            f.write(f"\nNrOfPOIMTCs: {self.nr_of_poi_mtcs}\n")


def read_poi(filename):
    poi = POI.read(filename)
    h = {
        "FileVersion": poi.file_version,
        "FromMeshFile": poi.from_mesh_file,
        "NrOfMeshVertices": poi.nr_of_mesh_vertices,
        "NrOfPOIs": poi.nr_of_pois,
        "NrOfPOIMTCs": poi.nr_of_poi_mtcs,
    }
    data = []
    for p in (poi.pois or []):
        data.append({
            "NameOfPOI": p.name,
            "InfoTextFile": p.info_text_file,
            "ColorOfPOI": p.color,
            "LabelVertex": p.label_vertex,
            "NrOfVertices": p.nr_of_vertices,
            "Vertices": p.vertices,
        })
    return h, data


def write_poi(filename, header, data_poi):
    poi = POI()
    poi.file_version = header.get("FileVersion", 0)
    poi.from_mesh_file = header.get("FromMeshFile", "").strip('"')
    poi.nr_of_mesh_vertices = header.get("NrOfMeshVertices", 0)
    poi.nr_of_pois = header.get("NrOfPOIs", 0)
    poi.nr_of_poi_mtcs = header.get("NrOfPOIMTCs", 0)
    pois = []
    for d in data_poi:
        p = PoiPatch()
        p.name = d.get("NameOfPOI", "")
        p.info_text_file = d.get("InfoTextFile", "")
        p.color = d.get("ColorOfPOI")
        p.label_vertex = d.get("LabelVertex", 0)
        p.nr_of_vertices = d.get("NrOfVertices", 0)
        p.vertices = d.get("Vertices")
        pois.append(p)
    poi.pois = pois
    poi.write(filename)
