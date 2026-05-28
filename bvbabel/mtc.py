"""Read, write, create BrainVoyager MTC (mesh time course) file format."""

import numpy as np
from bvbabel._binary_format import (
    Field, StringField, DataField, BinaryFormat, register_format,
)


@register_format(".mtc")
class MTC(BinaryFormat):
    """Typed BrainVoyager MTC (vertex-wise time course)."""

    file_version = Field("<i")
    nr_vertices = Field("<i")
    nr_time_points = Field("<i")

    vtc_name = StringField()
    prt_name = StringField()

    hemodynamic_delay = Field("<i")
    tr = Field("<f")
    delta = Field("<f")
    tau = Field("<f")

    segment_size = Field("<i")
    segment_offset = Field("<i")
    datatype = Field("<B")

    data = DataField(
        dtype="<f",
        shape_fields=("nr_vertices", "nr_time_points"),
    )

    @classmethod
    def create_default(cls, nr_vertices=3, nr_time_points=3):
        mtc = cls()
        mtc.file_version = 1
        mtc.nr_vertices = nr_vertices
        mtc.nr_time_points = nr_time_points
        mtc.vtc_name = " "
        mtc.prt_name = "<none>"
        mtc.hemodynamic_delay = 1
        mtc.tr = 1.0
        mtc.delta = 2.5
        mtc.tau = 1.25
        mtc.segment_size = 10
        mtc.segment_offset = 0
        mtc.datatype = 1
        mtc.data = (np.random.random(
            (nr_vertices, nr_time_points)) * 2 - 1
        ).astype(np.float32)
        return mtc

    _LEGACY_MAP = {
        "file_version": "File version",
        "nr_vertices": "Nr vertices",
        "nr_time_points": "Nr time points",
        "vtc_name": "VTC name",
        "prt_name": "PRT name",
        "hemodynamic_delay": "Hemodynamic delay",
        "tr": "TR",
        "delta": "delta",
        "tau": "tau",
        "segment_size": "segment size",
        "segment_offset": "segment offset",
        "datatype": "Datatype (1 = float)",
    }
    _LEGACY_REVERSE = {v: k for k, v in _LEGACY_MAP.items()}

    def to_legacy_dict(self):
        return {l: getattr(self, p) for p, l in self._LEGACY_MAP.items()}

    @classmethod
    def from_legacy_dict(cls, d, data=None):
        kwargs = {}
        for ln, pn in cls._LEGACY_REVERSE.items():
            if ln in d:
                kwargs[pn] = d[ln]
        instance = cls(**kwargs)
        if data is not None:
            instance.data = data
        return instance


def read_mtc(filename):
    mtc = MTC.read(filename)
    return mtc.to_legacy_dict(), mtc.data


def write_mtc(filename, header, data_mtc):
    MTC.from_legacy_dict(header, data=data_mtc).write(filename)


def create_mtc():
    mtc = MTC.create_default()
    return mtc.to_legacy_dict(), mtc.data
