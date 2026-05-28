"""Read, write, create BrainVoyager SSM (surface-to-surface mapping) file."""

import numpy as np
from bvbabel._binary_format import (
    Field, DataField, BinaryFormat, register_format,
)


@register_format(".ssm")
class SSM(BinaryFormat):
    """Typed BrainVoyager SSM (vertex index mapping between surfaces)."""

    file_version = Field("<h")
    nr_vertices_1 = Field("<i")
    nr_vertices_2 = Field("<i")

    data = DataField(dtype="<i", shape_fields=("nr_vertices_1",))

    @classmethod
    def create_default(cls, nr_vertices=32492):
        ssm = cls()
        ssm.file_version = 2
        ssm.nr_vertices_1 = int(nr_vertices)
        ssm.nr_vertices_2 = int(nr_vertices)
        ssm.data = np.arange(1, nr_vertices + 1, dtype=np.int32)
        return ssm

    _LEGACY_MAP = {
        "file_version": "File version",
        "nr_vertices_1": "Nr vertices 1",
        "nr_vertices_2": "Nr vertices 2",
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


def read_ssm(filename):
    ssm = SSM.read(filename)
    return ssm.to_legacy_dict(), ssm.data


def write_ssm(filename, header, data_ssm):
    SSM.from_legacy_dict(header, data=data_ssm).write(filename)


def create_ssm(nr_vertices=32492):
    ssm = SSM.create_default(nr_vertices)
    return ssm.to_legacy_dict(), ssm.data
