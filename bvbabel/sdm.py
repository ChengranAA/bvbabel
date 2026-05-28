"""Read, write, create BrainVoyager SDM (design matrix) file format.

Typed API
---------
    sdm = SDM.read("design.sdm")
    print(sdm.nr_of_predictors, len(sdm.predictors))
    for p in sdm.predictors:
        print(p.name, p.values.shape)
"""

import numpy as np
from bvbabel._binary_format import (
    Field, ObjectField, Section, BinaryFormat, register_format,
)


# =============================================================================
# Sub-objects
# =============================================================================

class SdmPredictor(Section):
    """A single predictor column in the design matrix."""

    name = Field(default="")
    color = Field(default=None)   # [R, G, B]
    values = Field(default=None)  # 1D float64 array


# =============================================================================
# SDM format
# =============================================================================

@register_format(".sdm")
class SDM(BinaryFormat):
    """Typed BrainVoyager SDM (single-subject design matrix)."""

    file_version = Field(default=0)
    nr_of_predictors = Field(default=0)
    nr_of_data_points = Field(default=0)
    includes_constant = Field(default=0)
    first_confound_predictor = Field(default=0)

    predictors = ObjectField(default_factory=list)

    # -- I/O -------------------------------------------------------------

    @classmethod
    def read(cls, filename, load_data=True):
        with open(filename, "r") as f:
            lines = [r for r in (line.strip() for line in f) if r]

        header = {}
        for line in lines[:5]:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            header[k] = int(v) if v.isdigit() else v

        instance = cls()
        instance.file_version = header.get("FileVersion", 0)
        instance.nr_of_predictors = header.get("NrOfPredictors", 0)
        instance.nr_of_data_points = header.get("NrOfDataPoints", 0)
        instance.includes_constant = header.get("IncludesConstant", 0)
        instance.first_confound_predictor = header.get(
            "FirstConfoundPredictor", 0
        )

        nr_cols = instance.nr_of_predictors
        nr_rows = instance.nr_of_data_points

        # Column colors (line 5)
        colors = [int(x) for x in lines[5].split() if x.lstrip("-").isdigit()]
        col_rgb = [
            [colors[i], colors[i + 1], colors[i + 2]]
            for i in range(0, nr_cols * 3, 3)
        ]

        # Column names (line 6)
        names = [n.strip('"') for n in lines[6].split('" "') if n]

        # Column values (line 7+)
        values = np.zeros((nr_rows, nr_cols))
        for r, line in enumerate(lines[7 : 7 + nr_rows]):
            tmp = line.replace("e-", "!@#$%")
            tmp = tmp.replace("-", " -")
            tmp = tmp.replace("!@#$%", "e-")
            vals = [float(x) for x in tmp.split() if x]
            values[r, :] = vals

        predictors = []
        for i in range(nr_cols):
            predictors.append(SdmPredictor(
                name=names[i] if i < len(names) else "",
                color=col_rgb[i] if i < len(col_rgb) else [0, 0, 0],
                values=values[:, i].copy(),
            ))
        instance.predictors = predictors
        return instance

    def write(self, filename):
        with open(filename, "w") as f:
            f.write(f"FileVersion:                   {self.file_version}\n\n")
            f.write(f"NrOfPredictors:                {self.nr_of_predictors}\n")
            f.write(f"NrOfDataPoints:                {self.nr_of_data_points}\n")
            f.write(f"IncludesConstant:              {self.includes_constant}\n")
            f.write(f"FirstConfoundPredictor:        {self.first_confound_predictor}\n\n")

            preds = self.predictors or []
            for i, p in enumerate(preds):
                c = p.color or [0, 0, 0]
                f.write(f"{c[0]} {c[1]} {c[2]}")
                if i < len(preds) - 1:
                    f.write("   ")
            f.write("\n")

            for i, p in enumerate(preds):
                f.write(f'"{p.name}"')
                if i < len(preds) - 1:
                    f.write(" ")
            f.write("\n")

            nr_rows = self.nr_of_data_points
            for r in range(nr_rows):
                for j, p in enumerate(preds):
                    v = p.values[r] if p.values is not None else 0.0
                    f.write(f"{v:12.9f}")
                    if j < len(preds) - 1:
                        f.write(" ")
                f.write("\n")

    @classmethod
    def create_default(cls, nr_predictors=3, nr_data_points=50):
        sdm = cls()
        sdm.file_version = 1
        sdm.nr_of_predictors = nr_predictors
        sdm.nr_of_data_points = nr_data_points
        sdm.includes_constant = 0
        sdm.first_confound_predictor = 1

        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        preds = []
        for i in range(nr_predictors):
            preds.append(SdmPredictor(
                name=f"Predictor {i + 1}",
                color=colors[i % 3],
                values=np.random.random(nr_data_points),
            ))
        sdm.predictors = preds
        return sdm

    # -- Legacy ----------------------------------------------------------

    def to_legacy_dict(self):
        return {
            "FileVersion": self.file_version,
            "NrOfPredictors": self.nr_of_predictors,
            "NrOfDataPoints": self.nr_of_data_points,
            "IncludesConstant": self.includes_constant,
            "FirstConfoundPredictor": self.first_confound_predictor,
        }

    @classmethod
    def from_legacy_dict(cls, d, data=None):
        instance = cls()
        instance.file_version = d.get("FileVersion", 0)
        instance.nr_of_predictors = d.get("NrOfPredictors", 0)
        instance.nr_of_data_points = d.get("NrOfDataPoints", 0)
        instance.includes_constant = d.get("IncludesConstant", 0)
        instance.first_confound_predictor = d.get("FirstConfoundPredictor", 0)
        if data:
            preds = []
            for item in data:
                preds.append(SdmPredictor(
                    name=item.get("NameOfPredictor", ""),
                    color=item.get("ColorOfPredictor", [0, 0, 0]),
                    values=np.asarray(item.get("ValuesOfPredictor", [])),
                ))
            instance.predictors = preds
        return instance


def read_sdm(filename):
    sdm = SDM.read(filename)
    data = []
    for p in (sdm.predictors or []):
        data.append({
            "NameOfPredictor": p.name,
            "ColorOfPredictor": p.color,
            "ValuesOfPredictor": p.values,
        })
    return sdm.to_legacy_dict(), data


def write_sdm(filename, header, data_sdm):
    SDM.from_legacy_dict(header, data=data_sdm).write(filename)


def create_sdm():
    sdm = SDM.create_default()
    data = []
    for p in sdm.predictors:
        data.append({
            "NameOfPredictor": p.name,
            "ColorOfPredictor": p.color,
            "ValuesOfPredictor": p.values,
        })
    return sdm.to_legacy_dict(), data
