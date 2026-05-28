"""Read, write, create BrainVoyager PRT (protocol) file format."""

import numpy as np
from bvbabel._binary_format import (
    Field, ObjectField, Section, BinaryFormat, register_format,
)


class PrtCondition(Section):
    """A single experimental condition in a protocol."""

    name = Field(default="")
    nr_of_occurrences = Field(default=0)
    time_start = Field(default=None)      # int array
    time_stop = Field(default=None)       # int array
    parametric_weight = Field(default=None)  # float array
    color = Field(default=None)           # [R, G, B] int array


@register_format(".prt")
class PRT(BinaryFormat):
    """Typed BrainVoyager PRT (stimulation protocol)."""

    file_version = Field(default="")
    resolution_of_time = Field(default="Volumes")
    experiment = Field(default="")
    background_color = Field(default="")
    text_color = Field(default="")
    time_course_color = Field(default="")
    time_course_thick = Field(default="")
    reference_func_color = Field(default="")
    reference_func_thick = Field(default="")
    parametric_weights = Field(default=0)
    nr_of_conditions = Field(default=0)

    conditions = ObjectField(default_factory=list)

    @classmethod
    def read(cls, filename, load_data=True):
        with open(filename, "r") as f:
            lines = [r for r in (line.strip().replace("\t", " ") for line in f) if r]

        instance = cls()
        header = {}
        header_rows = 0

        for j, line in enumerate(lines):
            parts = line.split(":", 1)
            parts = [p.strip() for p in parts]
            if len(parts) < 2 or parts[0].isdigit():
                continue

            key, val = parts[0], parts[1]
            if key == "FileVersion":
                instance.file_version = val
            elif key == "ResolutionOfTime":
                instance.resolution_of_time = val
            elif key == "Experiment":
                instance.experiment = val
            elif key == "BackgroundColor":
                instance.background_color = val
            elif key == "TextColor":
                instance.text_color = val
            elif key == "TimeCourseColor":
                instance.time_course_color = val
            elif key == "TimeCourseThick":
                instance.time_course_thick = val
            elif key == "ReferenceFuncColor":
                instance.reference_func_color = val
            elif key == "ReferenceFuncThick":
                instance.reference_func_thick = val
            elif key == "ParametricWeights":
                instance.parametric_weights = int(val)
            elif key == "NrOfConditions":
                instance.nr_of_conditions = int(val)
                header_rows = j + 1
                break

        # Parse conditions
        conditions = []
        i = header_rows
        while i < len(lines) and len(conditions) < instance.nr_of_conditions:
            cond = PrtCondition()
            cond.name = lines[i]
            n = int(lines[i + 1])
            cond.nr_of_occurrences = n

            t_start = np.zeros(n, dtype=int)
            t_stop = np.zeros(n, dtype=int)
            pw = np.zeros(n, dtype=float) if instance.parametric_weights > 0 else None

            for j in range(n):
                vals = lines[i + 2 + j].split()
                if instance.resolution_of_time == "Seconds":
                    t_start[j] = int(float(vals[0]) * 1000.0)
                    t_stop[j] = int(float(vals[1]) * 1000.0)
                else:
                    t_start[j] = int(vals[0])
                    t_stop[j] = int(vals[1])
                if pw is not None:
                    pw[j] = float(vals[2])

            cond.time_start = t_start
            cond.time_stop = t_stop
            if pw is not None:
                cond.parametric_weight = pw

            color_vals = [int(v) for v in lines[i + 2 + n].split() if v.lstrip("-").isdigit()]
            cond.color = np.array(color_vals[:3])
            conditions.append(cond)
            i += n + 3

        if instance.resolution_of_time == "Seconds":
            instance.resolution_of_time = "msec"
        instance.conditions = conditions
        return instance

    def write(self, filename):
        with open(filename, "w") as f:
            f.write(f"\nFileVersion:        {self.file_version}\n\n")
            f.write(f"ResolutionOfTime:   {self.resolution_of_time}\n\n")
            f.write(f"Experiment:         {self.experiment}\n\n")
            f.write(f"BackgroundColor:    {self.background_color}\n")
            f.write(f"TextColor:          {self.text_color}\n")
            f.write(f"TimeCourseColor:    {self.time_course_color}\n")
            f.write(f"TimeCourseThick:    {self.time_course_thick}\n")
            f.write(f"ReferenceFuncColor: {self.reference_func_color}\n")
            f.write(f"ReferenceFuncThick: {self.reference_func_thick}\n\n")
            if self.parametric_weights:
                f.write(f"ParametricWeights: {self.parametric_weights}\n\n")
            f.write(f"NrOfConditions: {self.nr_of_conditions}\n")

            for cond in (self.conditions or []):
                f.write(f"\n{cond.name}\n")
                f.write(f"{cond.nr_of_occurrences}\n")
                for j in range(cond.nr_of_occurrences):
                    v1 = cond.time_start[j] if cond.time_start is not None else 0
                    v2 = cond.time_stop[j] if cond.time_stop is not None else 0
                    if self.resolution_of_time.lower() == "volumes":
                        v1, v2 = int(v1), int(v2)
                    if cond.parametric_weight is not None:
                        f.write(f"{v1:>4} {v2:>4} {cond.parametric_weight[j]}\n")
                    else:
                        f.write(f"{v1:>4} {v2:>4}\n")
                c = cond.color if cond.color is not None else [0, 0, 0]
                f.write(f"Color: {c[0]} {c[1]} {c[2]}\n")


def read_prt(filename):
    prt = PRT.read(filename)
    h = {
        "FileVersion": prt.file_version,
        "ResolutionOfTime": prt.resolution_of_time,
        "Experiment": prt.experiment,
        "BackgroundColor": prt.background_color,
        "TextColor": prt.text_color,
        "TimeCourseColor": prt.time_course_color,
        "TimeCourseThick": prt.time_course_thick,
        "ReferenceFuncColor": prt.reference_func_color,
        "ReferenceFuncThick": prt.reference_func_thick,
        "ParametricWeights": prt.parametric_weights,
        "NrOfConditions": prt.nr_of_conditions,
    }
    data = []
    for c in (prt.conditions or []):
        d = {
            "NameOfCondition": c.name,
            "NrOfOccurances": c.nr_of_occurrences,
            "Time start": c.time_start,
            "Time stop": c.time_stop,
            "Color": c.color,
        }
        if c.parametric_weight is not None:
            d["Parametric weight"] = c.parametric_weight
        data.append(d)
    return h, data


def write_prt(filename, header, data_prt):
    prt = PRT()
    for k, v in header.items():
        py = k[0].lower() + k[1:] if k else k
        py = {"NrOfConditions": "nr_of_conditions"}.get(k, py)
        if hasattr(prt, py):
            setattr(prt, py, v)
    conds = []
    for d in data_prt:
        c = PrtCondition()
        c.name = d.get("NameOfCondition", "")
        c.nr_of_occurrences = d.get("NrOfOccurances", 0)
        c.time_start = d.get("Time start")
        c.time_stop = d.get("Time stop")
        c.color = d.get("Color")
        c.parametric_weight = d.get("Parametric weight")
        conds.append(c)
    prt.conditions = conds
    prt.write(filename)
