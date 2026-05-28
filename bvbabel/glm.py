"""Read, write, create BrainVoyager GLM (general linear model) file format.

Typed API
---------
    glm = GLM.read("results.glm")
    print(glm.glm_type, glm.nr_time_points)
    print(glm.data_R2.shape, glm.data_beta.shape)
"""

import struct
import numpy as np
from bvbabel._binary_format import (
    Field, StringField, RGBField, DataField, ObjectField,
    TypedSubRecordListField, BinaryFormat, register_format,
)
from bvbabel.utils import read_variable_length_string, read_RGB_bytes


# =============================================================================
# Sub-records
# =============================================================================

class StudyInfo(BinaryFormat):
    """Metadata for a single study within a multi-study GLM."""

    nr_time_points_in_study = Field("<i")
    name_of_study_data = StringField()
    name_of_ssm = StringField(condition=lambda s: False)   # set manually
    name_of_sdm = StringField()


class PredictorInfo(BinaryFormat):
    """Metadata for a single predictor."""

    name_internal = StringField()
    name_custom = StringField()
    color = RGBField()
    # 9 unknown bytes follow — handled manually


# =============================================================================
# GLM format
# =============================================================================

@register_format(".glm")
class GLM(BinaryFormat):
    """Typed BrainVoyager GLM (general linear model results)."""

    # -- Header ----------------------------------------------------------
    file_version = Field("<h")
    glm_type = Field("<B")    # 0=FMR-STC, 1=VMR-VTC, 2=SRF-MTC
    rfx_glm = Field("<B")     # 0=std, 1=RFX

    # RFX fields
    nr_subjects = Field(
        "<i", condition=lambda s: s.rfx_glm == 1
    )
    nr_predictors_per_subject = Field(
        "<i", condition=lambda s: s.rfx_glm == 1
    )

    nr_time_points = Field("<i")
    nr_all_predictors = Field("<i")
    nr_confound_predictors = Field("<i")
    nr_studies = Field("<i")

    nr_studies_with_confound_info = Field(
        "<i", condition=lambda s: s.nr_studies > 1
    )
    # nr_confounds_per_study: variable-length int list — handled manually

    separate_predictors = Field("<B")
    time_course_normalization = Field("<B")
    resolution_multiplier = Field("<h")
    serial_correlation = Field("<B")
    mean_serial_correlation_before = Field("<f")
    mean_serial_correlation_after = Field("<f")

    # -- Type-dependent spatial fields -----------------------------------
    # FMR-STC
    dim_x = Field("<h", condition=lambda s: s.glm_type == 0)
    dim_y = Field("<h", condition=lambda s: s.glm_type == 0)
    dim_z = Field("<h", condition=lambda s: s.glm_type == 0)

    # VMR-VTC
    x_start = Field("<h", condition=lambda s: s.glm_type == 1)
    x_end = Field("<h", condition=lambda s: s.glm_type == 1)
    y_start = Field("<h", condition=lambda s: s.glm_type == 1)
    y_end = Field("<h", condition=lambda s: s.glm_type == 1)
    z_start = Field("<h", condition=lambda s: s.glm_type == 1)
    z_end = Field("<h", condition=lambda s: s.glm_type == 1)

    # SRF-MTC
    nr_vertices = Field("<i", condition=lambda s: s.glm_type == 2)

    cortex_based_mask = Field("<B")
    nr_voxels_in_mask = Field("<i")
    name_of_cortex_based_mask = StringField()

    # -- Study info list (read manually — SSM field depends on glm_type)
    studies = ObjectField(default_factory=list)

    # -- Predictor info list ---------------------------------------------
    predictors_info = TypedSubRecordListField(
        count_field="nr_all_predictors",
        record_cls=PredictorInfo,
        post_record_read=lambda f, p, r: f.read(9),
    )

    # -- Design matrix (non-RFX only) ------------------------------------
    # Handled in _post_read_overrides

    # -- Raw data (split into named arrays in _post_read) ----------------
    _data_raw = DataField(dtype="<f", shape_fields=("_data_size",))

    # -- I/O override for complex sections -------------------------------

    @classmethod
    def read(cls, filename, load_data=True):
        instance = cls()

        with open(filename, "rb") as f:
            # --- Standard fields until nr_studies spatial fields ---
            for field in cls._fields:
                if field.name == "studies":
                    break
                field.read(f, instance, load_data=load_data)

            # --- Studies (manual: SSM name conditional on glm_type) ---
            study_list = []
            for _ in range(instance.nr_studies):
                s = StudyInfo()
                s.nr_time_points_in_study, = struct.unpack("<i", f.read(4))
                s.name_of_study_data = read_variable_length_string(f)
                if instance.glm_type == 2:
                    s._values["name_of_ssm"] = read_variable_length_string(f)
                s.name_of_sdm = read_variable_length_string(f)
                study_list.append(s)
            instance._values["studies"] = study_list

            # --- Continue: predictor info ---
            # Find predictors_info position in _fields
            start = next(i for i, fld in enumerate(cls._fields) if fld.name == "predictors_info")
            for field in cls._fields[start:]:
                if field.name == "_data_raw":
                    break
                field.read(f, instance, load_data=load_data)

            # --- nr_confounds_per_study ---
            if instance.nr_studies > 1:
                confounds = []
                for _ in range(instance.nr_studies_with_confound_info):
                    v, = struct.unpack("<i", f.read(4))
                    confounds.append(v)
                instance._values["nr_confounds_per_study"] = confounds

            # 9 unknown bytes already skipped by post_record_read callback

            # --- Design matrix + inverted X'X (non-RFX) ---
            if instance.rfx_glm == 0:
                N = instance.nr_time_points
                M = instance.nr_all_predictors
                dm = np.zeros((N, M), dtype=np.float32)
                for j in range(N):
                    for k in range(M):
                        dm[j, k], = struct.unpack("<f", f.read(4))
                instance._values["design_matrix"] = dm

                inv = np.zeros((M, M), dtype=np.float32)
                for j in range(M):
                    for k in range(M):
                        inv[j, k], = struct.unpack("<f", f.read(4))
                instance._values["inverted_xx_matrix"] = inv

            # --- Data block ---
            data_field = next(
                fld for fld in cls._fields if fld.name == "_data_raw"
            )
            data_field.read(f, instance, load_data=load_data)

        instance._split_data()
        return instance

    # -- Computed properties ---------------------------------------------

    @property
    def _nr_data_points(self):
        if self.glm_type == 0:
            return self.dim_x * self.dim_y * self.dim_z
        elif self.glm_type == 1:
            r = max(self.resolution_multiplier, 1)
            rx = (self.x_end - self.x_start) // r
            ry = (self.y_end - self.y_start) // r
            rz = (self.z_end - self.z_start) // r
            return rx * ry * rz
        elif self.glm_type == 2:
            return self.nr_vertices
        return 0

    @property
    def _nr_maps(self):
        sc = self.serial_correlation
        P = self.nr_all_predictors
        if sc == 0:
            return 2 + 2 * P + 1
        elif sc == 1:
            return 2 + 2 * P + 2
        elif sc == 2:
            return 2 + 2 * P + 3
        return 2 + 2 * P + 1

    @property
    def _data_size(self):
        return self._nr_maps * self._nr_data_points

    # -- Data splitting --------------------------------------------------

    def _split_data(self):
        raw = self._values.get("_data_raw")
        if raw is None:
            return

        P = self.nr_all_predictors
        if self.glm_type == 2:
            # Surface: 2D (maps, vertices)
            maps = raw
        else:
            # Volume: reshape + BV → Tal
            nr_maps = self._nr_maps
            ndp = self._nr_data_points
            if self.glm_type == 0:
                z, y, x = self.dim_z, self.dim_y, self.dim_x
            else:
                r = max(self.resolution_multiplier, 1)
                z = (self.z_end - self.z_start) // r
                y = (self.y_end - self.y_start) // r
                x = (self.x_end - self.x_start) // r
            maps = raw.reshape(nr_maps, z, y, x)
            maps = np.transpose(maps, (1, 3, 2, 0))
            maps = maps[::-1, ::-1, ::-1, :]

        self._values["data_R2"] = maps[..., 0]
        self._values["data_SS"] = maps[..., 1]
        self._values["data_beta"] = maps[..., 2:2 + P]

        if maps.ndim >= 4:
            self._values["data_SS_XiY"] = maps[..., 2 + P:2 + 2 * P]
            self._values["data_meantc"] = np.squeeze(
                maps[..., 2 + 2 * P:2 + 2 * P + 1]
            )
            sc = self.serial_correlation
            if sc == 1:
                self._values["data_ARlag"] = maps[
                    ..., 2 + 2 * P + 1:2 + 2 * P + 2
                ]
            elif sc == 2:
                self._values["data_ARlag"] = maps[
                    ..., 2 + 2 * P + 1:2 + 2 * P + 3
                ]
            else:
                self._values["data_ARlag"] = np.zeros(
                    self._values["data_R2"].shape, dtype=np.float32
                )
        else:
            # Surface: 2D arrays
            self._values["data_SS_XiY"] = maps[2 + P:2 + 2 * P, :]
            self._values["data_meantc"] = maps[2 + 2 * P, :]
            sc = self.serial_correlation
            if sc == 1:
                self._values["data_ARlag"] = maps[2 + 2 * P + 1:2 + 2 * P + 2, :]
            elif sc == 2:
                self._values["data_ARlag"] = maps[2 + 2 * P + 1:2 + 2 * P + 3, :]
            else:
                self._values["data_ARlag"] = np.zeros(
                    self._values["data_R2"].shape, dtype=np.float32
                )

    # -- Data accessors --------------------------------------------------

    @property
    def data_R2(self):
        return self._values.get("data_R2")

    @property
    def data_SS(self):
        return self._values.get("data_SS")

    @property
    def data_beta(self):
        return self._values.get("data_beta")

    @property
    def data_SS_XiY(self):
        return self._values.get("data_SS_XiY")

    @property
    def data_meantc(self):
        return self._values.get("data_meantc")

    @property
    def data_ARlag(self):
        return self._values.get("data_ARlag")

    @property
    def design_matrix(self):
        return self._values.get("design_matrix")

    @property
    def inverted_xx_matrix(self):
        return self._values.get("inverted_xx_matrix")

    @property
    def nr_confounds_per_study(self):
        return self._values.get("nr_confounds_per_study", [])

    # -- Legacy ----------------------------------------------------------

    _LEGACY_MAP = {
        "file_version": "File version",
        "glm_type": "Type (0: FMR-STC, 1:VMR-VTC, 2:SRF-MTC",
        "rfx_glm": "RFX-GLM (0:std, 1:RFX)",
        "nr_subjects": "Nr subjects",
        "nr_predictors_per_subject": "Nr predictors per subject",
        "nr_time_points": "Nr time points",
        "nr_all_predictors": "Nr all predictors",
        "nr_confound_predictors": "Nr confound predictors",
        "nr_studies": "Nr studies",
        "nr_studies_with_confound_info": "Nr studies with confound info",
        "separate_predictors": "Separate predictors (0:no, 1:studies, 2:subjects)",
        "time_course_normalization": "Time course normalization (1:z transform, 2:baseline z, 3:percent change)",
        "resolution_multiplier": "Resolution multiplier (1, 2, 3 times VMR resolution)",
        "serial_correlation": "Serial correlation(0:no, 1:AR(1), 2:AR(2))",
        "mean_serial_correlation_before": "Mean serial correlation before correction",
        "mean_serial_correlation_after": "Mean serial correlation after correction",
        "dim_x": "DimX",
        "dim_y": "DimY",
        "dim_z": "DimZ",
        "x_start": "XStart",
        "x_end": "XEnd",
        "y_start": "YStart",
        "y_end": "YEnd",
        "z_start": "ZStart",
        "z_end": "ZEnd",
        "nr_vertices": "Nr vertices",
        "cortex_based_mask": "Cortex-based mask (1:(grey matter) mask has been used)",
        "nr_voxels_in_mask": "Nr voxels in mask",
        "name_of_cortex_based_mask": "Name of cortex-based mask",
        "_nr_maps": "Nr maps",
    }

    def to_legacy_dict(self):
        result = {}
        for py_name, legacy_name in self._LEGACY_MAP.items():
            val = getattr(self, py_name)
            if val is not None:
                result[legacy_name] = val
        # Study info
        result["Study info"] = []
        for s in (self.studies or []):
            result["Study info"].append({
                "Nr time points (volumes) in study": s.nr_time_points_in_study,
                "Name of study data": s.name_of_study_data,
                "Name of SSM": getattr(s, "name_of_ssm", ""),
                "Name of SDM": s.name_of_sdm,
            })
        # Predictor info
        result["Predictor info"] = []
        for p in (self.predictors_info or []):
            result["Predictor info"].append({
                "Name (internal)": p.name_internal,
                "Name (custom)": p.name_custom,
                "Color": p.color,
            })
        result["Design matrix"] = self.design_matrix
        result["Inverted X'X matrix"] = self.inverted_xx_matrix
        result["Nr confounds per study"] = list(self.nr_confounds_per_study)
        return result


def read_glm(filename):
    glm = GLM.read(filename)
    return (
        glm.to_legacy_dict(),
        glm.data_R2,
        glm.data_SS,
        glm.data_beta,
        glm.data_SS_XiY,
        glm.data_meantc,
        glm.data_ARlag,
    )
