"""Typed read/write for GLM (general linear model)."""
import sys
from bvbabel.glm import GLM

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/example.glm"

glm = GLM.read(path)
print(f"type={glm.glm_type}  predictors={glm.nr_all_predictors}  TP={glm.nr_time_points}")
print(f"R²={glm.data_R2.shape}  beta={glm.data_beta.shape}  SS={glm.data_SS.shape}")
if glm.design_matrix is not None:
    print(f"design matrix={glm.design_matrix.shape}")
for p in glm.predictors_info:
    print(f"  predictor: {p.name_custom}  color={p.color}")
