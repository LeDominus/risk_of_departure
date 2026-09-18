import pandas as pd
import numpy as np
import onnxruntime as ort
from src.config.config import TARGET_COLS

n_features = len(TARGET_COLS)
X_sample = pd.DataFrame(
    np.zeros((3, n_features), dtype=np.float32),
    columns=TARGET_COLS,
)

sess = ort.InferenceSession("D://Programming//RiskOfDeparture//tmp//mlflow_cache//logistic_regression.onnx")
print("inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

input_name = sess.get_inputs()[0].name
res = sess.run(None, {input_name: X_sample.to_numpy(dtype=np.float32)})
print("shapes:", [np.asarray(r).shape for r in res])
print("res[1]:", res[1][:3])