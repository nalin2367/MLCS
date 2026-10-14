import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd

# Test sample matching Step 13.7
sample = pd.DataFrame(
    [
        {
            "Age": 30,
            "Sex": "F",
            "BP": "NORMAL",
            "Cholesterol": "NORMAL",
            "Na_to_K": 15.0,
        }
    ]
)

# 1. Original scikit-learn pipeline
model = joblib.load("model/drug_model.joblib")
sklearn_pred = model.predict(sample)[0]
sklearn_proba = model.predict_proba(sample)[0]
print(f"sklearn: {sklearn_pred} {sklearn_proba}")

# 2. Local ONNX Runtime (simulating onnxruntime-web)
sess = rt.InferenceSession("model/drug_model.onnx")
feeds = {
    "Age": np.array([[30]], dtype=np.int64),
    "Sex": np.array([["F"]], dtype=object),
    "BP": np.array([["NORMAL"]], dtype=object),
    "Cholesterol": np.array([["NORMAL"]], dtype=object),
    "Na_to_K": np.array([[15.0]], dtype=np.float64),
}
label, probs = sess.run(None, feeds)
print(f"onnx: {label[0]} {probs}")

assert sklearn_pred == label[0], f"Prediction mismatch: {sklearn_pred} vs {label[0]}"
print("SUCCESS: Scikit-learn and ONNX predictions match perfectly!")
