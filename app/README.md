---
title: Drug Classification - ML CI/CD
colorFrom: blue
colorTo: green
sdk: static
pinned: false
---

# Drug Classification — Static Web Inference (ONNX Runtime Web)

This Hugging Face Static Space runs client-side machine learning inference directly inside the browser using ONNX Runtime Web.

- **Model:** Scikit-Learn Pipeline (`OneHotEncoder` + `RandomForestClassifier`) converted to ONNX via `skl2onnx`.
- **Inference Runtime:** `onnxruntime-web` (WebAssembly backend, zero Python server required).
- **Automation:** GitHub Actions + CML automated model retraining, evaluation report generation, and continuous deployment to Hugging Face Spaces.

### Inputs
- **Age:** Patient age in years
- **Sex:** F or M
- **Blood Pressure (BP):** LOW, NORMAL, HIGH
- **Cholesterol:** NORMAL, HIGH
- **Na_to_K:** Sodium to Potassium ratio in blood

### Outputs
- **Predicted Drug:** `drugY`, `drugA`, `drugB`, `drugC`, or `drugX`
- **Model Confidence:** Prediction probability score
