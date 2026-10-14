# MLCS — Drug Classification ML CI/CD Pipeline

[![CI/CD Pipeline](https://github.com/nalin2367/MLCS/actions/workflows/ci.yml/badge.svg)](https://github.com/nalin2367/MLCS/actions)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime-WebAssembly-blue)](https://onnxruntime.ai/)

A complete end-to-end Machine Learning Continuous Integration & Continuous Deployment (CI/CD) implementation using **GitHub Actions**, **CML (Continuous Machine Learning)**, and a zero-server-compute **ONNX Runtime Web** static inference app deployed to **Hugging Face Static Spaces**.

---

## 🌟 Key Features

1. **Automated Continuous Integration (CI):**
   - Automatically triggered on `push` and `pull_request` to `main`.
   - Code formatting verification via `black --check`.
   - Automated model retraining on `data/drug200.csv` using Scikit-Learn.
   - Model export to both **Joblib** (reproducibility) and **ONNX** (portable web runtime).
   - Equivalence test verifying identical outputs between Scikit-learn and ONNX Runtime.

2. **Automated Reporting with CML:**
   - Automatically compiles evaluation metrics (Accuracy, Weighted F1, Classification Report).
   - Generates and embeds the confusion matrix plot directly into GitHub Pull Request comments and workflow summaries.

3. **Continuous Deployment (CD) to Hugging Face Static Spaces:**
   - Automatically synchronizes the static web app (`app/`), trained model artifacts (`model/`), and evaluation metrics (`results/`) to Hugging Face Spaces using `huggingface_hub`.
   - Hosted on a **100% free Hugging Face Static Space** with **zero server compute requirements**.

4. **Client-Side Browser Inference:**
   - Model inference executes entirely inside the user's browser using **ONNX Runtime Web (`ort-web`)** via WebAssembly/WebGL.
   - Preprocessing (`OneHotEncoder`) and classification (`RandomForestClassifier`) are compiled directly into the ONNX graph—no server-side Python required at runtime.
   - Immediate UI rendering with zero Pyodide/Gradio cold-start delays.

---

## 📁 Repository Structure

```
ml-cicd-gradio-lite/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD & CML pipeline
├── app/
│   ├── index.html                 # Production ONNX Runtime Web client application
│   └── README.md                  # Hugging Face Space configuration and metadata
├── data/
│   └── drug200.csv                # Training dataset (Age, Sex, BP, Cholesterol, Na_to_K, Drug)
├── model/
│   ├── drug_model.joblib          # Scikit-learn trained model artifact
│   └── drug_model.onnx            # Exported ONNX graph for browser inference
├── results/
│   ├── metrics.txt                # Accuracy, F1-score, and classification report
│   └── confusion_matrix.png       # Confusion matrix visualization
├── train.py                       # Training, evaluation, and ONNX conversion script
├── test_predict.py                # Verification script comparing Sklearn vs ONNX
├── requirements.txt               # CI environment dependencies
├── Makefile                       # Developer task runner
└── README.md                      # Project documentation
```

---

## 🚀 Quick Start & Local Development

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.11 or 3.12)
- Git

### 2. Setup Virtual Environment & Install Dependencies
```bash
# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.\.venv\Scripts\activate

# Install all dependencies
make install
# or: pip install -r requirements.txt
```

### 3. Verify Code Formatting
```bash
make format
# or: black --check train.py
```

### 4. Train Model and Export ONNX
```bash
make train
# or: python train.py
```
This fits the pipeline, produces `model/drug_model.joblib` and `model/drug_model.onnx`, and saves performance metrics in `results/`.

### 5. Verify Prediction Consistency
```bash
make test
# or: python test_predict.py
```
Ensures that Scikit-learn and ONNX Runtime return identical predictions and confidence scores for identical feature feeds.

### 6. Run Web App Locally
Serve the repository root using any static web server:
```bash
python -m http.server 8000
```
Open [http://localhost:8000/app/index.html](http://localhost:8000/app/index.html) in your browser. The app will fetch `model/drug_model.onnx` and execute inference locally in WebAssembly.

---

## ⚙️ Hugging Face & GitHub Actions Setup

### Step 1: Create a Hugging Face Static Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Set your Space name (e.g. `drug-classification-lite`).
4. Select **Static** (or **Static HTML**) as the Space SDK (do **not** select Docker or Gradio SDK).
5. Copy your Space ID: `YOUR_USERNAME/drug-classification-lite`.

### Step 2: Configure GitHub Repository Secrets
In your GitHub repository, navigate to:
**Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add the following secrets:
| Secret Name | Value | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | `hf_...` | Hugging Face user access token with **write** permissions ([Generate Token](https://huggingface.co/settings/tokens)). |
| `HF_SPACE_ID` | `YOUR_USERNAME/drug-classification-lite` | Your exact Hugging Face username and Static Space name. |

> **Note:** GitHub's built-in `GITHUB_TOKEN` is automatically used by CML to post evaluation reports on Pull Requests.

---

## 🔄 CI/CD Pipeline Workflow

When code is pushed to `main` or a Pull Request is opened:
1. **Source Checkout & Environment Setup:** Checks out the repository and sets up Python 3.11 and CML.
2. **Linting & Quality Gate:** Executes `make format` to verify PEP 8 compliance.
3. **Training & Artifact Generation:** Runs `make train` to fit the Random Forest classifier and compile the ONNX graph.
4. **Validation Test:** Runs `make test` to verify numerical alignment across runtimes.
5. **CML Reporting:** Generates `report.md` with evaluation metrics and confusion matrix plot, commenting directly on the PR or commit.
6. **Continuous Deployment (Pushes to `main` only):**
   - Uploads `app/` contents to the Hugging Face Space root.
   - Uploads `model/` to `<space>/model`.
   - Uploads `results/` to `<space>/results`.

---

## 📊 Model Evaluation Results

- **Model Type:** Scikit-Learn Pipeline (`OneHotEncoder` + `RandomForestClassifier(n_estimators=100)`)
- **Evaluation Set:** Stratified 20% hold-out test set
- **Accuracy:** 97.50%
- **Weighted F1-Score:** 97.36%

---

## ⚠️ Medical Disclaimer

This project is an **educational demonstration** of automated Machine Learning CI/CD, ONNX model serialization, and static browser-side inference. It is **not** a clinical or medical decision support tool. Never make medical or pharmacological decisions based on its predictions.
