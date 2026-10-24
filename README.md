# 🎗️ Breast Cancer Prediction using Machine Learning

## 📌 Project Overview

This project uses Machine Learning to classify breast cancer tumors as **Benign (B)** or **Malignant (M)** based on cell nucleus measurements from the Breast Cancer Wisconsin Diagnostic dataset.

The project includes data analysis, model training, a FastAPI REST API, and a Streamlit frontend for interactive predictions.

**Note:** This project is for educational purposes only and is not intended for medical diagnosis.

## 🎯 Objectives

- Analyze breast cancer data using Exploratory Data Analysis (EDA).
- Train and evaluate classification models.
- Use Support Vector Machine (SVM) for prediction.
- Build a REST API using FastAPI.
- Develop an interactive interface using Streamlit.
- Display predictions and model probability outputs.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Plotly
- FastAPI
- Streamlit
- Uvicorn
- Joblib
- Jupyter Notebook

## 📊 Dataset

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

The dataset contains measurements describing cell nuclei, including:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

The model uses 30 numerical input features.

**Target classes:**
- `B` — Benign
- `M` — Malignant

## 🤖 Machine Learning Workflow

1. Load and inspect the dataset.
2. Perform data cleaning and exploratory data analysis.
3. Prepare features and target labels.
4. Split the dataset into training and testing sets.
5. Train classification models.
6. Evaluate model performance.
7. Save the trained model using Joblib.
8. Build a prediction API with FastAPI.
9. Connect the API to a Streamlit frontend.

## 📈 Models Evaluated

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

SVM was selected for the current API implementation based on the evaluation results from this project.

## 📁 Project Structure

```text
Breast_Cancer_Prediction_ML_Project/
│
├── main.py
├── app.py
├── breast_cancer_model.joblib
├── scaler.joblib
├── requirements.txt
├── breast_cancer_wisconsin_diagnostic.csv
└── notebook.ipynb
```

*The `scaler.joblib` file is required only if the saved model was trained using a fitted scaler. Update the filenames above to match your actual project files.*

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Breast_Cancer_Prediction_ML_Project
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Run the Application

### Start FastAPI

```bash
uvicorn main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

### Start Streamlit

Open a second terminal and run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

## 🔌 API Endpoint

### POST `/predict`

Accepts 30 numerical features and returns the model's predicted class and, when supported, its class probabilities.

Example response for a Benign prediction:

```json
{
  "prediction": "B",
  "result": "Benign",
  "message": "Model prediction: Benign",
  "probabilities": {
    "0": 0.9923,
    "1": 0.0077
  }
}
```

The values above are an example response from a test run, not a guarantee for every input.

## 📋 Example Results

The API was tested with example inputs for both classes:

- **Benign example:** predicted class `B`.
- **Malignant example:** predicted class `M`.

Model outputs depend on the input values and the saved model.

## 🔒 Medical Disclaimer

This project is intended for learning and demonstration. It is not a clinically validated diagnostic tool. Do not use its predictions to diagnose a condition or make treatment decisions. Consult a qualified healthcare professional for medical concerns.

## 👩‍💻 Author

**Ramsha Noreen**

Aspiring Machine Learning Engineer | Python | Scikit-learn | FastAPI | Streamlit

## 🌱 Future Improvements

- Deploy the API and frontend online.
- Improve input validation and error handling.
- Add model evaluation visualizations.
- Write automated API tests.
- Improve the interface and accessibility.

## ⭐ Acknowledgements

Thanks to the contributors and maintainers of the Breast Cancer Wisconsin Diagnostic dataset and the open-source Python libraries used in this project.
