from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Breast Cancer classification using a trained SVM model.",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "breast_cancer_model.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Load scaler only if your training workflow used one.
scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None

FEATURES = [
    "radius1", "texture1", "perimeter1", "area1",
    "smoothness1", "compactness1", "concavity1",
    "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2",
    "smoothness2", "compactness2", "concavity2",
    "concave_points2", "symmetry2", "fractal_dimension2",
    "radius3", "texture3", "perimeter3", "area3",
    "smoothness3", "compactness3", "concavity3",
    "concave_points3", "symmetry3", "fractal_dimension3"
]

class BreastCancerInput(BaseModel):
    radius1: float = Field(gt=0)
    texture1: float = Field(gt=0)
    perimeter1: float = Field(gt=0)
    area1: float = Field(gt=0)
    smoothness1: float = Field(gt=0)
    compactness1: float = Field(ge=0)
    concavity1: float = Field(ge=0)
    concave_points1: float = Field(ge=0)
    symmetry1: float = Field(gt=0)
    fractal_dimension1: float = Field(gt=0)

    radius2: float = Field(gt=0)
    texture2: float = Field(gt=0)
    perimeter2: float = Field(gt=0)
    area2: float = Field(gt=0)
    smoothness2: float = Field(gt=0)
    compactness2: float = Field(ge=0)
    concavity2: float = Field(ge=0)
    concave_points2: float = Field(ge=0)
    symmetry2: float = Field(gt=0)
    fractal_dimension2: float = Field(gt=0)

    radius3: float = Field(gt=0)
    texture3: float = Field(gt=0)
    perimeter3: float = Field(gt=0)
    area3: float = Field(gt=0)
    smoothness3: float = Field(gt=0)
    compactness3: float = Field(ge=0)
    concavity3: float = Field(ge=0)
    concave_points3: float = Field(ge=0)
    symmetry3: float = Field(gt=0)
    fractal_dimension3: float = Field(gt=0)

@app.get("/")
def home():
    return {
        "message": "Breast Cancer Prediction API is running",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict")
def predict(data: BreastCancerInput):
    try:
        input_data = pd.DataFrame(
            [data.model_dump()],
            columns=FEATURES
        )

        # Use the same fitted scaler as during training, if applicable.
        processed_data = (
            scaler.transform(input_data)
            if scaler is not None
            else input_data
        )

        prediction = model.predict(processed_data)[0]

        # Convert model prediction to the correct class label.
        if prediction in (0, 1):
            label = "B" if int(prediction) == 0 else "M"
        elif str(prediction) in ("B", "M"):
            label = str(prediction)
        else:
            raise ValueError(f"Unexpected model label: {prediction}")

        result = "Benign" if label == "B" else "Malignant"
        response = {
            "prediction": label,
            "result": result,
            "message": f"Model prediction: {result}"
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(processed_data)[0]
            response["probabilities"] = {
                str(cls): round(float(prob), 4)
                for cls, prob in zip(model.classes_, probabilities)
            }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000,reload=True)
