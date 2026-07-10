import json
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import VoiceFeatures, PredictionResponse

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# ─── App ─────────────────────────────────────────────────
app = FastAPI(
    title="Parkinson's Voice Screening API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load model files (ek baar startup par) ──────────────
model  = joblib.load(MODEL_DIR / "voice_model.pkl")
scaler = joblib.load(MODEL_DIR / "voice_scaler.pkl")

with open(MODEL_DIR / "feature_columns.json") as f:
    FEATURE_COLS = json.load(f)

with open(MODEL_DIR / "feature_importance.json") as f:
    FEATURE_IMPORTANCE = json.load(f)

# API field name  →  training column name
FIELD_TO_COL = {
    "mdvp_fo":          "MDVP:Fo(Hz)",
    "mdvp_fhi":         "MDVP:Fhi(Hz)",
    "mdvp_flo":         "MDVP:Flo(Hz)",
    "mdvp_jitter_pct":  "MDVP:Jitter(%)",
    "mdvp_jitter_abs":  "MDVP:Jitter(Abs)",
    "mdvp_rap":         "MDVP:RAP",
    "mdvp_ppq":         "MDVP:PPQ",
    "jitter_ddp":       "Jitter:DDP",
    "mdvp_shimmer":     "MDVP:Shimmer",
    "mdvp_shimmer_db":  "MDVP:Shimmer(dB)",
    "shimmer_apq3":     "Shimmer:APQ3",
    "shimmer_apq5":     "Shimmer:APQ5",
    "mdvp_apq":         "MDVP:APQ",
    "shimmer_dda":      "Shimmer:DDA",
    "nhr":              "NHR",
    "hnr":              "HNR",
    "rpde":             "RPDE",
    "dfa":              "DFA",
    "spread1":          "spread1",
    "spread2":          "spread2",
    "d2":               "D2",
    "ppe":              "PPE",
}

# ─── Endpoints ───────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Parkinson's API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/voice", response_model=PredictionResponse)
def predict_voice(features: VoiceFeatures):
    try:
        # Step 1: API data in correct order arrangement
        data = features.dict()
        col_map = {FIELD_TO_COL[k]: v for k, v in data.items()}
        X = np.array([[col_map[col] for col in FEATURE_COLS]])

        # Step 2: Scale  
        X_scaled = scaler.transform(X)

        # Step 3: Predict 
        prob_pd    = float(model.predict_proba(X_scaled)[0][1])
        pred_class = int(model.predict(X_scaled)[0])

        # Step 4: Confidence level
        if prob_pd >= 0.75 or prob_pd <= 0.25:
            confidence = "high"
        elif prob_pd >= 0.60 or prob_pd <= 0.40:
            confidence = "moderate"
        else:
            confidence = "low"

        return PredictionResponse(
            prediction  = "Parkinson's indicators detected"
                          if pred_class == 1
                          else "No significant indicators detected",
            risk_label  = "high_risk" if pred_class == 1 else "low_risk",
            probability_pd = round(prob_pd, 4),
            confidence  = confidence,
            disclaimer  = (
                "This is a research screening tool, "
                "NOT a medical diagnosis. "
                "Please consult a neurologist."
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    import shutil
from fastapi import UploadFile, File
from skimage.feature import hog

# Load spiral model (startup par)
spiral_model  = joblib.load(MODEL_DIR / "spiral_model.pkl")
spiral_scaler = joblib.load(MODEL_DIR / "spiral_scaler.pkl")
with open(MODEL_DIR / "spiral_evaluation.json") as f:
    SPIRAL_EVAL = json.load(f)

def extract_hog_features(image_bytes):
    """Image bytes se HOG features extract karo"""
    import cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

@app.post("/predict/spiral")
async def predict_spiral(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        features = extract_hog_features(image_bytes)
        X = np.array([features])
        X_scaled = spiral_scaler.transform(X)

        prob_pd    = float(spiral_model.predict_proba(X_scaled)[0][1])
        pred_class = int(spiral_model.predict(X_scaled)[0])

        if prob_pd >= 0.75 or prob_pd <= 0.25:
            confidence = "high"
        elif prob_pd >= 0.60 or prob_pd <= 0.40:
            confidence = "moderate"
        else:
            confidence = "low"

        return {
            "prediction": "Parkinson's indicators detected"
                          if pred_class == 1
                          else "No significant indicators detected",
            "risk_label":     "high_risk" if pred_class == 1 else "low_risk",
            "probability_pd": round(prob_pd, 4),
            "confidence":     confidence,
            "disclaimer":     "This is a research screening tool, NOT a medical diagnosis. Please consult a neurologist."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))