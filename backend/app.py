# ============================================================
# ALZHEIMER'S DETECTION — FASTAPI BACKEND
# Run with: uvicorn app:app --reload
# Listens at: http://localhost:8000
# ============================================================

# WHY these imports:
# fastapi      → the web framework itself
# BaseModel    → lets us define exactly what data shape we expect
# joblib       → loads our saved .pkl model file
# pandas       → model expects a DataFrame, not raw dict
# numpy        → handles the probability array the model returns
# CORSMiddleware → allows our frontend (different port) to talk to backend

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# ============================================================
# STEP 1: Create the FastAPI app
# WHY: This one line creates our entire web application.
# Everything else just configures and adds to it.
# ============================================================
app = FastAPI(title="AlzAI API", version="1.0")

# ============================================================
# STEP 2: CORS — Cross Origin Resource Sharing
# WHY: Your frontend runs at localhost:5500 (Live Server)
#      Your backend runs at localhost:8000 (FastAPI)
# Browsers BLOCK requests between different ports by default
# for security reasons. CORS tells the browser:
# "it's okay, I allow this frontend to talk to me"
# Without this, every fetch() call would be blocked.
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # allow ALL origins (fine for local dev)
    allow_methods=["*"],    # allow GET, POST, etc.
    allow_headers=["*"],    # allow all headers
)

# ============================================================
# STEP 3: Load the trained model ONCE at startup
# WHY: We load it here (outside any function) so it loads
# ONCE when the server starts — not every time someone
# sends a request. Loading a .pkl file takes ~0.5 seconds.
# If we loaded it per-request, every prediction would be slow.
# ============================================================
MODEL_PATH = "model/alzheimer_model.pkl"
model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

# ============================================================
# STEP 4: Define the data shape we expect from the frontend
# WHY: BaseModel is like a contract. It says exactly what
# fields the POST request must contain and what type each is.
# FastAPI automatically validates incoming data against this.
# If the frontend sends wrong data, FastAPI rejects it with
# a clear error — we never even reach our prediction code.
# The field names MUST match the model's training column names.
# ============================================================
class PatientData(BaseModel):
    Age:                       float
    Gender:                    float
    Ethnicity:                 float
    EducationLevel:            float
    BMI:                       float
    Smoking:                   float
    AlcoholConsumption:        float
    PhysicalActivity:          float
    DietQuality:               float
    SleepQuality:              float
    FamilyHistoryAlzheimers:   float
    CardiovascularDisease:     float
    Diabetes:                  float
    Depression:                float
    HeadInjury:                float
    Hypertension:              float
    SystolicBP:                float
    DiastolicBP:               float
    CholesterolTotal:          float
    CholesterolLDL:            float
    CholesterolHDL:            float
    CholesterolTriglycerides:  float
    MMSE:                      float
    FunctionalAssessment:      float
    MemoryComplaints:          float
    BehavioralProblems:        float
    ADL:                       float
    Confusion:                 float
    Disorientation:            float
    PersonalityChanges:        float
    DifficultyCompletingTasks: float
    Forgetfulness:             float

# ============================================================
# STEP 5: The health check endpoint
# WHY: A simple GET route at "/" so we can confirm the
# server is running. Visit localhost:8000 in browser
# and you should see {"status": "AlzDetect API is running"}
# This is standard practice in every real backend.
# ============================================================
@app.get("/")
def root():
    return {"status": "AlzAI API is running"}

# ============================================================
# STEP 6: The prediction endpoint — the heart of the backend
# WHY: @app.post("/predict") means: when a POST request
# arrives at localhost:8000/predict, run this function.
# FastAPI automatically parses the JSON body into PatientData.
# ============================================================
@app.post("/predict")
def predict(patient: PatientData):

    # --- Convert incoming data to a DataFrame ---
    # WHY: Our model was trained on a pandas DataFrame.
    # It expects a DataFrame, not a dictionary.
    # dict(patient) turns PatientData object → Python dict
    # pd.DataFrame([...]) wraps it in a single-row DataFrame
    # The column order is automatically correct because
    # PatientData field names match training column names exactly
    patient_df = pd.DataFrame([dict(patient)])

    # --- Run prediction ---
    # WHY: predict() returns the class label: 0 or 1
    # predict_proba() returns probabilities: [[0.91, 0.09]]
    # [0] gets the first (only) row
    prediction  = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0]

    # probability is an array like [0.91, 0.09]
    # index 0 = probability of class 0 (healthy)
    # index 1 = probability of class 1 (sick)
    prob_healthy = float(probability[0])
    prob_sick    = float(probability[1])

    # --- Log to terminal so we can see what's happening ---
    # WHY: Every time a prediction is made, print it in the
    # terminal. Invaluable for debugging during development.
    result_label = "ALZHEIMER'S DETECTED" if prediction == 1 else "No Alzheimer's"
    print(f"\n📋 Prediction request received")
    print(f"   Result:      {result_label}")
    print(f"   Prob Healthy: {prob_healthy:.2%}")
    print(f"   Prob Sick:    {prob_sick:.2%}")

    # --- Return JSON response to frontend ---
    # WHY: FastAPI automatically converts this dict to JSON.
    # Frontend's fetch() receives exactly this object and
    # uses result.prediction, result.prob_healthy, result.prob_sick
    return {
        "prediction":  int(prediction),
        "prob_healthy": prob_healthy,
        "prob_sick":    prob_sick,
    }