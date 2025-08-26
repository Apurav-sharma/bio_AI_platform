# aegis_complete_api.py
# Complete AEGIS Cancer Risk Prediction API - Single File Solution
# pip install fastapi uvicorn pandas numpy scikit-learn scipy pydantic

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import uvicorn

# --------- LOGGING SETUP ---------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------- CONFIG ---------
ESSENTIAL_FEATURES = [
    'TP53', 'BRCA1', 'EGFR', 'MYC',  # Genetic biomarkers
    'age', 'bmi', 'smoking_history', 'family_history', 
    'previous_cancer_history', 'inflammatory_markers'  # Clinical factors
]

# --------- PYDANTIC MODELS ---------
class PatientInput(BaseModel):
    """Input model for patient data"""
    
    # Genetic biomarkers (expression values)
    TP53: float = Field(..., ge=0.0, le=15.0, description="TP53 gene expression level")
    BRCA1: float = Field(..., ge=0.0, le=15.0, description="BRCA1 gene expression level") 
    EGFR: float = Field(..., ge=0.0, le=15.0, description="EGFR gene expression level")
    MYC: float = Field(..., ge=0.0, le=15.0, description="MYC gene expression level")
    
    # Clinical factors
    age: float = Field(..., ge=18.0, le=120.0, description="Patient age in years")
    bmi: float = Field(..., ge=12.0, le=50.0, description="Body Mass Index")
    smoking_history: int = Field(..., ge=0, le=1, description="Smoking history (0=No, 1=Yes)")
    family_history: int = Field(..., ge=0, le=1, description="Family cancer history (0=No, 1=Yes)")
    previous_cancer_history: int = Field(..., ge=0, le=1, description="Previous cancer diagnosis (0=No, 1=Yes)")
    inflammatory_markers: float = Field(..., ge=0.0, le=20.0, description="Inflammatory markers level")

    class Config:
        schema_extra = {
            "example": {
                "TP53": 4.2, "BRCA1": 4.5, "EGFR": 5.8, "MYC": 6.2,
                "age": 55.0, "bmi": 26.5, "smoking_history": 1,
                "family_history": 0, "previous_cancer_history": 0,
                "inflammatory_markers": 2.1
            }
        }

class BatchPatientInput(BaseModel):
    """Input model for batch predictions"""
    patients: List[PatientInput] = Field(..., max_items=100)

class RiskPrediction(BaseModel):
    """Output model for risk prediction"""
    patient_id: Optional[str] = None
    cancer_risk_percentage: float
    risk_category: str
    survival_months: float
    survival_years: float
    confidence_score: float

class BatchRiskPrediction(BaseModel):
    """Output model for batch predictions"""
    predictions: List[RiskPrediction]
    batch_stats: Dict

# --------- DATA GENERATION AND TRAINING ---------
def generate_training_data(n_samples=3000, seed=42):
    """Generate synthetic biomedical training data"""
    rng = np.random.default_rng(seed)
    
    # Generate gene expression data
    gene_data = {
        'TP53': rng.normal(4.2, 1.2, n_samples),
        'BRCA1': rng.normal(4.5, 1.0, n_samples),
        'EGFR': rng.normal(5.8, 1.5, n_samples),
        'MYC': rng.normal(6.2, 1.3, n_samples)
    }
    
    # Generate clinical data
    ages = rng.normal(55, 15, n_samples).clip(18, 90)
    clinical_data = {
        'age': ages,
        'bmi': rng.normal(26, 4, n_samples).clip(15, 45),
        'smoking_history': rng.binomial(1, 0.3, n_samples),
        'family_history': rng.binomial(1, 0.15, n_samples),
        'previous_cancer_history': rng.binomial(1, 0.05, n_samples),
        'inflammatory_markers': rng.exponential(2, n_samples)
    }
    
    # Combine features
    df = pd.DataFrame({**gene_data, **clinical_data})
    
    # Generate cancer labels based on biological relationships
    risk_score = np.zeros(n_samples)
    
    # Genetic factors
    risk_score += (4.5 - df['TP53']) * 0.25      # TP53 suppressor
    risk_score += (4.5 - df['BRCA1']) * 0.20     # BRCA1 suppressor
    risk_score += (df['EGFR'] - 5.5) * 0.20      # EGFR oncogene
    risk_score += (df['MYC'] - 6.0) * 0.18       # MYC oncogene
    
    # Clinical factors
    risk_score += df['age'] * 0.03
    risk_score += df['smoking_history'] * 1.2
    risk_score += df['family_history'] * 1.5
    risk_score += df['previous_cancer_history'] * 2.0
    risk_score += df['bmi'] * 0.08
    risk_score += df['inflammatory_markers'] * 0.15
    
    # Add noise and convert to probabilities
    risk_score += rng.normal(0, 1.2, n_samples)
    probabilities = 1 / (1 + np.exp(-risk_score + 2.5))
    labels = rng.binomial(1, probabilities, n_samples)
    
    # Generate survival data
    base_survival = rng.exponential(240, n_samples)
    survival_months = base_survival.copy()
    
    # Cancer patients have reduced survival
    cancer_mask = labels.astype(bool)
    if cancer_mask.any():
        cancer_survival = rng.exponential(60, cancer_mask.sum())
        survival_months[cancer_mask] = cancer_survival
    
    # Adjust based on clinical factors
    age_factor = (df['age'] - 40) * 0.8
    survival_months -= age_factor
    survival_months -= df['smoking_history'] * 80
    survival_months -= df['previous_cancer_history'] * 60
    survival_months -= df['inflammatory_markers'] * 5
    
    # BMI effect (U-shaped)
    bmi_deviation = np.abs(df['bmi'] - 22)
    survival_months -= bmi_deviation * 2
    
    survival_months = np.maximum(survival_months, 6)
    
    df['cancer_diagnosis'] = labels
    df['survival_months'] = survival_months
    
    return df

def train_models():
    """Train cancer prediction and survival models"""
    logger.info("Generating training data...")
    df = generate_training_data(n_samples=3000, seed=42)
    
    # Prepare features
    X = df[ESSENTIAL_FEATURES].copy()
    y = df['cancer_diagnosis']
    survival = df['survival_months']
    
    # Log transform gene expression data
    gene_cols = ['TP53', 'BRCA1', 'EGFR', 'MYC']
    for col in gene_cols:
        X[col] = np.log2(X[col].clip(lower=0.1) + 1)
    
    logger.info("Training cancer prediction model...")
    
    # Train cancer model
    cancer_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=10,
            min_samples_leaf=5, random_state=42, class_weight='balanced'
        ))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    cancer_pipeline.fit(X_train, y_train)
    
    # Evaluate cancer model
    y_pred = cancer_pipeline.predict(X_test)
    y_proba = cancer_pipeline.predict_proba(X_test)[:, 1]
    
    cancer_metrics = {
        'auc': float(roc_auc_score(y_test, y_proba)),
        'f1': float(f1_score(y_test, y_pred)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'n_features': len(ESSENTIAL_FEATURES)
    }
    
    logger.info("Training survival prediction model...")
    
    # Train survival model
    survival_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
            n_estimators=120, max_depth=8, min_samples_split=10,
            min_samples_leaf=5, random_state=42
        ))
    ])
    
    X_train_surv, X_test_surv, y_train_surv, y_test_surv = train_test_split(
        X, survival, test_size=0.2, random_state=42
    )
    
    survival_pipeline.fit(X_train_surv, y_train_surv)
    
    # Evaluate survival model
    y_pred_surv = survival_pipeline.predict(X_test_surv)
    y_test_years = np.array(y_test_surv) / 12
    y_pred_years = y_pred_surv / 12
    
    survival_metrics = {
        'mae_months': float(mean_absolute_error(y_test_surv, y_pred_surv)),
        'mae_years': float(mean_absolute_error(y_test_years, y_pred_years)),
        'r2_score': float(r2_score(y_test_surv, y_pred_surv)),
        'mean_predicted_years': float(np.mean(y_pred_years)),
        'mean_actual_years': float(np.mean(y_test_years))
    }
    
    logger.info(f"Cancer model - AUC: {cancer_metrics['auc']:.3f}, F1: {cancer_metrics['f1']:.3f}")
    logger.info(f"Survival model - MAE: {survival_metrics['mae_years']:.2f} years, R²: {survival_metrics['r2_score']:.3f}")
    
    return cancer_pipeline, survival_pipeline, cancer_metrics, survival_metrics

# --------- FASTAPI APP ---------
app = FastAPI(
    title="AEGIS Cancer Risk Prediction API",
    description="AI-powered cancer risk assessment and survival prediction using biomarkers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
cancer_model = None
survival_model = None
model_metrics = None
survival_metrics = None
model_loaded = False
last_trained = None

# --------- MODEL LOADING ---------
def load_or_train_models():
    """Load existing models or train new ones"""
    global cancer_model, survival_model, model_metrics, survival_metrics, model_loaded, last_trained
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    cancer_model_path = model_dir / "cancer_model.pkl"
    survival_model_path = model_dir / "survival_model.pkl"
    metrics_path = model_dir / "model_metrics.pkl"
    
    try:
        if (cancer_model_path.exists() and 
            survival_model_path.exists() and 
            metrics_path.exists()):
            
            logger.info("Loading existing models...")
            
            with open(cancer_model_path, 'rb') as f:
                cancer_model = pickle.load(f)
            
            with open(survival_model_path, 'rb') as f:
                survival_model = pickle.load(f)
                
            with open(metrics_path, 'rb') as f:
                metrics_data = pickle.load(f)
                model_metrics = metrics_data['cancer_metrics']
                survival_metrics = metrics_data['survival_metrics']
                last_trained = metrics_data.get('trained_at', 'Unknown')
            
            logger.info("Models loaded successfully")
            
        else:
            logger.info("Training new models...")
            train_and_save_models()
            
        model_loaded = True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.info("Training new models due to loading error...")
        train_and_save_models()

def train_and_save_models():
    """Train new models and save them"""
    global cancer_model, survival_model, model_metrics, survival_metrics, last_trained
    
    cancer_model, survival_model, model_metrics, survival_metrics = train_models()
    
    # Save models
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    with open(model_dir / "cancer_model.pkl", 'wb') as f:
        pickle.dump(cancer_model, f)
        
    with open(model_dir / "survival_model.pkl", 'wb') as f:
        pickle.dump(survival_model, f)
        
    # Save metrics
    metrics_data = {
        'cancer_metrics': model_metrics,
        'survival_metrics': survival_metrics,
        'trained_at': datetime.now().isoformat()
    }
    
    with open(model_dir / "model_metrics.pkl", 'wb') as f:
        pickle.dump(metrics_data, f)
    
    last_trained = datetime.now().isoformat()
    logger.info("Models trained and saved successfully")

# --------- PREDICTION FUNCTIONS ---------
def predict_single_patient(patient_data: PatientInput) -> RiskPrediction:
    """Make prediction for a single patient"""
    try:
        # Convert to DataFrame
        data_dict = patient_data.dict()
        df = pd.DataFrame([data_dict])
        
        # Preprocess gene expression (log2 transform)
        gene_cols = ['TP53', 'BRCA1', 'EGFR', 'MYC']
        for col in gene_cols:
            df[col] = np.log2(df[col].clip(lower=0.1) + 1)
        
        # Make predictions
        cancer_proba = cancer_model.predict_proba(df)[0, 1]
        survival_months = survival_model.predict(df)[0]
        
        # Calculate risk category
        if cancer_proba < 0.25:
            risk_category = "Low Risk"
        elif cancer_proba < 0.50:
            risk_category = "Moderate Risk"
        elif cancer_proba < 0.75:
            risk_category = "High Risk"
        else:
            risk_category = "Very High Risk"
        
        # Calculate confidence
        confidence = float(max(cancer_proba, 1 - cancer_proba))
        
        return RiskPrediction(
            cancer_risk_percentage=round(float(cancer_proba * 100), 2),
            risk_category=risk_category,
            survival_months=round(float(survival_months), 1),
            survival_years=round(float(survival_months / 12), 2),
            confidence_score=round(confidence, 3)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# --------- API ENDPOINTS ---------
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting AEGIS API...")
    load_or_train_models()
    logger.info("AEGIS API ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "api_version": "1.0.0",
        "message": "AEGIS Cancer Risk Prediction API"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model_loaded else "models_not_loaded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "api_version": "1.0.0"
    }

@app.get("/model-info")
async def get_model_info():
    """Get model information"""
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )
    
    return {
        "model_version": "1.0.0",
        "features_used": ESSENTIAL_FEATURES,
        "last_trained": last_trained or "Unknown",
        "performance_metrics": {
            "cancer_model": model_metrics,
            "survival_model": survival_metrics
        }
    }

@app.post("/predict", response_model=RiskPrediction)
async def predict_risk(patient: PatientInput):
    """Predict cancer risk and survival for a single patient"""
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please try again later."
        )
    
    try:
        prediction = predict_single_patient(patient)
        logger.info(f"Prediction: {prediction.risk_category}, {prediction.cancer_risk_percentage}%")
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-batch", response_model=BatchRiskPrediction)
async def predict_risk_batch(batch: BatchPatientInput):
    """Predict cancer risk and survival for multiple patients"""
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please try again later."
        )
    
    if len(batch.patients) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No patients provided"
        )
    
    try:
        predictions = []
        
        for i, patient in enumerate(batch.patients):
            prediction = predict_single_patient(patient)
            prediction.patient_id = f"patient_{i+1}"
            predictions.append(prediction)
        
        # Calculate batch statistics
        risk_percentages = [p.cancer_risk_percentage for p in predictions]
        survival_years = [p.survival_years for p in predictions]
        
        batch_stats = {
            "total_patients": len(predictions),
            "average_risk": round(np.mean(risk_percentages), 2),
            "average_survival_years": round(np.mean(survival_years), 2),
            "high_risk_count": len([p for p in predictions if p.cancer_risk_percentage >= 75]),
            "low_risk_count": len([p for p in predictions if p.cancer_risk_percentage < 25])
        }
        
        logger.info(f"Batch prediction completed for {len(predictions)} patients")
        
        return BatchRiskPrediction(
            predictions=predictions,
            batch_stats=batch_stats
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.post("/retrain")
async def retrain_models():
    """Retrain models with new data"""
    try:
        logger.info("Retraining models...")
        train_and_save_models()
        logger.info("Models retrained successfully")
        
        return {
            "status": "success",
            "message": "Models retrained successfully",
            "timestamp": datetime.now().isoformat(),
            "performance": {
                "cancer_model": model_metrics,
                "survival_model": survival_metrics
            }
        }
        
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining failed: {str(e)}"
        )

@app.get("/example-input")
async def get_example_input():
    """Get example input for testing"""
    return {
        "high_risk_example": {
            "TP53": 2.5, "BRCA1": 3.0, "EGFR": 8.5, "MYC": 9.0,
            "age": 65, "bmi": 32.0, "smoking_history": 1,
            "family_history": 1, "previous_cancer_history": 1,
            "inflammatory_markers": 8.5
        },
        "low_risk_example": {
            "TP53": 6.0, "BRCA1": 5.5, "EGFR": 4.0, "MYC": 4.5,
            "age": 35, "bmi": 22.0, "smoking_history": 0,
            "family_history": 0, "previous_cancer_history": 0,
            "inflammatory_markers": 1.0
        },
        "field_descriptions": {
            "TP53": "Tumor suppressor gene expression (0-15)",
            "BRCA1": "DNA repair gene expression (0-15)", 
            "EGFR": "Growth factor receptor expression (0-15)",
            "MYC": "Oncogene expression (0-15)",
            "age": "Patient age in years (18-120)",
            "bmi": "Body Mass Index (12-50)",
            "smoking_history": "0=No smoking history, 1=Has smoking history",
            "family_history": "0=No family cancer history, 1=Has family history",
            "previous_cancer_history": "0=No previous cancer, 1=Previous cancer",
            "inflammatory_markers": "Inflammatory markers level (0-20)"
        }
    }

# --------- RUN SERVER ---------
if __name__ == "__main__":
    uvicorn.run(
        "aegis_complete_api:app",
        host="0.0.0.0",
        port=10000,
        reload=False,
        log_level="info"
    )