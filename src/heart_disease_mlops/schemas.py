"""Validated request and response contracts for model serving."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """The 13 UCI Cleveland predictors for one patient encounter."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "age": 63,
                    "sex": 1,
                    "cp": 1,
                    "trestbps": 145,
                    "chol": 233,
                    "fbs": 1,
                    "restecg": 2,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 2.3,
                    "slope": 3,
                    "ca": 0,
                    "thal": 6,
                }
            ]
        },
    )

    age: int = Field(ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(description="0=female, 1=male")
    cp: Literal[1, 2, 3, 4] = Field(description="Chest pain category")
    trestbps: float = Field(ge=50, le=300, description="Resting blood pressure, mm Hg")
    chol: float = Field(ge=50, le=700, description="Serum cholesterol, mg/dl")
    fbs: Literal[0, 1] = Field(description="Fasting blood sugar above 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(description="Resting ECG category")
    thalach: float = Field(ge=40, le=250, description="Maximum heart rate")
    exang: Literal[0, 1] = Field(description="Exercise-induced angina")
    oldpeak: float = Field(ge=0, le=10, description="Exercise ST depression")
    slope: Literal[1, 2, 3] = Field(description="Peak exercise ST slope")
    ca: Literal[0, 1, 2, 3] = Field(description="Major vessels seen by fluoroscopy")
    thal: Literal[3, 6, 7] = Field(description="Thalassemia category")


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    risk_label: str
    confidence: float = Field(ge=0, le=1)
    disease_probability: float = Field(ge=0, le=1)
    decision_threshold: float = Field(gt=0, lt=1)
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
