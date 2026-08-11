"""Pydantic request/response models for the prediction API."""

from pydantic import BaseModel, Field

from src.pipeline.predict_pipeline import MODEL_PATHS

AVAILABLE_MODELS = list(MODEL_PATHS)


class PredictRequest(BaseModel):
    smiles: str
    models: list[str] = Field(
        default_factory=lambda: AVAILABLE_MODELS,
        description=f"Subset of {AVAILABLE_MODELS} to run. Defaults to all three.",
    )


class BatchPredictRequest(BaseModel):
    smiles_list: list[str]
    models: list[str] = Field(default_factory=lambda: AVAILABLE_MODELS)


class ApplicabilityDomain(BaseModel):
    nearest_neighbor_similarity: float
    in_domain: bool
    threshold: float


class ModelPrediction(BaseModel):
    pic50: float
    lower: float
    upper: float


class PredictResponse(BaseModel):
    smiles: str
    applicability_domain: ApplicabilityDomain
    predictions: dict[str, ModelPrediction]


class BatchResultRow(BaseModel):
    smiles: str
    applicability_domain: ApplicabilityDomain | None = None
    predictions: dict[str, ModelPrediction] | None = None
    error: str | None = None


class BatchPredictResponse(BaseModel):
    results: list[BatchResultRow]


class HealthResponse(BaseModel):
    status: str
    available_models: list[str]
