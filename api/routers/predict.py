"""Prediction endpoints: single molecule and batch."""

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_prediction_pipeline
from api.schemas import (
    AVAILABLE_MODELS,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from src.pipeline.predict_pipeline import PredictionPipeline

router = APIRouter(tags=["Prediction"])


def _validate_models(models: list[str]) -> None:
    unknown = set(models) - set(AVAILABLE_MODELS)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model name(s) {sorted(unknown)}. "
                f"Choose from {AVAILABLE_MODELS}."
            ),
        )


@router.post("/predict/single", response_model=PredictResponse)
def predict_single(
    request: PredictRequest,
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline),
):
    _validate_models(request.models)
    result = pipeline.predict(request.smiles, request.models)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    request: BatchPredictRequest,
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline),
):
    _validate_models(request.models)
    if not request.smiles_list:
        raise HTTPException(status_code=400, detail="smiles_list must not be empty")
    return {"results": pipeline.predict_batch(request.smiles_list, request.models)}
