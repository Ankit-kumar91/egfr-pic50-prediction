"""Process-wide singletons shared across API requests."""

from functools import lru_cache

from src.pipeline.predict_pipeline import PredictionPipeline


@lru_cache(maxsize=1)
def get_prediction_pipeline() -> PredictionPipeline:
    """One PredictionPipeline per process, so the Random Forest and training
    fingerprints load once instead of on every request."""
    return PredictionPipeline()
