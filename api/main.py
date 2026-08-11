"""FastAPI backend for EGFR pIC50 prediction.

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import get_prediction_pipeline
from api.routers import predict_router
from api.schemas import AVAILABLE_MODELS, HealthResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup rather than on the first request, so the first
    # user isn't the one who pays for the Random Forest / fingerprint load.
    logger.info("Loading prediction pipeline...")
    get_prediction_pipeline()
    logger.info(
        "Prediction pipeline ready: %s", get_prediction_pipeline().available_models()
    )
    yield


app = FastAPI(
    title="EGFR pIC50 Prediction API",
    description=(
        "Predicts EGFR kinase pIC50 from a SMILES string using Random Forest, "
        "a CheMeleon fine-tune, and a Chemprop D-MPNN, with an uncertainty "
        "interval and an applicability domain flag on every prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/", tags=["Root"])
def root():
    return {
        "name": "EGFR pIC50 Prediction API",
        "version": "1.0.0",
        "available_models": AVAILABLE_MODELS,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    pipeline = get_prediction_pipeline()
    return HealthResponse(
        status="healthy", available_models=pipeline.available_models()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
