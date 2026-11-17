"""Local, synthetic-only inference. No training or tracking server at startup."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pharmasignal.data.schema import PREDICTORS
from pharmasignal.models.artifact import load_artifact, predict_records
from pharmasignal.serving.requests import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


def create_app(model_dir: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.model, app.state.metadata = load_artifact(model_dir)
        except Exception:
            raise RuntimeError("Trusted model artifact unavailable or incompatible") from None
        try:
            yield
        finally:
            app.state.model = None
            app.state.metadata = None

    app = FastAPI(
        title="PharmaSignal local demonstration",
        description=(
            "Synthetic report triage only. predicted_signal is a review-priority label, "
            "not a population safety signal. Scores are not calibrated clinical risk."
        ),
        lifespan=lifespan,
    )
    app.state.model = None
    app.state.metadata = None

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Pydantic errors otherwise echo input, including raw narratives and
        # user-controlled extra keys. Expose only known field locations/types.
        details = []
        for error in exc.errors():
            location = [
                part if part in ("body", *PREDICTORS) else "unknown" for part in error["loc"]
            ]
            details.append({"loc": location, "type": error["type"], "msg": "Invalid request"})
        return JSONResponse(status_code=422, content={"detail": details})

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        if app.state.model is None or app.state.metadata is None:
            raise HTTPException(status_code=503, detail="Model unavailable")
        return {
            "status": "ready",
            "model_version": app.state.metadata["model_version"],
            "schema_version": app.state.metadata["schema_version"],
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(record: PredictionRequest):
        ready()
        try:
            result = predict_records(app.state.model, app.state.metadata, [record.model_dump()])[0]
            return PredictionResponse(**result)
        except Exception as exc:
            # No payload, exception message or traceback in service logs.
            logger.error("Inference failed (%s)", type(exc).__name__)
            return JSONResponse(status_code=500, content={"detail": "Prediction failed"})

    return app
