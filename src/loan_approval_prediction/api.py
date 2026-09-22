"""FastAPI service for loan approval predictions."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from loan_approval_prediction.inference import load_bundle, predict_one
from loan_approval_prediction.training import DEFAULT_MODEL_PATH

LOGGER = logging.getLogger(__name__)


class LoanFeatures(BaseModel):
    ApplicantIncome: float = Field(ge=0)
    CoapplicantIncome: float = Field(ge=0)
    LoanAmount: float | None = Field(default=None, gt=0)
    Loan_Amount_Term: float | None = Field(default=None, gt=0)
    Dependents: Literal["0", "1", "2", "3+"] | None = None
    Education: Literal["Graduate", "Not Graduate"] | None = None
    Self_Employed: Literal["Yes", "No"] | None = None
    Credit_History: Literal[0, 1] | None = None
    Property_Area: Literal["Urban", "Semiurban", "Rural"] | None = None

    model_config = {"extra": "forbid"}


class Prediction(BaseModel):
    prediction: Literal["Y", "N"]
    approval_probability: float
    decision_threshold: float
    model: str


def create_app(model_path: str | Path = DEFAULT_MODEL_PATH) -> FastAPI:
    """Build an app that loads the bundle once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = load_bundle(model_path)
        LOGGER.info("Loaded model from %s", model_path)
        yield

    app = FastAPI(title="Loan Approval API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=Prediction)
    def predict(features: LoanFeatures, request: Request) -> dict:
        try:
            return predict_one(request.app.state.bundle, features.model_dump())
        except Exception:
            LOGGER.exception("Prediction failed")
            raise

    return app


app = create_app()
