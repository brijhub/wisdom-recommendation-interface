"""FastAPI backend. Run with: python -m uvicorn backend:app --app-dir src"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rag import recommend_wisdom


logger = logging.getLogger(__name__)
app = FastAPI(title="Wisdom Recommendation API", version="1.0.0")


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    transcript: str = Field(min_length=1, max_length=50_000)
    person: str = Field(min_length=1, max_length=80)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend")
def recommend(request: RecommendationRequest) -> dict:
    try:
        return recommend_wisdom(request.transcript, request.person)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Recommendation failed")
        raise HTTPException(
            status_code=500,
            detail="Could not create a recommendation. Check OpenAI settings and the local ChromaDB index.",
        ) from error

