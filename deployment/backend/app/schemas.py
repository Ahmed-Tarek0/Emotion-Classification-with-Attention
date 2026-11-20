from typing import Dict

from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    color: str
    emoji: str
    probabilities: Dict[str, float]
