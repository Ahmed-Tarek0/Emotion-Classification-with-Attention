from fastapi import APIRouter, HTTPException

from app.inference import classify_text
from app.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["emotion"])


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    result = classify_text(text)
    return PredictResponse(**result)
