"""Owns loading the model and tokenizer. Kept separate from inference.py so 'how the
model gets loaded' doesn't mix with 'how a prediction is computed'."""

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import MODEL_PATH, settings


class ModelBundle:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device


def _model_source() -> str:
    """A local folder if MODEL_DIR resolves to one on disk; otherwise MODEL_DIR is
    treated as a HuggingFace Hub repo id (e.g. 'Ahmed-Tarek0/emotion-roberta-large')
    and `transformers` downloads + caches it automatically."""
    return str(MODEL_PATH) if MODEL_PATH.exists() else settings.model_dir


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    source = _model_source()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source).to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the model from '{source}'. Set MODEL_DIR in backend/.env to "
            "either a local folder path or a HuggingFace Hub repo id "
            "(e.g. 'Ahmed-Tarek0/emotion-roberta-large')."
        ) from exc
    return ModelBundle(tokenizer, model, device)
