import torch

from app.config import EMOTION_COLORS, EMOTION_EMOJI, settings
from app.model_loader import get_model_bundle


@torch.no_grad()
def classify_text(text: str) -> dict:
    bundle = get_model_bundle()
    inputs = bundle.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=settings.max_len
    ).to(bundle.device)
    logits = bundle.model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    id2label = bundle.model.config.id2label
    probabilities = {id2label[i]: round(p, 4) for i, p in enumerate(probs)}
    top_idx = max(range(len(probs)), key=lambda i: probs[i])
    top_label = id2label[top_idx]

    return {
        "label": top_label,
        "confidence": round(probs[top_idx], 4),
        "color": EMOTION_COLORS.get(top_label, "#999999"),
        "emoji": EMOTION_EMOJI.get(top_label, ""),
        "probabilities": probabilities,
    }
