from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_dir: str = "finetuned_transformer"
    max_len: int = 64
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"  # comma-separated, e.g. "http://127.0.0.1:5500,http://localhost:5500"


settings = Settings()

BACKEND_DIR = Path(__file__).resolve().parent.parent  # .../backend
MODEL_PATH = (BACKEND_DIR / settings.model_dir).resolve()

EMOTION_COLORS = {
    "joy": "#F4B400",
    "sadness": "#4A7FE8",
    "anger": "#E63946",
    "fear": "#8338EC",
    "surprise": "#FB8500",
    "disgust": "#52B788",
}

EMOTION_EMOJI = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
}
