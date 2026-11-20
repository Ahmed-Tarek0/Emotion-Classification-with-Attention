from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model_loader import get_model_bundle
from app.routers import emotion, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = get_model_bundle()  # load once at startup instead of on the first request
    print(f"Model loaded on {bundle.device}.")
    yield


app = FastAPI(title="Emotion Classifier API", lifespan=lifespan)

origins = ["*"] if settings.cors_origins == "*" else settings.cors_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(emotion.router)
