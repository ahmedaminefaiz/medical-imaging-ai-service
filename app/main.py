import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.registry import MODEL_SPECS

logger = logging.getLogger("ai-detection-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "Démarrage : seuil=%s, modèles déclarés=%s (chargement paresseux)",
        settings.confidence_threshold,
        [f"{m}/{z}" for m, z in MODEL_SPECS],
    )
    yield


app = FastAPI(title="ai-detection-service", version="1.0.0", lifespan=lifespan)
app.include_router(router)
