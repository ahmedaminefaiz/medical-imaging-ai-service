from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    confidence_threshold: float = 0.5
    models_dir: str = "./models"
    infer_patch_size: list[int] | None = None  # None = valeur du bundle ([192,192,80])
    seg_roi_size: list[int] | None = None  # None = valeur du bundle ; ex. [96,96,96] pour réduire sur GPU 2 Go


@lru_cache
def get_settings() -> Settings:
    return Settings()
