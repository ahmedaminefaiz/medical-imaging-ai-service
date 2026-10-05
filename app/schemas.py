from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class DetectionBox(BaseModel):
    label: str
    bbox: BBox
    confiance: float = Field(ge=0.0, le=1.0)
    coupe: int = Field(ge=0, description="Index 0-based de la coupe dans la série triée")


class DetectionMasque(BaseModel):
    """Masque de segmentation 2D par coupe, encodé en PNG base64. Produit
    par les modèles de type "segmentation" du registre (ex. spleen_ct_segmentation).
    Le backend est responsable de le persister dans MinIO."""

    label: str
    masque_base64: str
    confiance: float = Field(ge=0.0, le=1.0)
    coupe: int = Field(ge=0)


class PredictResponse(BaseModel):
    type: Literal["BOX", "MASQUE"]
    detections: list[DetectionBox] | list[DetectionMasque]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
