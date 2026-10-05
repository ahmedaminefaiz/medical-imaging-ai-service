from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app import inference, registry
from app.config import get_settings
from app.schemas import HealthResponse, PredictResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post("/predict", response_model=PredictResponse)
def predict(
    files: list[UploadFile] = File(..., description="Coupes DICOM brutes de la série"),
    modalite: str = Query(...),
    zone: str = Query(...),
    seuil: float | None = Query(None, ge=0.0, le=1.0, description="Surcharge CONFIDENCE_THRESHOLD"),
) -> PredictResponse:
    try:
        model, model_type = registry.get_model(modalite, zone)
    except registry.UnknownModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except registry.ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    threshold = seuil if seuil is not None else get_settings().confidence_threshold
    contents = [f.file.read() for f in files]
    try:
        if model_type == "segmentation":
            detections = inference.run_inference(model, contents, threshold, task_type="segmentation")
            return PredictResponse(type="MASQUE", detections=detections)
        detections = inference.run_inference(model, contents, threshold, task_type="detection")
        return PredictResponse(type="BOX", detections=detections)
    except inference.InvalidSeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
