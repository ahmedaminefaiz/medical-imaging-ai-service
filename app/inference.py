"""DICOM (octets) -> volume -> transforms du bundle -> détecteur -> boxes par coupe.

Les imports torch/monai sont paresseux : ce module reste importable sans eux (tests).
"""

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

LABELS = {0: "nodule"}


class InvalidSeriesError(ValueError):
    pass


@dataclass
class Geometry:
    z_positions: np.ndarray  # mm, LPS, trié croissant : index = numéro de coupe
    origin_x: float  # ImagePositionPatient x de la 1re coupe (LPS)
    origin_y: float
    spacing_row: float  # PixelSpacing[0] (entre lignes, axe y)
    spacing_col: float  # PixelSpacing[1] (entre colonnes, axe x)


@dataclass
class RawBox:
    """Boîte 3D en coordonnées monde LPS (mm), non filtrée."""

    world_min: tuple[float, float, float]
    world_max: tuple[float, float, float]
    score: float
    label: int


def read_series(files: list[bytes]) -> tuple[Geometry, list[bytes]]:
    if not files:
        raise InvalidSeriesError("Aucun fichier DICOM fourni.")
    slices = []
    for raw in files:
        try:
            ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)
            pos = [float(v) for v in ds.ImagePositionPatient]
            iop = [float(v) for v in ds.ImageOrientationPatient]
            pixel_spacing = [float(v) for v in ds.PixelSpacing]
        except Exception as exc:
            raise InvalidSeriesError("Fichier DICOM illisible ou métadonnées de géométrie manquantes.") from exc
        if not np.allclose(np.abs(iop), [1, 0, 0, 0, 1, 0], atol=1e-3):
            raise InvalidSeriesError("Seules les coupes axiales sont supportées en v1.")
        slices.append((pos, pixel_spacing, raw))

    slices.sort(key=lambda s: s[0][2])
    first_pos, spacing, _ = slices[0]
    geometry = Geometry(
        z_positions=np.array([s[0][2] for s in slices]),
        origin_x=first_pos[0],
        origin_y=first_pos[1],
        spacing_row=spacing[0],
        spacing_col=spacing[1],
    )
    return geometry, [s[2] for s in slices]


def infer_volume(model: Any, series_dir: Path) -> list[RawBox]:
    """Preprocessing du bundle (HU, resampling, normalisation) + détecteur RetinaNet 3D."""
    import torch

    data = model["preprocessing"]({"image": str(series_dir)})
    image = data["image"]
    affine = np.asarray(image.affine, dtype=float)  # voxel -> monde RAS

    with torch.no_grad():
        outputs = model["detector"](input_images=[image.to(model["device"])], use_inferer=True)
    out = outputs[0]

    boxes = out["box"].cpu().numpy()  # xyzxyz en voxels de l'image rééchantillonnée
    scores = out["label_scores"].cpu().numpy()
    labels = out["label"].cpu().numpy()

    raw: list[RawBox] = []
    for box, score, label in zip(boxes, scores, labels):
        c1 = affine @ np.append(box[:3], 1.0)
        c2 = affine @ np.append(box[3:], 1.0)
        # RAS -> LPS : inversion de x et y
        p1 = np.array([-c1[0], -c1[1], c1[2]])
        p2 = np.array([-c2[0], -c2[1], c2[2]])
        raw.append(
            RawBox(
                world_min=tuple(float(v) for v in np.minimum(p1, p2)),
                world_max=tuple(float(v) for v in np.maximum(p1, p2)),
                score=float(score),
                label=int(label),
            )
        )
    return raw


def raw_to_detections(raw: list[RawBox], geometry: Geometry, threshold: float) -> list[dict]:
    """Filtre par seuil et projette chaque boîte 3D sur sa coupe centrale (1 détection = 1 coupe)."""
    detections = []
    for box in raw:
        if box.score < threshold:
            continue
        z_center = (box.world_min[2] + box.world_max[2]) / 2.0
        coupe = int(np.argmin(np.abs(geometry.z_positions - z_center)))
        x0 = (box.world_min[0] - geometry.origin_x) / geometry.spacing_col
        x1 = (box.world_max[0] - geometry.origin_x) / geometry.spacing_col
        y0 = (box.world_min[1] - geometry.origin_y) / geometry.spacing_row
        y1 = (box.world_max[1] - geometry.origin_y) / geometry.spacing_row
        detections.append(
            {
                "label": LABELS.get(box.label, "nodule"),
                "bbox": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
                "confiance": min(max(box.score, 0.0), 1.0),
                "coupe": coupe,
            }
        )
    detections.sort(key=lambda d: d["confiance"], reverse=True)
    return detections


def encoder_masque_png(masque_2d: np.ndarray) -> str:
    """Masque binaire (0/1) -> PNG niveaux de gris 8 bits (0/255) -> base64 ascii."""
    import base64

    from PIL import Image

    image = Image.fromarray((masque_2d.astype(np.uint8)) * 255, mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def infer_segmentation(model: Any, series_dir: Path) -> list[dict]:
    """Preprocessing du bundle + réseau de segmentation -> masque binaire par coupe.

    Le réalignement sur la géométrie DICOM d'origine réutilise les transforms
    "pred" du bundle (Activationsd=softmax, Invertd) chargées dans
    registry._load_segmentation_bundle — Compose.inverse(data) ne convient
    PAS ici : il inverse la clé "image" (celle que les transforms de
    preprocessing connaissent), jamais une clé "pred" ajoutée après coup, et
    laisserait silencieusement le masque dans l'espace rééchantillonné
    (confirmé par test manuel : des indices de coupe hors de la plage de la
    série d'origine).
    """
    import torch

    data = model["preprocessing"]({"image": str(series_dir)})
    image = data["image"].unsqueeze(0).to(model["device"])

    with torch.no_grad():
        output = model["inferer"](image, model["network"])  # [1, 2, X, Y, Z]

    data["pred"] = output[0]  # [2, X, Y, Z] ; Activationsd applique le softmax
    data = model["postprocessing"](data)  # Activationsd -> Invertd (résolution DICOM d'origine)
    proba_originale = data["pred"]  # [2, X0, Y0, Z0] à la résolution native

    masque = torch.argmax(proba_originale, dim=0).cpu().numpy()  # [X0, Y0, Z0], 0/1
    proba_rate = proba_originale[1].cpu().numpy()  # proba classe "rate", même forme

    detections: list[dict] = []
    for z in range(masque.shape[-1]):
        coupe_masque = masque[..., z]
        if not coupe_masque.any():
            continue
        confiance = float(proba_rate[..., z][coupe_masque == 1].mean())
        detections.append(
            {
                "coupe": z,
                "label": "rate",
                "confiance": min(max(confiance, 0.0), 1.0),
                "masque_base64": encoder_masque_png(coupe_masque),
            }
        )
    return detections


def run_inference(model: Any, files: list[bytes], threshold: float, task_type: str) -> list[dict]:
    geometry, ordered = read_series(files)
    # Répertoire temporaire éphémère uniquement pour le lecteur ITK de MONAI ; jamais persisté.
    with tempfile.TemporaryDirectory() as tmp:
        for i, raw in enumerate(ordered):
            (Path(tmp) / f"{i:05d}.dcm").write_bytes(raw)
        series_dir = Path(tmp)
        if task_type == "segmentation":
            return infer_segmentation(model, series_dir)
        raw_boxes = infer_volume(model, series_dir)
    return raw_to_detections(raw_boxes, geometry, threshold)
