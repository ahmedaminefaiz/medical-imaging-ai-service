import threading
from pathlib import Path
from typing import Any

from app.config import get_settings


class UnknownModelError(LookupError):
    pass


class ModelLoadError(RuntimeError):
    pass


def _build_specs() -> dict[tuple[str, str], dict[str, Any]]:
    return {
        ("CT", "THORAX"): {
            "name": "lung_nodule",
            "path": str(Path(get_settings().models_dir) / "lung_nodule_ct_detection"),
            "type": "detection",
        },
        ("CT", "ABDOMEN"): {
            "name": "spleen",
            "path": str(Path(get_settings().models_dir) / "spleen_ct_segmentation"),
            "type": "segmentation",
        },
    }


MODEL_SPECS: dict[tuple[str, str], dict[str, Any]] = _build_specs()

_cache: dict[tuple[str, str], Any] = {}
_lock = threading.Lock()


def _parser_for_bundle(spec: dict[str, Any]) -> tuple[Any, Path]:
    from monai.bundle import ConfigParser

    bundle = Path(spec["path"])
    config_file = bundle / "configs" / "inference.json"
    weights_file = bundle / "models" / "model.pt"
    for f in (config_file, weights_file):
        if not f.is_file():
            raise ModelLoadError(
                f"Bundle MONAI incomplet : {f} introuvable (voir README, téléchargement du bundle)."
            )

    parser = ConfigParser()
    parser.read_config(str(config_file))
    meta_file = bundle / "configs" / "metadata.json"
    if meta_file.is_file():
        parser.read_meta(str(meta_file))
    return parser, weights_file


def _load_detection_bundle(spec: dict[str, Any]) -> dict[str, Any]:
    import torch

    parser, weights_file = _parser_for_bundle(spec)

    device = parser.get_parsed_content("device")
    network = parser.get_parsed_content("network")
    network.load_state_dict(torch.load(weights_file, map_location=device))
    detector = parser.get_parsed_content("detector")
    parser.get_parsed_content("detector_ops")
    patch = get_settings().infer_patch_size
    if patch:
        detector.set_sliding_window_inferer(
            roi_size=patch, overlap=0.25, sw_batch_size=1, mode="constant", device="cpu"
        )
    detector.eval()
    preprocessing = parser.get_parsed_content("preprocessing")

    return {"detector": detector, "preprocessing": preprocessing, "device": device}


def _load_segmentation_bundle(spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    from monai.transforms import Compose

    parser, weights_file = _parser_for_bundle(spec)

    device = parser.get_parsed_content("device")
    network = parser.get_parsed_content("network")
    ckpt = torch.load(weights_file, map_location=device, weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    network.load_state_dict(state)
    network.eval()

    inferer = parser.get_parsed_content("inferer")
    roi = get_settings().seg_roi_size
    if roi and hasattr(inferer, "roi_size"):
        inferer.roi_size = roi

    preprocessing = parser.get_parsed_content("preprocessing")

    # Réutilise les transforms "pred" du bundle (Activationsd=softmax, Invertd
    # =réalignement sur la géométrie DICOM d'origine via l'historique de
    # "preprocessing") plutôt que de réinventer l'inversion à la main — voir
    # postprocessing du bundle. On exclut AsDiscreted (l'argmax se fait après,
    # côté infer_segmentation, pour garder les probas et calculer la
    # confiance) et SaveImaged (ce service est stateless, n'écrit jamais sur
    # disque).
    postprocessing_bundle = parser.get_parsed_content("postprocessing")
    retenues = [t for t in postprocessing_bundle.transforms if type(t).__name__ in ("Activationsd", "Invertd")]
    if not retenues:
        raise ModelLoadError(
            "Bundle de segmentation inattendu : aucune transform Activationsd/Invertd "
            "trouvée dans postprocessing (structure du bundle différente de celle prévue)."
        )
    postprocessing = Compose(retenues)

    return {
        "network": network,
        "preprocessing": preprocessing,
        "postprocessing": postprocessing,
        "inferer": inferer,
        "device": device,
    }


def _load_bundle(spec: dict[str, Any]) -> Any:
    if spec["type"] == "detection":
        return _load_detection_bundle(spec)
    if spec["type"] == "segmentation":
        return _load_segmentation_bundle(spec)
    raise NotImplementedError(f"Type de modèle non supporté : {spec['type']}")


def get_model(modalite: str, zone: str) -> tuple[Any, str]:
    key = (modalite.strip().upper(), zone.strip().upper())
    spec = MODEL_SPECS.get(key)
    if spec is None:
        supported = ", ".join(f"{m}/{z}" for m, z in MODEL_SPECS)
        raise UnknownModelError(
            f"Aucun modèle pour (modalite={key[0]}, zone={key[1]}). Combinaisons supportées : {supported}."
        )
    with _lock:
        if key not in _cache:
            _cache[key] = _load_bundle(spec)
        return _cache[key], spec["type"]
