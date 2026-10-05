import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from app import inference


class FakePreprocessing:
    """Mock de la transform preprocessing du bundle : renvoie un volume factice
    de forme connue (le contenu n'importe pas, seule sa forme compte)."""

    def __call__(self, data):
        return {"image": torch.zeros(1, 4, 4, 4)}  # [1, X, Y, Z]


class FakePostprocessing:
    """Mock du Compose [Activationsd, Invertd] réellement chargé par
    registry._load_segmentation_bundle : applique le softmax (comme
    Activationsd) et, puisqu'il n'y a pas de vrai réalignement géométrique à
    tester ici (couvert par le test manuel contre le bundle réel), restitue
    "pred" à la même résolution que le volume factice (Invertd ferait de même
    pour un bundle réel appliqué à une image non rééchantillonnée)."""

    def __call__(self, data):
        data["pred"] = torch.softmax(data["pred"], dim=0)
        return data


class FakeInferer:
    def __call__(self, image, network):
        # [1, 2, X, Y, Z] : rate (classe 1) présente sur les coupes z=1 et z=3
        # d'un volume Z=4, absente ailleurs.
        output = torch.zeros(1, 2, 4, 4, 4)
        output[0, 0] = 10.0  # logit classe fond, dominant partout par défaut
        output[0, 1, :, :, 1] = 20.0  # coupe 1 : classe rate dominante partout
        output[0, 1, :, :, 3] = 20.0  # coupe 3 : classe rate dominante partout
        return output


@pytest.fixture
def fake_model():
    return {
        "preprocessing": FakePreprocessing(),
        "postprocessing": FakePostprocessing(),
        "inferer": FakeInferer(),
        "network": object(),
        "device": "cpu",
    }


def test_infer_segmentation_une_entree_par_coupe_avec_rate(fake_model, tmp_path):
    detections = inference.infer_segmentation(fake_model, tmp_path)

    coupes = sorted(d["coupe"] for d in detections)
    assert coupes == [1, 3]


def test_infer_segmentation_exclut_les_coupes_sans_rate(fake_model, tmp_path):
    detections = inference.infer_segmentation(fake_model, tmp_path)

    assert all(d["coupe"] in (1, 3) for d in detections)
    assert len(detections) == 2


def test_infer_segmentation_label_et_format(fake_model, tmp_path):
    detections = inference.infer_segmentation(fake_model, tmp_path)

    for d in detections:
        assert d["label"] == "rate"
        assert 0.0 <= d["confiance"] <= 1.0
        assert isinstance(d["masque_base64"], str) and d["masque_base64"]


def test_infer_segmentation_confiance_proche_de_1_quand_le_logit_est_tranche(fake_model, tmp_path):
    # Logits très séparés (10 vs 20) -> softmax quasi 1.0 sur la classe rate.
    detections = inference.infer_segmentation(fake_model, tmp_path)

    for d in detections:
        assert d["confiance"] == pytest.approx(1.0, abs=1e-3)


def test_infer_segmentation_masque_decode_correspond_aux_voxels_rate(fake_model, tmp_path):
    detections = inference.infer_segmentation(fake_model, tmp_path)
    detection_coupe_1 = next(d for d in detections if d["coupe"] == 1)

    png_bytes = base64.b64decode(detection_coupe_1["masque_base64"])
    image = Image.open(io.BytesIO(png_bytes))
    pixels = np.array(image)

    # Toute la coupe 1 est classée "rate" dans FakeInferer -> masque tout à 255.
    assert pixels.shape == (4, 4)
    assert (pixels == 255).all()


def test_encoder_masque_png_pixels_0_255():
    masque = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    png_b64 = inference.encoder_masque_png(masque)
    image = Image.open(io.BytesIO(base64.b64decode(png_b64)))
    pixels = np.array(image)

    assert pixels.tolist() == [[0, 255], [255, 0]]
