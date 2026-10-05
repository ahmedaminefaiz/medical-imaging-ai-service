import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from app import inference, registry
from app.inference import RawBox
from app.main import app

client = TestClient(app)


def make_dicom(z: float) -> bytes:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.ImagePositionPatient = [0.0, 0.0, z]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [0.5, 0.5]
    ds.Rows = ds.Columns = 4
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.zeros((4, 4), dtype=np.int16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


@pytest.fixture
def mocked(monkeypatch):
    monkeypatch.setattr(registry, "get_model", lambda m, z: (object(), "detection"))
    raw = [
        RawBox((10.0, 20.0, 1.9), (20.0, 30.0, 2.1), 0.9, 0),  # centre z=2.0 -> coupe 2
        RawBox((0.0, 0.0, 0.9), (5.0, 5.0, 1.1), 0.3, 0),  # sous le seuil
    ]
    monkeypatch.setattr(inference, "infer_volume", lambda model, path: raw)


def files():
    return [("files", (f"{i}.dcm", make_dicom(float(i)), "application/dicom")) for i in range(4)]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_format_and_threshold(mocked):
    r = client.post("/predict?modalite=CT&zone=THORAX", files=files())
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "BOX"
    assert len(body["detections"]) == 1  # 0.3 < seuil 0.5
    d = body["detections"][0]
    assert d["label"] == "nodule"
    assert d["coupe"] == 2
    assert d["confiance"] == pytest.approx(0.9)
    assert d["bbox"] == {"x": 20.0, "y": 40.0, "w": 20.0, "h": 20.0}


def test_predict_seuil_override(mocked):
    r = client.post("/predict?modalite=CT&zone=THORAX&seuil=0.1", files=files())
    assert len(r.json()["detections"]) == 2


def test_predict_unknown_combination():
    r = client.post("/predict?modalite=IRM&zone=CERVEAU", files=files())
    assert r.status_code == 422
    assert "CT/THORAX" in r.json()["detail"]


def test_predict_invalid_dicom(mocked):
    r = client.post(
        "/predict?modalite=CT&zone=THORAX",
        files=[("files", ("x.dcm", b"pas du dicom", "application/dicom"))],
    )
    assert r.status_code == 422


@pytest.fixture
def mocked_segmentation(monkeypatch):
    monkeypatch.setattr(registry, "get_model", lambda m, z: (object(), "segmentation"))
    monkeypatch.setattr(
        inference,
        "infer_segmentation",
        lambda model, series_dir: [
            {"coupe": 2, "label": "rate", "confiance": 0.9, "masque_base64": "iVBORw0KG..."}
        ],
    )


def test_predict_segmentation_format(mocked_segmentation):
    r = client.post("/predict?modalite=CT&zone=ABDOMEN", files=files())
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "MASQUE"
    d = body["detections"][0]
    assert d["label"] == "rate"
    assert d["coupe"] == 2
    assert d["confiance"] == pytest.approx(0.9)
    assert "masque_base64" in d
    assert "bbox" not in d


def test_predict_segmentation_nest_plus_un_501(mocked_segmentation):
    r = client.post("/predict?modalite=CT&zone=ABDOMEN", files=files())
    assert r.status_code != 501
