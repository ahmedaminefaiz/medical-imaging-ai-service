# ai-detection-service

Microservice FastAPI de détection et segmentation (nodules pulmonaires sur
CT thorax ; rate sur CT abdomen). Stateless, appelé uniquement par
`medical-imaging-backend`. Voir `CLAUDE.md`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
```

## Récupérer les modèles (MONAI Model Zoo)

```bash
python -m monai.bundle download "lung_nodule_ct_detection" --bundle_dir ./models
python -m monai.bundle download "spleen_ct_segmentation" --bundle_dir ./models
```

Résultat attendu : `models/<bundle>/{configs,models}/`
(`configs/inference.json` et `models/model.pt` sont requis). Chaque bundle est
chargé au premier appel de `/predict` pour sa combinaison `(modalite, zone)`,
puis gardé en mémoire.

## Lancer

```bash
uvicorn app.main:app --reload --port 8000
```

- `GET /health`
- `POST /predict?modalite=CT&zone=THORAX[&seuil=0.5]` — multipart, champ `files`
  répété (une part par coupe DICOM brute). Détection (nodules pulmonaires).
- `POST /predict?modalite=CT&zone=ABDOMEN` — même forme de requête.
  Segmentation (rate).

```bash
curl -F "files=@slice1.dcm" -F "files=@slice2.dcm" \
  "http://localhost:8000/predict?modalite=CT&zone=THORAX"
```

Réponse (détection) : `{"type":"BOX","detections":[{"label","bbox":{x,y,w,h},"confiance","coupe"}]}`
(`coupe` = index 0-based dans la série triée par position z ; `bbox` en pixels de la coupe).

```bash
curl -F "files=@slice1.dcm" -F "files=@slice2.dcm" \
  "http://localhost:8000/predict?modalite=CT&zone=ABDOMEN"
```

Réponse (segmentation) :
`{"type":"MASQUE","detections":[{"coupe":42,"label":"rate","confiance":0.93,"masque_base64":"iVBORw0KG..."}]}`
(une entrée par coupe contenant au moins un voxel de la classe rate ; pas de
champ `bbox` ; `masque_base64` = PNG niveaux de gris 0/255 de la coupe,
encodé en base64).

## Configuration

| Variable | Défaut | Rôle |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.5` | Seuil sous lequel une détection est ignorée |
| `MODELS_DIR` | `./models` | Racine des bundles |
| `INFER_PATCH_SIZE` | bundle (`[192,192,80]`) | Patch de la fenêtre glissante (détection), ex. `[128,128,64]` |
| `SEG_ROI_SIZE` | bundle | Taille de ROI de la fenêtre glissante (segmentation), ex. `[96,96,96]` pour réduire sur GPU 2 Go |

## Ressources (mesuré sur CPU, série NLST de 150 coupes 512×512 — détection)

- Le patch par défaut du bundle demande ~3 Go d'allocation pour la 1re convolution :
  sur une machine avec ~2,6 Go de RAM libre, ça échoue (`not enough memory`).
  Avec `INFER_PATCH_SIZE=[128,128,64]` : ~10 min d'inférence, RAM OK.
- Prévoir un GPU ou un serveur avec plus de RAM en pratique ; à ~10 min par série, le mode
  synchrone v1 est **trop lent pour un usage réel** (voir décision async dans `CLAUDE.md`).
- Le chargement du bundle prend ~10 s (une fois, puis cache).
- Téléchargement du bundle : nécessite de l'espace disque (`model.pt` ≈ 84 Mo) et les paquets
  `requests`, `huggingface_hub`, `torchvision` (déjà dans `requirements.txt`).

### Segmentation (mesuré sur GPU, série CT abdomen de 33 coupes 512×512, ROI par défaut `[96,96,96]`)

- Chargement du bundle : ~3 s.
- Inférence : ~18 s pour 33 coupes.
- VRAM pic : ~770 Mo — largement sous la cible 2 Go avec le ROI par défaut
  du bundle ; `SEG_ROI_SIZE` reste disponible pour réduire davantage si une
  série beaucoup plus longue ou une carte plus contrainte l'exigent.
- Testé avec un volume du dataset Medical Segmentation Decathlon
  (Task09_Spleen, converti en série DICOM synthétique) ; le masque produit
  retombe sur des valeurs HU de tissu mou (pas de l'air) à l'endroit attendu
  de la rate.

## Tests

```bash
pytest
```

Les tests ne chargent pas le vrai modèle (mocks).

## Docker

```bash
docker build -t ai-detection-service .
docker run -p 8000:8000 -v "$PWD/models:/models" ai-detection-service
```
