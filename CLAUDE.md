# ai-detection-service — CLAUDE.md

> **STATUT : en cours d'implémentation (Sprint 2, v1).**

Microservice IA de la plateforme (voir `../CLAUDE.md` pour la vue
d'ensemble). Python, **FastAPI**.

## Rôle

- **Stateless** : ne conserve aucun état entre deux appels (le cache des
  modèles chargés en mémoire n'est pas un état métier).
- Reçoit une image, exécute un modèle **pré-entraîné** (PyTorch/MONAI),
  renvoie des résultats structurés.
- **Aucun entraînement** de modèle dans ce service — un modèle pré-entraîné
  est intégré tel quel. Ce n'est pas un service de MLOps/training.
- Supporte **deux types** de résultats, discriminés par le registre de
  modèles : **détection** (bounding boxes, CT/THORAX) et **segmentation**
  (masques binaires par coupe, CT/ABDOMEN). Chaque combinaison
  `(modalite, zone)` route vers un seul type — jamais de mélange dans une
  même réponse.

## Entrée

- Le backend envoie les fichiers **DICOM bruts**, pas les PNG d'aperçu.
  Raison : les modèles CT ont besoin des vraies valeurs Hounsfield (HU) ;
  les PNG 8-bit ont perdu cette information.
- Le backend transmet aussi `modalite` et `zone` en **paramètres de
  requête** : ils servent de clé de routage vers le bon modèle.

## Routage des modèles

- Registre `MODEL_SPECS` : dict `(modalite, zone)` → spécification du modèle.
- Chargement **paresseux** (lazy) au premier appel + **cache mémoire**.
- Chaque entrée porte son `type` (`detection` / `segmentation`).
- Combinaison `(modalite, zone)` inconnue → erreur explicite (pas de
  modèle par défaut).

## Préprocessing

On **réutilise les transforms fournis par le bundle MONAI** (fenêtrage HU,
resampling, normalisation) — on ne les réécrit pas.

## Modèle v1

`lung_nodule_ct_detection` (CT · thorax · nodules) — seul modèle de
détection du MONAI Model Zoo.

**Limite connue** : le modèle sort des boîtes **3D** (un nodule peut
s'étaler sur plusieurs coupes). En v1, chaque boîte est projetée sur la
coupe où le nodule est le plus visible : **une détection = une coupe**.

## Modèle v2 — segmentation

`spleen_ct_segmentation` (CT · abdomen · rate) — modèle de segmentation du
MONAI Model Zoo, routé sur `(CT, ABDOMEN)`.

- Classe unique produite : `rate`. Pas de multi-organes en v2.
- Le volume de sortie du réseau est réaligné sur la géométrie DICOM
  d'origine via les transforms d'inversion du bundle (`preprocessing.inverse`),
  puis découpé **par coupe** : une entrée par coupe contenant au moins un
  voxel classé rate (contrairement à la détection, au plus une entrée par
  coupe ici, puisqu'il n'y a qu'une seule classe cible).
- Chaque masque de coupe est encodé en **PNG niveaux de gris (0/255)**, puis
  en base64 (`masque_base64`).
- `confiance` : moyenne de la probabilité softmax de la classe rate sur les
  voxels classés rate de la coupe (pas un seuil de filtrage — une coupe est
  incluse dès qu'elle contient un voxel rate après `argmax`, quelle que soit
  sa confiance).

## Contrat de réponse (indicatif)

La réponse porte un champ discriminant `type` ∈ {`BOX`, `MASQUE`} ; chaque
détection a `label`, `confiance` et `coupe` (index 0-based dans la série
triée par position z) :

```json
{
  "type": "BOX",
  "detections": [
    {
      "label": "string",
      "bbox": { "x": 0, "y": 0, "w": 0, "h": 0 },
      "confiance": 0.0,
      "coupe": 0
    }
  ]
}
```

- `type=BOX` → objet `bbox {x,y,w,h}` en pixels de la coupe. Produit par les
  modèles de type `detection` (v1, `lung_nodule_ct_detection`).
- `type=MASQUE` → le service renvoie le masque **encodé en base64**
  (`masque_base64`) dans la réponse ; c'est le **backend** qui le persiste
  dans MinIO (le service n'accède jamais au stockage). Produit par les
  modèles de type `segmentation` (v2, `spleen_ct_segmentation`). Pas de
  champ `bbox` dans ce cas.
- `confiance` : score entre 0 et 1.
- Seuil de confiance en dessous duquel une détection est ignorée :
  **configurable** (variable d'environnement ou paramètre de requête), pas
  codé en dur.

Endpoints et format exact de requête : à fixer en coordination avec le
`DetectionClient` du backend (voir `../medical-imaging-backend/CLAUDE.md`).

## Mode d'appel

**Synchrone** en v1 : le backend attend la réponse. L'asynchrone (jobs +
polling) est une évolution future, à décider après mesure du temps
d'inférence réel.

## Frontières strictes

- Appelé **uniquement** par `medical-imaging-backend` (Spring Boot). Jamais
  appelé directement par le frontend Angular.
- N'accède **jamais** à la base de données de la plateforme.
- N'accède **jamais** au stockage de fichiers (dossier local/MinIO) — c'est
  le backend qui lui transmet les fichiers et qui persiste tout résultat
  (y compris les masques).
- Ne prend aucune décision clinique définitive : il produit des propositions
  que le backend expose au clinicien pour validation/correction — cohérent
  avec le principe directeur "l'IA propose, le clinicien valide" (voir
  `../CLAUDE.md`).
