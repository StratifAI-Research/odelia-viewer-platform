# ODELIA generalized model service (ODV-214)

A single service image that builds **any** model by `MODEL_NAME` through the
vendored MediSwarm `create_model` factory. It reuses the MST service structure
and keeps the `initialize_model` / `analyze_*` contract. `MST-classification/`
is left untouched.

## Layout

- `mediswarm/` — vendored MediSwarm `_shared/custom` (`models/` + `create_model`
  + `base_model`, plus `env_config.py`). Two documented patches carry the
  `ODV-214` marker:
  - `models/models_config.py` — CUDA guard (see below).
  - `env_config.py` — `ODELIA_Dataset3D` import made lazy so the serving image
    does not pull the training-side `data/` stack.
- `model_loader.py` — `build_model(model_name, num_classes)` via `create_model`.
- `model_service.py` — `ModelService`; retrieval → NIfTI → forward → response.
- `config.py`, `app_refactored.py`, `manifest.json`, `Dockerfile`.
- `smoke_test.py` — build a built-in + forward a dummy volume.

## Configuration (env)

| Var                | Default   | Meaning                                            |
|--------------------|-----------|----------------------------------------------------|
| `MODEL_NAME`       | `MST`     | Model to build (`ResNet50`, `MST`, …).             |
| `NUM_CLASSES`      | `3`       | Classification head width.                         |
| `MODEL_DEVICE`     | **required** | `cpu` or `cuda` — must be set (see below).       |
| `PORT`             | `5556`    | HTTP port (ODV-218 allocates per model).           |

### Device selection (`MODEL_DEVICE`)

MediSwarm's `create_model` hard-raised without a GPU. Here a device is never
chosen implicitly: `MODEL_DEVICE` **must** be set explicitly to `cpu` or `cuda`.
`cuda` requires an available GPU (fails loudly otherwise); `cpu` runs on CPU with
or without a GPU. If `MODEL_DEVICE` is unset or invalid, the service fails loudly
at startup rather than guessing.

## Tests

Unit tests live under `orthanc/tests/unit/MLIntegration/odelia_classification/`
so the standard CI `pytest -m unit` job runs them (a conftest autouse fixture
puts this service's dir on `sys.path` and evicts colliding sibling module names,
matching the other ML services). They cover device resolution, the CUDA
guard, the classification response, and a real `create_model` build+forward
(`ResNet18`) using **random weights** (MONAI's pretrained download is
neutralized, so no network). Run from `orthanc/`:

```bash
pytest tests -m unit
```

Verified green on **both** the pinned MediSwarm stack (monai 1.4 / numpy 1.26 /
torch 2.2) and the CI/platform stack (monai 1.6 / numpy 2.2.6 / torch 2.12) —
only the unused, upstream-broken `Swin3D` baseline fails and is not exercised.

`smoke_test.py` is a quick CLI check against a single model (uses pretrained
weights, so it may download):

```bash
MODEL_NAME=ResNet18 python smoke_test.py     # defaults MODEL_DEVICE=cpu
```

## Scope / deferred

- Trained-weight loading from HuggingFace → **ODV-216** (init weights only here).
- Exact shared single-channel transform → **ODV-217** (`preprocessing.py` is a
  minimal placeholder seam).
- Router/compose template + add-a-model recipe → **ODV-218**.
- Per-model serving/E2E smoke tests → **ODV-219 / ODV-220**.
