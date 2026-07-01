# ODELIA generalized model service (ODV-214)

A single service image that builds any **roster** model by `MODEL_NAME` through
the local `models` package. It reuses the MST service structure and keeps the
`initialize_model` / `analyze_*` contract. `MST-classification/` is left
untouched.

**Roster** (each pinned to its trained config): `MST` plus the five ODELIA
challenge models — `DivideAndConquer`, `BCN_AIM`, `agaldran`, `LME_ABMIL`,
`Pimed`.

## Layout

- `models/` — the roster classifiers, reimplemented cleanly from the MediSwarm
  definitions (matching semantics + `state_dict` layout, not a verbatim vendor):
  - `factory.py` — `create_model(name)` dispatch (challenge models imported lazily).
  - `base.py` — inference bases (`BasicClassifier`, `ModelWrapper`): backbone +
    optional `_class_weight` buffer; training/optimizer/metric machinery dropped.
  - `mst.py` — MST (DINOv2 + transformer fusion).
  - `challenge/<team>/` — the five challenge models, faithfully ported (build +
    output shape + `state_dict` keys preserved so trained checkpoints load, ODV-216).
- `model_loader.py` — `build_model(model_name)`: resolves the device and builds
  via `create_model` (init weights).
- `model_service.py` — `ModelService`; retrieval → NIfTI → forward → response.
- `config.py`, `app_refactored.py`, `manifest.json`, `Dockerfile`.
- `smoke_test.py` — build a roster model + forward a dummy volume.

## Configuration (env)

| Var                | Default   | Meaning                                            |
|--------------------|-----------|----------------------------------------------------|
| `MODEL_NAME`       | `MST`     | Roster model to build (`MST`, `Pimed`, `BCN_AIM`, `DivideAndConquer`, `agaldran`, `LME_ABMIL`). |
| `MODEL_DEVICE`     | **required** | `cpu` or `cuda` — must be set (see below).       |
| `PORT`             | `5556`    | HTTP port (ODV-218 allocates per model).           |

Each model is built at its trained config (channels/depth/etc.); `num_classes`
(3) is fixed at the training value and not service-configurable.

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
matching the other ML services). They cover device resolution (the required
`MODEL_DEVICE` contract), the classification response, and a real `create_model`
build+forward of `Pimed` (a MONAI-ResNet roster model that builds from scratch
with no network). Run from `orthanc/`:

```bash
pytest tests -m unit
```

All six roster models build init-only + forward to `[B, 3]` with matching
`state_dict` keys, verified on both the pinned MediSwarm stack (monai 1.4) and
the CI/platform stack (monai 1.6). Per-model serving/E2E coverage is ODV-219.

`smoke_test.py` is a quick CLI check against a single model:

```bash
MODEL_NAME=Pimed python smoke_test.py        # defaults MODEL_DEVICE=cpu
```

## Scope / deferred

- Trained-weight loading from HuggingFace → **ODV-216** (init weights only here).
- Exact shared single-channel transform → **ODV-217** (`preprocessing.py` is a
  minimal placeholder seam).
- Router/compose template + add-a-model recipe → **ODV-218**.
- Per-model serving/E2E smoke tests → **ODV-219 / ODV-220**.
