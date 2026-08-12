# ODELIA generalized model service (ODV-214)

One service image serves one **roster** model. A `MODEL` build-arg selects a
vendored model subunit and prunes the rest, so the runtime serves exactly one
model with no runtime selection. The service reuses the MST service structure
and keeps the `initialize_model` / `analyze_*` contract. `MST-classification/`
is left untouched.

**Roster** (each pinned to its trained config): `MST` plus the five ODELIA
challenge models — `DivideAndConquer`, `BCN_AIM`, `agaldran`, `LME_ABMIL`,
`Pimed`.

## Layout

- `models/` — one **subunit** package dir per roster model, reimplemented
  cleanly from the MediSwarm definitions (matching semantics + `state_dict`
  layout, not a verbatim vendor):
  - `models/<name>/loader.py` — the subunit entry point (contract below).
  - `models/loader_util.py` — subunit discovery + build: `available_models()`,
    `create_model(name)`, `input_size(name)`, `resolve_baked_model()`,
    `assert_forward_contract()`.
  - `models/base.py` — inference bases (`BasicClassifier`, `ModelWrapper`):
    backbone + optional `_class_weight` buffer; training/optimizer/metric
    machinery dropped.
- `preprocessing/` — single-channel inference preprocessing, MediSwarm
  compatible (ODV-217).
- `tools/` — build helpers: `select_subunit.py` (prune to the one baked
  subunit), `bake_weights.py` (stage a trained weights file).
- `model_loader.py` — `build_model(model_name)`: resolves the device and builds
  via `create_model` (init weights).
- `model_service.py` — `ModelService`; retrieval → NIfTI → forward → response.
- `config.py`, `app_refactored.py`, `manifest.json`, `Dockerfile`.
- `smoke_test.py` — build a roster model + forward a dummy volume.

## Subunit contract (adding a model)

A subunit is a package dir under `models/` whose `loader.py` exposes:

```python
NAME: str                         # canonical model name
INPUT_SIZE: tuple[int, int, int]  # (D, H, W) the service preprocesses to
def create(num_classes=3, loss_kwargs=None) -> nn.Module: ...
```

There is no central registry to edit — adding a model is dropping in a subunit
dir; discovery walks `models/*/loader.py`. Discovery reads only `loader.py`
module attributes (no model construction, no network), so a subunit with a
broken heavy dependency cannot break discovery of the others. Sanity-check a
new subunit with `MODEL_NAME=<name> python smoke_test.py`; compose/router
wiring for the new model is ODV-218 (`docs/usage/adding_custom_models.md`).

## Building an image

One image = one model. Build from the `MLIntegration/` directory:

```bash
docker build -f odelia-classification/Dockerfile \
  --build-arg MODEL=BCN_AIM \
  [--build-arg WEIGHT_PATH=weights/bcn_aim.safetensors] \
  -t odelia-model-bcnaim .
```

`MODEL` (required) names the subunit to keep; `tools/select_subunit.py` prunes
the others so exactly one ships. `WEIGHT_PATH` (optional) bakes a trained
weights file; unset builds an init-only image.

## Configuration (env)

| Var            | Default      | Meaning                                            |
|----------------|--------------|----------------------------------------------------|
| `MODEL_NAME`   | *(none)*     | Dev-checkout disambiguation only (see below).      |
| `MODEL_DEVICE` | **required** | `cpu` or `cuda` — must be set (see below).         |
| `PORT`         | `5556`       | HTTP port (ODV-218 allocates per model).           |

Each model is built at its trained config (channels/depth/etc.); `num_classes`
(3) is fixed at the training value and not service-configurable.

### Model selection (`MODEL_NAME`)

The served model is resolved from what is baked into the image, not from env: a
built image carries exactly one subunit, which is served (`MODEL_NAME` is
ignored). In a dev checkout every subunit is present, so `MODEL_NAME` picks the
one to serve or smoke-test; with several subunits and no `MODEL_NAME`, startup
fails loudly rather than guessing.

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
python smoke_test.py                          # Pimed on CPU
MODEL_NAME=DivideAndConquer python smoke_test.py
```

## Scope / deferred

- Trained-weight loading from HuggingFace → **ODV-216** (init weights only here).
- Router/compose template + per-model wiring → **ODV-218**.
- Per-model serving/E2E smoke tests → **ODV-219 / ODV-220**.
