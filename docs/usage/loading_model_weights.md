# Loading trained ODELIA model packages

The generalized `odelia-classification` service loads weights baked into its
image. Startup verifies their SHA-256 and strictly restores the model state;
missing tensors, extra tensors, shape mismatches, conflicting parameter aliases
and nonfinite values fail startup. Runtime construction and inference require
no Hub credentials or network downloads.

## Build from a private Hugging Face repository

Use a trusted model package compatible with the selected subunit and the
service's pinned dependencies. The package contains executable Python code;
review it before selecting its immutable commit. It must include:

- `__init__.py`, `loader.py`, and all architecture dependencies. `loader.py`
  exports `NAME`, `INPUT_SIZE`, and `create(num_classes=3, loss_kwargs=None)`.
- `weights.safetensors`, with the complete state dict expected by `create()`.
- `release.json`, containing `model` (matching `NAME`), `release`,
  `weights: "weights.safetensors"`, and `weights_sha256`.

From `orthanc/MLIntegration`, with `MODEL`, `HF_REPO`, and `HF_REVISION` set:

```sh
docker build -f odelia-classification/Dockerfile \
  --build-arg MODEL="$MODEL" \
  --build-arg HF_REPO="$HF_REPO" \
  --build-arg HF_REVISION="$HF_REVISION" \
  --secret id=hf_token,src="$HOME/.cache/huggingface/token" \
  -t odelia-trained-model .
```

`HF_REVISION` must be a full 40-character commit SHA. Use an HF token with read
access to the private repository. The token is read only from the BuildKit
secret; it is not saved in the image or passed to model code. Download and
checksum verification run in the secret-bearing step; source import and strict
loading run in a subsequent step. Image builds do not run inference.

The download replaces the selected subunit and records its repository, commit,
release and weight checksum in `models-lock.json`. The model service reports
this provenance through its model information. Downloaded requirements are
informational: the build uses the service's pinned requirements.

See the [Hub download documentation](https://huggingface.co/docs/huggingface_hub/guides/download)
and [Docker build secrets](https://docs.docker.com/build/building/secrets/).

## Build from local weights

Place an exported safetensors file inside the build context, then use
`--build-arg WEIGHT_PATH=weights/model.safetensors` instead of `HF_REPO` and
`HF_REVISION`. The build records its SHA-256 and validates it against the
selected vendored architecture. Checkpoint conversion and publishing tools are
maintained separately. Do not use `HF_REPO` and `WEIGHT_PATH` together.

## Runtime and preview behavior

Set `MODEL_DEVICE=cpu` or `MODEL_DEVICE=cuda` for runtime. Leave
`ALLOW_INIT_WEIGHTS` unset (or set it to `0`) for trained deployments. A trained
release with missing or invalid weights always fails, even when previews are
allowed.

Images built without weights require `ALLOW_INIT_WEIGHTS=1` to run. The default
Compose roster requires trained weights (`ALLOW_INIT_WEIGHTS=0`). Pimed remains
an explicit development preview behind the `model-previews` profile and is not
registered in the viewer. Preview outputs do not represent a trained model.

For MST, the build vendors a pinned DINOv2 source revision. The constructor uses
that local source with `pretrained=False`; the complete model weights come from
the selected artifact. No secondary pretrained checkpoint is downloaded.

On the inference host, run a separate full-size synthetic forward check:

```sh
docker run --rm --network none \
  -e ALLOW_INIT_WEIGHTS=0 \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  odelia-trained-model python -m tools.validate_model "$MODEL"
```

Use `--load-only` to verify strict weight loading without a forward pass, as
the image build and CI do. Loading still requires memory for the model and its
checkpoint, but avoids inference activation memory and compute.

Passing the synthetic forward check verifies loading and tensor compatibility.
It does not establish the training preprocessing, class ordering, or clinical
performance. Confirm those against the selected release before interpreting
predictions.

## Default viewer deployment

The default Compose deployment starts five trained model/router pairs using
published images pinned by digest. No `.env` or HF token is required. To select
a different release, optionally override its image using `.env.models.example`:

| Viewer label | Architecture | Image variable | Router HTTP / backend port |
| --- | --- | --- | --- |
| MST swarm May 2026 | MST | `MST_IMAGE` | 8043 / 5556 |
| agaldran swarm May 2026 | agaldran | `AGALDRAN_IMAGE` | 8045 / 5558 |
| SwinUNETR swarm May 2026 | BCN_AIM | `BCN_AIM_IMAGE` | 8046 / 5559 |
| ABMIL swarm May 2026 | LME_ABMIL | `LME_ABMIL_IMAGE` | 8048 / 5562 |
| MeisenMeister swarm August 2026 | DivideAndConquer | `DIVIDE_AND_CONQUER_IMAGE` | 8047 / 5561 |

Use the full image reference from a successful trained-model build, preferably
pinned by registry digest. Local image tags also work on their build host.
Set `DIVIDE_AND_CONQUER_PORT` if its default host port 5561 is already in use;
the internal model/router address is unchanged. Export the same value when
running the integration suites.
Unset or empty image variables use the pinned defaults. Update the digests in
`docker-compose.yml` manually when adopting new releases. Building/publishing
these images is a separate step; the public source configuration does not
include private HF release coordinates.

`config/app-config.js` pairs each viewer entry with its corresponding router.
MST keeps `mst-ai` and `orthanc-router-mst`, so existing saved selections
continue to route to the trained backend, now named `odelia-classification-mst`.
There is no second MST pair. Each router waits for its model's health check before
starting.

The MST endpoint now uses `odelia-classification` and returns classification
results and DICOM SRs only. It does not return attention maps, so the router no
longer creates heatmap Secondary Captures for new MST analyses. Existing stored
results remain available. The legacy `MST-classification` source is retained for
reference but is no longer deployed, built or promoted by CI.

The viewer image is also pinned by digest in `docker-compose.yml`. Use
`VIEWER_IMAGE` for a full image reference or `VIEWER_TAG` for a different tag
(optionally including `@sha256:...`). All published defaults use the `stratifai`
namespace. Set `ORTHANC_VIEWER_IMAGE`, `ORTHANC_ROUTER_IMAGE`, or `CHAT_IMAGE`
to a full reference to use another registry or namespace for those services.
`TAG` selects an Orthanc/chat tag under `stratifai` when no full image override
is set; it does not change the trained models or viewer. `DOCKER_HUB_USERNAME`
affects only locally built model previews. Reload the viewer
and reopen the AI panel to pick up added models; user-added endpoints are
preserved. An existing customized MST label may remain, but its original router
URL still reaches trained MST.

Pimed has no swarm weights and remains behind `--profile model-previews`.
It is absent from the default viewer list. The legacy MedGemma MRI service and
router are no longer deployed or published by CI; their source remains available.

## GitHub Actions builds

The `Trained model images` workflow downloads pinned packages and embeds their
weights using the same Dockerfile. It is started manually from `main` and uses the existing
`release` environment, whose deployment branch policy must allow only the
`main` branch. Pull requests continue to build untrained preview images without
HF credentials.

Configure these settings in the platform GitHub repository:

- Environment secret `release/HF_TOKEN`: a fine-grained HF token granting read
  access only to the required private model repositories. Store it only in the
  `release` environment, not as a repository or organization secret, so the
  environment's `main`-only deployment policy controls access.
- Existing environment secrets `release/DOCKER_HUB_USERNAME` and
  `release/DOCKER_HUB_TOKEN`: used only when pushing the validated image.
- Repository variable `ODELIA_TRAINED_MODELS`: a JSON build matrix. For example,
  replace the placeholders in this configuration with the selected release:

  ```json
  {
    "include": [
      {
        "model": "MST",
        "image": "odelia-classification-mst",
        "hf_repo": "ORGANIZATION/PRIVATE-MODEL",
        "hf_revision": "FULL_40_CHARACTER_COMMIT_SHA"
      }
    ]
  }
  ```

An unset matrix disables trained builds. Repository identifiers and commit SHAs
are configuration, not credentials; they can appear in workflow logs and image
provenance. Keep tokens in the `release` environment secrets.

Each build verifies checksums and strict loading, then checks weight loading
in a container with networking disabled. Neither step runs a forward pass.
Full inference and the Orthanc pipeline are validated separately on the
inference host; a successful CI build does not establish inference correctness.
The token is passed
to BuildKit as `hf_token`; it is neither a Docker build argument nor a runtime
environment variable. This is the same secret mount used for local builds.

Pushes to `main` do not trigger this workflow. In GitHub Actions, select
`Trained model images`, choose `Run workflow` on `main`, and leave `publish`
unchecked to build and validate only. Select `publish` explicitly to release
the validated images to
`stratifai/<image>:sha-<application-commit>-hf-<model-commit>`. These
trained tags include both revisions and are separate from the existing
`latest`/semantic-version promotion workflow.

Publishing an image to public Docker Hub makes its embedded weights publicly
accessible, even when the source HF repository is private. Treat each published
trained image as a model release and confirm that the selected weights are
approved for distribution in that form.

Trained builds do not export Actions caches, model files, or build-record
artifacts. Their distribution artifact is the Docker image, including its
weights. The workflow does not create or upload Hugging Face repositories.

See [Docker's GitHub Actions secret mounts](https://docs.docker.com/build/ci/github-actions/secrets/)
and [Hugging Face token permissions](https://huggingface.co/docs/hub/security-tokens).
