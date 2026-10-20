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
checksum verification run in the secret-bearing step; source import, strict
loading and a full-size synthetic forward check run in a subsequent step.

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
Compose model roster sets this explicitly for development previews. Set it to
`0` in trained deployment overrides. Preview outputs do not represent a
trained model.

For MST, the build vendors a pinned DINOv2 source revision. The constructor uses
that local source with `pretrained=False`; the complete model weights come from
the selected artifact. No secondary pretrained checkpoint is downloaded.

Passing the synthetic forward check verifies loading and tensor compatibility.
It does not establish the training preprocessing, class ordering, or clinical
performance. Confirm those against the selected release before interpreting
predictions.

## GitHub Actions builds

The `Trained model images` workflow downloads pinned packages and embeds their
weights using the same Dockerfile. It runs only from `main` and uses the existing
`release` environment, whose deployment branch policy must allow only the
`main` branch. Pull requests continue to build untrained preview images without
HF credentials.

Configure these settings in the platform GitHub repository:

- Environment secret `release/HF_TOKEN`: preferably a fine-grained HF token
  granting read access only to the required private model repositories.
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
provenance. Keep tokens in environment secrets.

Each build first verifies checksums and strict loading, then runs full-size
synthetic inference in a container with networking disabled. The token is passed
to BuildKit as `hf_token`; it is neither a Docker build argument nor a runtime
environment variable. This is the same secret mount used for local builds.

Main-branch pushes publish successful builds to
`stratifai/<image>:sha-<application-commit>-hf-<model-commit>`. Manual dispatch
builds and validates by default; select `publish` to push it as well. These
trained tags include both revisions and are separate from the existing
`latest`/semantic-version promotion workflow.

Trained builds do not export Actions caches, model files, or build-record
artifacts. Their distribution artifact is the Docker image, including its
weights. The workflow does not create or upload Hugging Face repositories.

See [Docker's GitHub Actions secret mounts](https://docs.docker.com/build/ci/github-actions/secrets/)
and [Hugging Face token permissions](https://huggingface.co/docs/hub/security-tokens).
