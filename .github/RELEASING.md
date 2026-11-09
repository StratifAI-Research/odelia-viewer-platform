# Releasing

Maintainer steps for publishing a new version. For what versions mean to the people
running the platform, see [Versions and releases](../docs/releases.md).

The ODELIA Viewer and the platform are released separately, from their own repositories,
with a manual **Docker Build and Push** run on `main` in each. Pushing a git tag does not
release anything. Docker Hub credentials are secrets of the `release` environment, which
only `main` can use.

## Version numbers

Both repositories use `MAJOR.MINOR.PATCH`. Git tags carry a `v` (`v2.3.0`); Docker tags
do not (`2.3.0`).

- **MAJOR.MINOR is shared.** A release that changes both the viewer and the platform
  raises the minor number in both, e.g. 2.3.x → 2.4.0.
- **PATCH counts independently** in each repository.
- **The platform release notes name the viewer version** the platform ships.

| Change | Viewer | Platform |
| --- | --- | --- |
| Platform-only fix | unchanged | 2.3.0 → 2.3.1 |
| Viewer fix | 2.3.0 → 2.3.1 | next patch, pinning viewer 2.3.1 |
| Feature release touching both | 2.3.x → 2.4.0 | 2.3.x → 2.4.0 |

Versions are always three numbers. A published version is never moved: if something is
wrong with it, release the next patch.

## What every merge to `main` does

- **Platform:** builds `odelia-orthanc-viewer`, `odelia-orthanc-router` and `odelia-chat`,
  pushes each as `sha-<commit>`, then moves `latest` once all three succeed. The
  `odelia-classification-*` preview images are built to catch breakage but never pushed.
- **Viewer:** builds and pushes `stratifai/odelia-viewer:latest`.

Deployments do not change: `docker-compose.yml` pins images by version and digest, not
`latest`.

## Release order when the viewer changed

1. Release the viewer image (viewer steps 1–3 below).
2. Release the platform with the new viewer pinned.
3. Publish the viewer's GitHub Release (viewer step 4).

The viewer shows a "new version available" notice when the viewer repository gets a new
GitHub Release, and the notice tells users to contact their administrator. Publishing it
last means administrators already have a platform release to upgrade to.

## Release the viewer

In [StratifAI-Research/odelia-viewer](https://github.com/StratifAI-Research/odelia-viewer):

**1. Set the version.** Open a PR that sets `productVersion` in
`platform/app/package.json`, e.g. `2.3.1`, and merge it.

**2. Tag the merge commit.**

```bash
git fetch origin
git tag -a v2.3.1 -m "ODELIA Viewer 2.3.1" origin/main
git push origin v2.3.1
```

**3. Build the release image.** Actions → Docker Build and Push → Run workflow on `main`
with `release_tag` set to `v2.3.1`, or:

```bash
gh workflow run docker-build-push.yml -R StratifAI-Research/odelia-viewer --ref main -f release_tag=v2.3.1
```

The run checks that the tag equals `v` + `productVersion` and points at the tip of
`main`, then builds and pushes `stratifai/odelia-viewer:2.3.1`, `:2.3` and `:2`.

**4. Publish the GitHub Release** once the image is on Docker Hub and, if the viewer
ships with a platform release, after that release:

```bash
gh release create v2.3.1 -R StratifAI-Research/odelia-viewer --verify-tag --title v2.3.1 --generate-notes
```

The workflow does not create it. Without it, users are never told about the new version.

## Release the platform

In this repository:

**1. Prepare `main`.** In one PR:

- If the viewer changed, pin it. Look up its digest and use `2.3.1@sha256:<digest>` as
  the `VIEWER_TAG` default in `docker-compose.yml` and in the `.env.models.example`
  example:

  ```bash
  docker buildx imagetools inspect stratifai/odelia-viewer:2.3.1 --format '{{.Manifest.Digest}}'
  ```

- Optionally write `.github/release-notes/vX.Y.Z.md`. Without it, the notes are
  generated from the titles of PRs merged since the previous release, grouped by label
  as configured in [`release.yml`](release.yml). Write one when PR titles do not tell
  operators what to do.

**2. Wait for `main` to build.** The Docker Build and Push run for the merge commit must
succeed; it pushes the `sha-<commit>` images the release reuses.

**3. Run the release.** Actions → Docker Build and Push → Run workflow on `main` with
`version` set to `2.3.1` (no `v`), or:

```bash
gh workflow run docker-build-push.yml --ref main -f version=2.3.1
```

The run releases whatever commit is at the tip of `main`, so do not merge anything in
between. It does not rebuild. It tags that commit's three `sha-<commit>` images as
`2.3.1`, creates the git tag `v2.3.1` on the commit, and creates the GitHub Release.

**4. Adopt it in Compose (optional).** The release does not change `docker-compose.yml`.
Operators run a release by checking out its tag and setting `TAG`, as described in
[Run a specific release](../docs/releases.md#run-a-specific-release). To make the
default deployment on `main` use the release, pin `2.3.1@sha256:<digest>` for the three
platform images in `docker-compose.yml` and `docker-compose.model-template.yml` in a
follow-up PR.

## When a release run fails

Rerunning the same version for the same commit is safe: steps that already finished are
skipped. Do not create platform `v*` tags by hand; the workflow creates them, and the
`release-tags-immutable` ruleset blocks moving or deleting them.

| Message | What to do |
| --- | --- |
| `version must be three numbers like 2.3.0` | Enter `X.Y.Z`. |
| `cannot resolve stratifai/<image>:sha-<commit>` | The `main` build for that commit is still running or failed. Wait for it or fix it, then rerun. |
| `git tag vX.Y.Z already exists on <sha>, not <sha>` | The version belongs to another commit. Release the next patch. |
| `stratifai/<image>:X.Y.Z already exists and points at different content` | Same: release the next patch. |
| `a GitHub Release vX.Y.Z exists but its git tag is gone` | Restore or delete that GitHub Release, then rerun. |
| Viewer: `Release tag must match the stable productVersion` | `release_tag` must be `v` + `productVersion`. |
| Viewer: `Release tag must point to the main commit selected for this run` | Something merged after the tag was created. Bump `productVersion` to the next patch, merge, tag that commit and run again. |

## Trained model images

Trained models are released on their own and are not tied to platform versions. Actions
→ Trained model images builds each model listed in the repository variable
`ODELIA_TRAINED_MODELS` with its pinned Hugging Face weights, checks that the weights load
without network access, and pushes `stratifai/<image>:sha-<commit>-hf-<revision>` only
when `publish` is ticked. Pin the new digest in `docker-compose.yml` and
`.env.models.example` in a PR. See
[Loading trained model packages](../docs/usage/loading_model_weights.md).
