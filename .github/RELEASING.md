# Releasing

Maintainer steps for publishing a new version. For what versions mean to the people
running the platform, see [Versions and releases](../docs/releases.md).

The platform and the [ODELIA Viewer](https://github.com/StratifAI-Research/odelia-viewer)
release the same way: **merge a PR that bumps the version.** The Build and release
workflow in each repository does the rest. There is no release button.

## Version numbers

Both repositories use `MAJOR.MINOR.PATCH`. Git tags carry a `v` (`v2.3.0`); Docker tags
do not (`2.3.0`).

- **Each repository counts on its own.** The viewer and platform numbers are not tied
  together: platform 2.3.2 may ship viewer 2.4.0.
- **A platform release names the viewer it ships** in its notes.
- **The reviewer of a platform PR that moves the viewer pin checks that the stack works**
  with that viewer. If the platform needs changes for it, they go into the same PR.

| Change | Viewer | Platform |
| --- | --- | --- |
| Platform-only fix | unchanged | 2.3.0 → 2.3.1 |
| Viewer fix | 2.3.0 → 2.3.1 | next patch, pinning viewer 2.3.1 |
| Viewer feature that needs platform changes | 2.3.x → 2.4.0 | next version, with the changes and the new pin |

A published version never changes. If something is wrong with it, release the next
patch.

## What the workflow does

| Event | What happens |
| --- | --- |
| Pull request | Builds the images; checks the version bump (see below). Pushes nothing. |
| Merge to `main` | Builds the images and pushes them as `sha-<commit>`. |
| Merge that changes the version | Also tags those same images `X.Y.Z` and `latest`, then creates the `vX.Y.Z` git tag and the GitHub Release. Nothing is rebuilt. |

On Docker Hub, `sha-<commit>` and `X.Y.Z` never change; `latest` always means the
newest release.

## Release the platform

Open one PR in this repository that:

1. Sets `VERSION` to the new version, e.g. `2.3.1`.
2. Sets the three platform images under `x-images` in `docker-compose.yml` to the same
   version, and the router image in `docker-compose.model-template.yml`.
3. When shipping a new viewer or trained model, moves its pin under `x-images`. Look up
   a viewer's digest with:

   ```bash
   docker buildx imagetools inspect stratifai/odelia-viewer:2.3.1 --format '{{.Manifest.Digest}}'
   ```

4. Optionally adds release notes as `.github/release-notes/v2.3.1.md`. Without them,
   the notes list the PRs merged since the previous release, grouped by label as
   configured in [`release.yml`](release.yml), under a line naming the viewer.

The PR check fails if the version is already released or older than the latest
release, if the compose files do not use that version, if the pinned viewer or a
pinned trained model is not on Docker Hub, or if the viewer's tag and digest point at
different images. Once it is approved, merge it.

## Release the viewer

In odelia-viewer, open a PR that bumps `productVersion` in `platform/app/package.json`
and optionally adds `.github/release-notes/vX.Y.Z.md`. Merge it. The GitHub Release,
which makes installed viewers show "new version available", is created only after the
image is on Docker Hub.

When the viewer changed, release it first, then merge the platform PR that pins it. The
platform's PR check refuses a viewer that is not published yet. Users on an older viewer
may see the notice before the platform release exists, so merge the platform PR soon
after.

## When a release fails

- **Re-run it.** Open the failed run and choose Re-run failed jobs. Steps that already
  finished are skipped.
- **If the cause needs a fix on `main`,** merge the fix, then choose Actions → Build and
  release → Run workflow on `main`. It releases the current `main` commit under the
  version that has not been released yet.
- **Do not create `v*` tags by hand.** The workflow creates them, and the
  `release-tags-immutable` ruleset blocks moving or deleting them.

| Message | What to do |
| --- | --- |
| `VERSION must be three numbers like 2.3.1` | Use `X.Y.Z`. |
| `vX.Y.Z is already released` or `must be newer than the latest release` | Choose the next version. |
| `docker-compose files must use X.Y.Z for the platform images` | Update `x-images` and the model template to the new version. |
| `stratifai/odelia-viewer:… is not on Docker Hub` | Release the viewer first, or fix the pin. |
| `stratifai/odelia-classification-…@sha256:… is not on Docker Hub` | Publish the model image first, or fix the digest. |
| `stratifai/odelia-viewer:X.Y.Z is sha256:… on Docker Hub, but docker-compose.yml pins sha256:…` | The digest belongs to another image. Look up that tag's digest again. |
| `cannot resolve stratifai/<image>:sha-<commit>` | The build of that commit failed. Fix it on `main`, then use Run workflow. |
| `… already exists with different content` | That version is taken on Docker Hub. Release the next patch. |

## Trained model images

Trained models are released on their own and are not tied to platform versions. Actions
→ Trained model images builds each model listed in the repository variable
`ODELIA_TRAINED_MODELS` with its pinned Hugging Face weights, checks that the weights
load without network access, and pushes `stratifai/<image>:sha-<commit>-hf-<revision>`
only when `publish` is ticked. A tag that is already published is never pushed again.
Pin the new digest under `x-images` in the next platform release PR. See
[Loading trained model packages](../docs/usage/loading_model_weights.md).
