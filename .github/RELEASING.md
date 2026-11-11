# Releasing

Maintainer steps for publishing a new version. For what versions mean to the people
running the platform, see [Versions and releases](../docs/releases.md).

The platform and the [ODELIA Viewer](https://github.com/StratifAI-Research/odelia-viewer)
release the same way: **merge a PR that bumps the version and adds a changelog
section.** The Build and release workflow in each repository does the rest. There is no
release button.

Sites update with `git pull origin main`, so `main` of the platform must always be a
release. That is enforced by the rule in the next section.

## When a platform PR must be a release

Sites take `docker-compose.yml` and the files it mounts into containers straight from
the checkout, but get new images only with a release. A PR that changes any of those
files must therefore bump `VERSION`, or a site would run new files inside old images.
Today that is `docker-compose.yml`, `config/`, the scripts and `orthanc.json` in
`orthanc/viewer/`, `orthanc/router/server.py`, the model `manifest.json`,
`themes/odelia` and `grafana/provisioning`. CI reads the list from compose itself, so a
new mount is covered automatically.

Changes that only reach sites inside an image (chat code, Dockerfiles, Python
dependencies, Dependabot updates) and changes to docs, CI or tests need no bump. They
ship with the next release.

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
| Pull request | Builds the images; checks the version bump (see below). Uploads nothing. |
| Merge to `main` | Builds the images to catch breakage. Uploads nothing. |
| Merge that changes the version | Builds the images and uploads them as `sha-<commit>`, then tags those same images `X.Y.Z` and `latest` and creates the `vX.Y.Z` git tag and the GitHub Release. Its text is the version's changelog section. |

Only releases reach Docker Hub, because sites only run releases. There, `sha-<commit>`
and `X.Y.Z` never change, and `latest` always means the newest release. To try
unreleased code, build it locally: `docker-compose.dev.yml` for the platform images,
`docker build` in odelia-viewer for the viewer.

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

4. Adds a `## 2.3.1` section at the top of [`CHANGELOG.md`](../CHANGELOG.md) saying what
   changed and anything admins need to do. It becomes the GitHub Release text. If it
   doesn't name the viewer the release ships, a line naming it is added.

The PR check fails if the version is already released or older than the latest
release, if the changelog section is missing, not at the top or empty, if the compose
files do not use that version, if the pinned viewer or a pinned trained model is not on
Docker Hub, or if the viewer's tag and digest point at different images. Once it is
approved, merge it.

## Release the viewer

In odelia-viewer, open a PR that bumps `productVersion` in `platform/app/package.json`
and adds a `## X.Y.Z` section at the top of `ODELIA-CHANGELOG.md` (the repository's
`CHANGELOG.md` belongs to OHIF upstream). Merge it. CI creates the GitHub Release, whose
text is that section, only after the image is on Docker Hub.

When the viewer changed, release it first, then merge the platform PR that pins it. The
platform's PR check refuses a viewer that is not published yet. Installed viewers tell
users about new **platform** releases, comparing them with the `VERSION` file the
platform serves as `platform-version.txt`. Viewer 2.3.0 still announces viewer releases,
so while it is pinned, merge the platform PR soon after a viewer release.

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
| `this PR changes files sites run from the checkout (…)` | Make the PR a release: bump `VERSION`, the platform images in `x-images`, and add a changelog section. |
| `CHANGELOG.md must start with a section "## X.Y.Z"` or `… section in CHANGELOG.md is empty` | Add the new version's section at the top of `CHANGELOG.md` and describe the change. |
| `cannot resolve stratifai/<image>:sha-<commit>` | The release build did not upload. Re-run the failed jobs, or fix it on `main` and use Run workflow. |
| `… already exists with different content` | That version is taken on Docker Hub. Release the next patch. |

## Trained model images

Trained models are released on their own and are not tied to platform versions. Actions
→ Trained model images builds each model listed in the repository variable
`ODELIA_TRAINED_MODELS` with its pinned Hugging Face weights, checks that the weights
load without network access, and pushes `stratifai/<image>:sha-<commit>-hf-<revision>`
only when `publish` is ticked. A tag that is already published is never pushed again.
Pin the new digest under `x-images` in the next platform release PR. See
[Loading trained model packages](../docs/usage/loading_model_weights.md).
