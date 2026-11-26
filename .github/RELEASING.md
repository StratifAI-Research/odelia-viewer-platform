# Releasing

Maintainer steps for publishing a new version. For what versions mean to the people
running the platform, see [Versions and releases](../docs/releases.md).

The platform and the [ODELIA Viewer](https://github.com/StratifAI-Research/odelia-viewer)
release the same way: **merge a PR that bumps the version and turns the changelog's
`[Unreleased]` section into that version.** The Build and release workflow in each
repository does the rest. There is no release button.

Sites update with `git pull origin main`, so `main` of the platform must always be a
release. That is enforced by the rule in the next section.

## Pull requests and the changelog

Every pull request fills in [the template](pull_request_template.md): what and why, a
Yes/No/Unsure answer with a short reason for each impact question, how it was verified,
and whether it needs a release note. A pull request that users or admins would notice
replaces N/A under the right heading of `## [Unreleased]` in
[`CHANGELOG.md`](../CHANGELOG.md) with a line ending in its PR number. Not every such
pull request is a release: the line waits under `[Unreleased]` until the next one.

The Change control check fails when a question is unanswered or has no reason, when
"What and why" or "How it was verified" is empty, when not exactly one release-note box
is ticked, when "Added to the changelog" is ticked but the changelog has no new line, or
when the changelog breaks its format (all eight headings in every version, in order,
each with entries or N/A; see the top of the file). It does not judge the answers. Its
job summary shows them and the approval the change needs:

| Answers | Approval before merge |
| --- | --- |
| All No | one engineering review |
| Yes or Unsure on Verification evidence, Security and data or Existing installations | a code owner acting for QA |
| Yes or Unsure on Hazards, Requirements or AI output | QA approves on the pull request |

Dependabot pull requests skip the description check; the reviewer's approval is the
record. To export the changelog or the merged pull requests with their answers as CSV
or JSON, for example for the change control log:

```bash
python3 .github/scripts/change_control.py export            # changelog entries
python3 .github/scripts/change_control.py changes --since 2026-11-01   # merged PRs, needs gh
```

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

4. In [`CHANGELOG.md`](../CHANGELOG.md), renames `## [Unreleased]` to
   `## [2.3.1] - YYYY-MM-DD` (the planned merge date), adds this PR's own lines and
   anything admins need to do under **Upgrade notes**, and adds a fresh `## [Unreleased]`
   above it with N/A under each of the eight headings. The section, without its N/A
   headings, becomes the GitHub Release text. If it doesn't name the viewer the release ships, a
   line naming it is added.

The PR check fails if the version is already released or older than the latest
release, if the changelog section is missing, not directly below an `[Unreleased]`
that says N/A everywhere, or has no entries, if the compose
files do not use that version, if the pinned viewer or a pinned trained model is not on
Docker Hub, or if the viewer's tag and digest point at different images. Once it is
approved, merge it.

## Release the viewer

In odelia-viewer, open a PR that bumps `VERSION_ODELIA` and turns `## [Unreleased]` in
`CHANGELOG_ODELIA.md` into `## [X.Y.Z] - YYYY-MM-DD` the same way (the repository's
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
| `this PR changes files sites run from the checkout (…)` | Make the PR a release: bump `VERSION`, the platform images in `x-images`, and turn `[Unreleased]` in `CHANGELOG.md` into the new version. |
| `the new version needs a section "## [X.Y.Z] - YYYY-MM-DD" directly below "## [Unreleased]"` | Rename `[Unreleased]` to the new version and add a fresh `[Unreleased]` above it, with N/A under every heading. |
| `[Unreleased] still has entries` | Move them into the new version's section. |
| `[X.Y.Z] has no entries` or `needs exactly these headings` | Describe the change under the right heading; keep all eight headings, in order, with N/A under those with nothing new. |
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
