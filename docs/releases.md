# Versions and releases

The ODELIA Viewer Platform is published in numbered releases. A release is a fixed set of
images built from one commit of this repository, so every installation running 2.3.0 runs
the same software. Each release has notes on the
[Releases page](https://github.com/StratifAI-Research/odelia-viewer-platform/releases)
describing what changed and anything you need to do when updating.

To be told about new releases, choose **Watch → Custom → Releases** on the repository
page.

## What a version number means

Versions have three numbers, such as `2.3.1`:

| Number that changes | Example | What it means for you |
| --- | --- | --- |
| Last | 2.3.0 → 2.3.1 | Fixes and small changes. Updating is recommended. |
| Middle | 2.3.1 → 2.4.0 | New features, usually in the viewer as well. Read the release notes before updating. |
| First | 2.4.0 → 3.0.0 | Larger changes. Read the release notes before updating. |

The platform and the ODELIA Viewer share the first two numbers: platform 2.3.x ships
viewer 2.3.x. The last number counts separately in each, so platform 2.3.2 might ship
viewer 2.3.1. Each platform release's notes name the exact viewer version.

## Which version am I running?

- **Viewer:** open **Settings → About** in the viewer.
- **Platform:** run this in your checkout:

  ```bash
  git fetch --tags
  git describe --tags
  ```

  `v2.3.0` means you are on that release. `v2.3.0-4-g1a2b3c4` means you are four commits
  past it on `main`. `docker compose config --images` lists the exact images your
  checkout runs.

## A release or `main`

- **A release** stays exactly as it is until you choose to update. Use one when results
  must be reproducible, for example for the length of a reader study, and record the
  version with your results.
- **`main`** has the latest reviewed changes, including ones not yet in a release.
  [Updating an existing deployment](setup/updating.md) follows `main`.

## Run a specific release

Back up first, as described in [Updating an existing deployment](setup/updating.md).
Commit or `git stash` any local changes to tracked files such as `config/`, and re-apply
them after the checkout. Then:

```bash
docker compose down
git fetch --tags
git checkout v2.3.0
docker compose pull
docker compose up -d --remove-orphans
```

The checked-out `docker-compose.yml` names every image the release runs, under
`x-images` at the top: the platform's own images at that version, plus the viewer and
the AI models pinned by digest. Several services also load configuration files from the
checkout, which is why a release runs from its own checkout. `--remove-orphans` removes
the containers of services a release has retired; their data is kept.

A `docker-compose.override.yml` in your checkout takes precedence over the release's
images. Remove it to run exactly the release; see `docker-compose.override.example.yml`
for running other images on purpose.

To move to a newer release, repeat these steps with its version. To go back to following
`main`, run `git switch main` and follow
[Updating an existing deployment](setup/updating.md).

## The "new version available" notice

When a newer ODELIA Viewer is released, the patient list shows a dialog titled **A new
ODELIA Viewer version is available**, with the installed and the new version. Each user
chooses **Dismiss** or **Release notes** to continue, and does not see the dialog again
until the next release.

- **Users:** the viewer keeps working as before. Let your administrator know.
- **Administrators:** viewer updates arrive through platform releases. Install the newest
  platform release; its notes name the viewer version it ships.

The notice covers the viewer only. A platform release that does not change the viewer
shows no notice, so watch the repository's releases as described above.

Each browser asks GitHub's public API for the latest viewer release once a day. The
request carries no study or user data. Without internet access the check fails silently
and no notice appears. To turn the check off, add this to `window.config` in
`config/app-config.js`:

```js
viewerUpdates: {
  enabled: false,
},
```

To check less or more often, set `checkIntervalHours` there instead (1 to 168, default
24).

## How releases are made

A release is a reviewed pull request that bumps the version. When it is merged, the
project's CI publishes the images it built from that commit under the version number,
then creates the git tag and the release notes. A version always refers to one commit,
and its images are never replaced once published. On Docker Hub, `latest` is always the
newest release. Images are built for linux/amd64 only. Trained AI model images are
released separately; each platform release pins specific ones by digest, and its notes
say which models are included.

Maintainers: the release steps are in [`.github/RELEASING.md`](../.github/RELEASING.md).
