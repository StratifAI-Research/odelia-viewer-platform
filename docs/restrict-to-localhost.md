# Restricting published ports to localhost

By design, every published port in `docker-compose.yml` binds to `0.0.0.0`
so colleagues on the same LAN can reach Orthanc / OHIF / chat-middleware /
Grafana for joint research. **That is the desired default.**

If you are running on a multi-tenant network, a VPS, or a cloud VM and you
do *not* want LAN exposure, set the `BIND_HOST` environment variable to
restrict every published port to `127.0.0.1` (loopback only).

## Usage

In `.env` (preferred — gitignored, so it does not leak into commits):

```bash
BIND_HOST=127.0.0.1:
```

…or as a one-shot shell override:

```bash
BIND_HOST=127.0.0.1: docker compose up -d
```

**The trailing colon is required** — each port mapping is interpolated as
`${BIND_HOST:-}<host_port>:<container_port>`, so:

* `BIND_HOST` unset / empty → `8081:8081` (current LAN-shared default)
* `BIND_HOST=127.0.0.1:`    → `127.0.0.1:8081:8081` (loopback only)

Verify with `docker compose config` — every published port should show
`host_ip: 127.0.0.1` when `BIND_HOST` is set.

## How it works

Every port mapping in [`docker-compose.yml`](../docker-compose.yml) is
written as:

```yaml
ports:
  - '${BIND_HOST:-}8081:8081'
```

Docker Compose runs variable interpolation on port strings, so this
single env var flips every published port in the stack at once. No
override files, no parallel definitions to drift.

## Affected ports

| Service                     | Default       | With `BIND_HOST=127.0.0.1:` |
| --------------------------- | ------------- | --------------------------- |
| `viewer` (OHIF)             | `8081`        | `127.0.0.1:8081`            |
| `orthanc-viewer` HTTP       | `8000`        | `127.0.0.1:8000`            |
| `orthanc-viewer` DICOM      | `2000`        | `127.0.0.1:2000`            |
| `orthanc-router-mst` DICOM  | `4243`        | `127.0.0.1:4243`            |
| `orthanc-router-mst` HTTP   | `8043`        | `127.0.0.1:8043`            |
| `mst-classifier`            | `5556`        | `127.0.0.1:5556`            |
| `orthanc-router-medgemma` DICOM | `4244`    | `127.0.0.1:4244`            |
| `orthanc-router-medgemma` HTTP  | `8044`    | `127.0.0.1:8044`            |
| `medgemma-mri`              | `5557`        | `127.0.0.1:5557`            |
| `chat-middleware`           | `5560`        | `127.0.0.1:5560`            |
| `llamacpp-server` (profile) | `8090`        | `127.0.0.1:8090`            |
| `grafana`                   | `3000`        | `127.0.0.1:3000`            |
| `keycloak`                  | `8080`        | `127.0.0.1:8080`            |

## When NOT to use this

- You want a colleague at the next desk (same LAN / VPN) to open the OHIF
  viewer at `http://<your-ip>:8081`. `BIND_HOST=127.0.0.1:` blocks that —
  leave it unset.
- You are running in a cloud VM but already using a managed firewall /
  security group. `BIND_HOST` is belt-and-braces, not a replacement; both
  layers are fine to stack.

## Related hardening

For production hardening beyond just port exposure (passwords, auth,
HTTPS, CORS), see [`production-hardening.md`](production-hardening.md).
