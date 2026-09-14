# Security configuration and deployment hardening

The default stack supports local research and LAN collaboration. The viewer login does
not enforce backend authorization: Orthanc authentication is disabled, nginx proxies are
not OIDC resource servers, and chat REST/WebSocket sessions are not owned by authenticated
subjects. Neither a UI login nor a 300-second token lifetime closes these paths.

## Implemented controls

- The imported public `ohif_viewer` client requires Authorization Code with PKCE S256;
  implicit flow, direct grants and service accounts remain disabled.
- Conditional OTP is already bound into browser/forms login. Enrolled users are challenged;
  enrollment is optional. No new password policy or brute-force lockout is enabled.
- Active MST code/configuration/weights have an exact revision and expected SHA-256 checksums,
  checked before loading, including local configuration/weight loading.
- The viewer image checks the publisher's oauth2-proxy executable digest before installation.
  This is not a deployment image-signature verification gate.
- nginx access logs omit query strings and referrers. Auth callback and chat/feedback responses
  use `no-store`. Referrer policy and MIME-sniffing protection are present. Other application
  logs and URL path identifiers are not promised to be free of sensitive information.
- Credential overrides, loopback binding, outbound routing allowlists and browser WebSocket
  origin checks are available without changing default deployment behavior.

See the README **Updating** section and [update guide](../setup/updating.md) for exact
activation and rollback steps. Existing Keycloak realms are not updated by startup import.

## Controls requiring deployment configuration

| Area | Current default and required operational work |
| --- | --- |
| Authentication/privileges | Define and implement an end-to-end resource-server boundary, machine callers and chat session ownership before claiming enforced viewer/admin roles. `viewer` and `pacsadmin` are seeded usernames; `pacsadmin` is also a group. |
| Ports | Services, including AI backends, publish host ports by default. Use `BIND_HOST`, selected ingress and firewall/egress policy appropriate to LAN/PACS access. |
| Routing | Set `ROUTER_HOST_ALLOWLIST` on viewer/router/model services. Allowed hosts can be private Docker/PACS names. Redirects are refused on protected requests when enabled. DNS rebinding and other protocol paths require network controls. |
| Chat/debug | Keep the debug API: model/configuration/cache/session endpoints serve the existing UI. Optional WebSocket Origin checks and CORS limit browsers, not authenticated access. Never treat them as session ownership or authorization. |
| TLS | The committed stack uses HTTP, internal HTTP/JDBC and unprotected DICOM. Configure TLS at your ingress with TLS 1.2 minimum and TLS 1.3 support, approved suites and certificate lifecycle. DICOM peers need coordinated TLS/VPN configuration. |
| Passwords/MFA | No password policy; brute-force protection is off despite dormant threshold/wait values. Apply an organization-approved policy deliberately. Optional OTP works; mandatory enrollment changes login UX. |
| Keys | Keycloak's generated providers persist keys in PostgreSQL. The committed provider configuration does not contain signing private keys. Establish backup and rotation procedures. |
| Secrets | Demo passwords are public defaults. Environment overrides are supported; they do not rotate initialized accounts. Use a managed secret mechanism where required and coordinate database rotation. |
| Data/backup encryption | No stack-managed encryption or backup schedule is configured. Host encryption, encrypted backups, retention and restore verification need deployment evidence. |
| Images/models | Checksum checks establish expected bytes, not publisher signatures. Establish trusted image identities and signature enforcement; validate upgrades against inference and browser regressions. Preview roster checksum recording is not trained-model runtime verification. |
| Reviews/lifecycle | Specify owners and policy for access review, inactive accounts, certificates, signing keys and dependency refresh. Source defaults cannot define an organization's approved intervals. |

The realm runs in Keycloak 24.0.5 `start-dev` mode. Production startup/TLS configuration
and brute-force protection are separate controls; changing the startup command does not
itself establish a realm lockout policy. Keep any version/startup-mode migration explicit.
