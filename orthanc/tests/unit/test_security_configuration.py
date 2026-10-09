import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]


def module(path):
    spec = importlib.util.spec_from_file_location("security_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("role", ["viewer", "router"])
def test_routing_policy( role, monkeypatch):
    policy = module(ROOT / "orthanc" / role / "host_allowlist.py")
    monkeypatch.delenv("ROUTER_HOST_ALLOWLIST", raising=False)
    assert policy.host_is_allowed("http://custom-pacs:8042")
    monkeypatch.setenv("ROUTER_HOST_ALLOWLIST", "orthanc-viewer, PACS")
    for target in ["http://orthanc-viewer:8042/dicom-web", "https://pacs:443", "pacs:4242/AET"]:
        assert policy.host_is_allowed(target)
    for target in ["http://evil", "file://pacs/path", "http://user@pacs", "http://pacs:99999", "http://pacs#x", "http://pacs\\evil", "http://pacs evil"]:
        assert not policy.host_is_allowed(target)


def test_realm_preserves_login_sessions_and_optional_otp():
    realm = json.loads((ROOT / "config/ohif-keycloak-realm.json").read_text())
    client = next(c for c in realm["clients"] if c["clientId"] == "ohif_viewer")
    assert client["attributes"]["pkce.code.challenge.method"] == "S256"
    assert client["publicClient"] and client["standardFlowEnabled"]
    assert not any(client[k] for k in ["implicitFlowEnabled", "directAccessGrantsEnabled", "serviceAccountsEnabled"])
    assert [realm[k] for k in ["accessTokenLifespan", "ssoSessionIdleTimeout", "ssoSessionMaxLifespan"]] == [300, 1800, 36000]
    forms = next(f for f in realm["authenticationFlows"] if f["alias"] == "forms")
    assert any(e.get("flowAlias") == "Browser - Conditional OTP" and e["requirement"] == "CONDITIONAL" for e in forms["authenticationExecutions"])
    assert not next(a for a in realm["requiredActions"] if a["alias"] == "CONFIGURE_TOTP")["defaultAction"]
