"""Exercise redirect refusal through the actual DICOMweb HTTP transport."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest


@pytest.fixture
def redirect_server():
    hits = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            if self.path == "/redirect-target":
                hits.append(self.path)
                body = json.dumps([{
                    "00200032": {"vr": "DS", "Value": [0, 0, 0]},
                    "00200013": {"vr": "IS", "Value": [1]},
                }]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/dicom+json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{self.server.server_port}/redirect-target")
                self.end_headers()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", hits
    server.shutdown(); server.server_close(); thread.join()


def test_configured_allowlist_stops_redirect_before_second_host(redirect_server, monkeypatch):
    import requests
    from wado_utils import retrieve_series_metadata_sorted
    url, hits = redirect_server
    monkeypatch.setenv("ROUTER_HOST_ALLOWLIST", "127.0.0.1")
    with pytest.raises(requests.exceptions.TooManyRedirects):
        retrieve_series_metadata_sorted([{"retrieval_url": url, "study_uid": "1.2", "series_uid": "1.3"}])
    assert hits == []


def test_unconfigured_transport_retains_redirect_compatibility(redirect_server, monkeypatch):
    from wado_utils import retrieve_series_metadata_sorted
    url, hits = redirect_server
    monkeypatch.delenv("ROUTER_HOST_ALLOWLIST", raising=False)
    _, positions, _ = retrieve_series_metadata_sorted([{"retrieval_url": url, "study_uid": "1.2", "series_uid": "1.3"}])
    assert positions == [[0, 0, 0]]
    assert hits == ["/redirect-target"]
