"""Orthanc test stub.

Installs a fake `orthanc` module into sys.modules before any module under test
imports it. Provides FakeOutput, opt-in REST/DICOM fakes, and per-test reset.
"""
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
import pytest

# Allow feedback_db to initialize its SQLite store in a writable temp location.
# PYTEST_ODV133_FEEDBACK_DIR lets a caller pin a specific directory; otherwise
# we create a fresh per-process tmp dir so parallel test runs on the same host
# don't share SQLite state.
os.environ.setdefault(
    "ORTHANC_FEEDBACK_DB_DIR",
    os.environ.get("PYTEST_ODV133_FEEDBACK_DIR") or tempfile.mkdtemp(prefix="odv133_fb_"),
)


def _no_orthanc_handler(*a, **kw):
    raise RuntimeError(
        'orthanc REST/DICOM call from a test that did not request the rest_fake '
        'or dicom_fake fixture; bind responses explicitly'
    )


def _install_orthanc_stub():
    if 'orthanc' in sys.modules and getattr(sys.modules['orthanc'], '_is_test_stub', False):
        return

    m = ModuleType('orthanc')
    m._is_test_stub = True

    # ---- Constants ----
    class _ChangeType:
        STABLE_STUDY = 'STABLE_STUDY'
        STABLE_SERIES = 'STABLE_SERIES'
        NEW_INSTANCE = 'NEW_INSTANCE'
        STABLE_PATIENT = 'STABLE_PATIENT'
    m.ChangeType = _ChangeType

    class _ResourceType:
        STUDY = 'STUDY'
        SERIES = 'SERIES'
        INSTANCE = 'INSTANCE'
        PATIENT = 'PATIENT'
    m.ResourceType = _ResourceType

    # ---- Callback registration: capture for inspection ----
    m._rest_callbacks = []
    m._onchange_callbacks = []
    m.RegisterRestCallback = lambda uri, fn: m._rest_callbacks.append((uri, fn))
    m.RegisterOnChangeCallback = m._onchange_callbacks.append

    # ---- KV store: backed by a dict; tests can inspect _kv directly ----
    m._kv = {}  # {(bucket, key): bytes}

    def _put(bucket, key, value):
        m._kv[(bucket, key)] = value if isinstance(value, bytes) else str(value).encode()
    def _get(bucket, key):
        return m._kv.get((bucket, key))
    def _del(bucket, key):
        m._kv.pop((bucket, key), None)

    class _KVIterator:
        """Mirrors Orthanc's iterator API: Next() -> bool, GetKey() -> str, GetValue() -> bytes.

        GetKey/GetValue raise RuntimeError if called outside a valid Next() position
        (i.e., before the first Next() or after Next() returned False).
        """
        def __init__(self, items):
            self._items = items   # list of (key, value)
            self._idx = -1
            self._valid = False

        def Next(self):
            self._idx += 1
            self._valid = self._idx < len(self._items)
            return self._valid

        def GetKey(self):
            if not self._valid:
                raise RuntimeError('_KVIterator: GetKey() called outside a valid Next() position')
            return self._items[self._idx][0]

        def GetValue(self):
            if not self._valid:
                raise RuntimeError('_KVIterator: GetValue() called outside a valid Next() position')
            return self._items[self._idx][1]

    def _iter(bucket):
        items = [(k, v) for (b, k), v in sorted(m._kv.items()) if b == bucket]
        return _KVIterator(items)

    m.StoreKeyValue = _put
    m.GetKeyValue = _get
    m.DeleteKeyValue = _del
    m.CreateKeysValuesIterator = _iter

    # ---- REST + DICOM: default raises; tests bind via fixtures ----
    m.RestApiGet = m.RestApiPost = m.RestApiPut = m.RestApiDelete = _no_orthanc_handler
    m.GetDicomForInstance = _no_orthanc_handler

    # ---- Logging: no-op ----
    m.LogInfo = m.LogWarning = m.LogError = lambda msg: None

    sys.modules['orthanc'] = m


_install_orthanc_stub()

# Make the orthanc/ directory importable for sibling-module tests at this level
_ORTHANC_DIR = str(Path(__file__).resolve().parents[2])
if _ORTHANC_DIR not in sys.path:
    sys.path.insert(0, _ORTHANC_DIR)


# =====================================================================
# Helpers & fixtures
# =====================================================================

class FakeOutput:
    """Captures Orthanc Output object method calls."""
    def __init__(self):
        self.status = None
        self.body = None
        self.content_type = None
        self.allowed = None

    def AnswerBuffer(self, body, content_type):
        self.status, self.body, self.content_type = 200, body, content_type

    def SendHttpStatus(self, code, body=''):
        self.status, self.body = code, body

    def SendMethodNotAllowed(self, allowed):
        self.status, self.allowed = 405, allowed


@pytest.fixture
def out():
    return FakeOutput()


@pytest.fixture(autouse=True)
def _reset_orthanc_state():
    """Clear KV store + captured callbacks + REST/DICOM stubs after each test.

    Initial state is clean because _install_orthanc_stub() seeds empty containers
    and default _no_orthanc_handler raisers; teardown brings the module back to
    that baseline so the next test starts identically.
    """
    yield
    import orthanc
    orthanc._kv.clear()
    orthanc._rest_callbacks.clear()
    orthanc._onchange_callbacks.clear()
    # restore default raisers in case the test bound a fake
    orthanc.RestApiGet = orthanc.RestApiPost = orthanc.RestApiPut = orthanc.RestApiDelete = _no_orthanc_handler
    orthanc.GetDicomForInstance = _no_orthanc_handler
    # Drop ALL `ups`-prefixed entries — both viewer/ups and router/ups land in
    # sys.modules under the bare `ups` name once their side imports first; evict
    # between tests so the next test's side imports cleanly.
    for key in [k for k in sys.modules if k == 'ups' or k.startswith('ups.')]:
        del sys.modules[key]


@pytest.fixture
def rest_fake(monkeypatch):
    """Records orthanc.RestApi* calls and lets tests bind responses.

    Usage:
        def test_x(rest_fake):
            rest_fake.responses[('GET', '/studies/abc')] = b'{"foo":1}'
            ... call code that invokes orthanc.RestApiGet('/studies/abc') ...
            assert rest_fake.calls == [('GET', '/studies/abc', None)]
    """
    import orthanc
    calls = []
    responses = {}

    def _dispatch(method, uri, body=None):
        calls.append((method, uri, body))
        key = (method, uri)
        if key not in responses:
            raise RuntimeError(f'rest_fake: no response bound for {method} {uri}')
        v = responses[key]
        return v(body) if callable(v) else v

    monkeypatch.setattr(orthanc, "RestApiGet", lambda uri: _dispatch("GET", uri))
    monkeypatch.setattr(orthanc, "RestApiPost", lambda uri, body=b"": _dispatch("POST", uri, body))
    monkeypatch.setattr(orthanc, "RestApiPut", lambda uri, body=b"": _dispatch("PUT", uri, body))
    monkeypatch.setattr(orthanc, "RestApiDelete", lambda uri: _dispatch("DELETE", uri))

    return type('RestFake', (), {'calls': calls, 'responses': responses})()


@pytest.fixture
def dicom_fake(monkeypatch):
    """Bind {instance_id: bytes} for orthanc.GetDicomForInstance calls."""
    import orthanc
    store = {}

    def _get(instance_id):
        if instance_id not in store:
            raise KeyError(f'dicom_fake: no fixture for instance_id={instance_id!r}')
        return store[instance_id]

    monkeypatch.setattr(orthanc, "GetDicomForInstance", _get)
    return store
