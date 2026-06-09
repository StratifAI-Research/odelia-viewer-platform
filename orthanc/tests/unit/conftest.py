"""Orthanc test stub.

Installs a fake `orthanc` module into sys.modules before any module under test
imports it. Provides FakeOutput, opt-in REST/DICOM fakes, and per-test reset.
"""
import os
import sys
import tempfile
from types import ModuleType
from typing import Any, Iterator
import pytest

# Allow feedback_db to initialize its SQLite store in a writable temp location.
# PYTEST_ODV133_FEEDBACK_DIR lets a caller pin a specific directory; otherwise
# we create a fresh per-process tmp dir so parallel test runs on the same host
# don't share SQLite state.
_FB_DIR_CREATED_BY_US = False
if not os.environ.get("ORTHANC_FEEDBACK_DB_DIR"):
    _pinned = os.environ.get("PYTEST_ODV133_FEEDBACK_DIR")
    if _pinned:
        os.environ["ORTHANC_FEEDBACK_DB_DIR"] = _pinned
    else:
        os.environ["ORTHANC_FEEDBACK_DB_DIR"] = tempfile.mkdtemp(prefix="odv133_fb_")
        _FB_DIR_CREATED_BY_US = True

if _FB_DIR_CREATED_BY_US:
    import atexit
    import shutil
    _FB_DIR_TO_CLEAN = os.environ["ORTHANC_FEEDBACK_DB_DIR"]
    atexit.register(lambda: shutil.rmtree(_FB_DIR_TO_CLEAN, ignore_errors=True))


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
        # Mirror real Orthanc: StoreKeyValue requires bytes. Silent coercion would
        # mask type bugs (e.g. accidentally passing a dict or unencoded str).
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError(
                f"StoreKeyValue requires bytes; got {type(value).__name__}"
            )
        m._kv[(bucket, key)] = bytes(value)
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
        # Real Orthanc's CreateKeysValuesIterator order is unspecified. Yielding
        # reverse-sorted (rather than ascending) deterministically surfaces tests
        # that depend on alphabetical/ascending iteration.
        items = [(k, v) for (b, k), v in sorted(m._kv.items(), reverse=True) if b == bucket]
        return _KVIterator(items)

    m.StoreKeyValue = _put
    m.GetKeyValue = _get
    m.DeleteKeyValue = _del
    m.CreateKeysValuesIterator = _iter

    # ---- REST + DICOM: default raises; tests bind via fixtures ----
    m.RestApiGet = m.RestApiPost = m.RestApiPut = m.RestApiDelete = _no_orthanc_handler
    m.GetDicomForInstance = _no_orthanc_handler

    # ---- Logging: capture into m._log_calls so tests can assert on log-only side effects ----
    m._log_calls = []  # list of (level, msg) tuples; cleared by _reset_orthanc_state
    def _make_logger(level):
        def _log(msg):
            m._log_calls.append((level, msg))
        return _log
    m.LogInfo = _make_logger("info")
    m.LogWarning = _make_logger("warning")
    m.LogError = _make_logger("error")

    sys.modules['orthanc'] = m


_install_orthanc_stub()

# Make the orthanc/ directory importable for sibling-module tests at this level
_HERE = os.path.dirname(os.path.abspath(__file__))
_ORTHANC_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
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
def out() -> FakeOutput:
    return FakeOutput()


@pytest.fixture(autouse=True)
def _reset_orthanc_state() -> Iterator[None]:
    """Clear KV store + captured callbacks + REST/DICOM stubs between tests."""
    import orthanc
    orthanc._kv.clear()
    orthanc._rest_callbacks.clear()
    orthanc._onchange_callbacks.clear()
    orthanc._log_calls.clear()
    # restore default raisers in case a prior test bound a fake
    orthanc.RestApiGet = orthanc.RestApiPost = orthanc.RestApiPut = orthanc.RestApiDelete = _no_orthanc_handler
    orthanc.GetDicomForInstance = _no_orthanc_handler
    # Evict modules that load under the same bare name from both viewer/ and router/
    # subtrees. Without this, the second side imports a stale module reference from
    # the first side's path. Includes top-level service modules (server, router,
    # feedback_db, feedback_routes, wado_utils) in addition to the `ups` package.
    _EVICT_NAMES = ('ups', 'server', 'router', 'feedback_db', 'feedback_routes', 'wado_utils')
    for key in list(sys.modules):
        if key in _EVICT_NAMES or any(key.startswith(n + '.') for n in _EVICT_NAMES):
            del sys.modules[key]
    yield


@pytest.fixture
def rest_fake() -> Any:
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

    orthanc.RestApiGet = lambda uri: _dispatch('GET', uri)
    orthanc.RestApiPost = lambda uri, body=b'': _dispatch('POST', uri, body)
    orthanc.RestApiPut = lambda uri, body=b'': _dispatch('PUT', uri, body)
    orthanc.RestApiDelete = lambda uri: _dispatch('DELETE', uri)

    return type('RestFake', (), {'calls': calls, 'responses': responses})()


@pytest.fixture
def dicom_fake() -> dict[str, Any]:
    """Bind {instance_id: bytes} for orthanc.GetDicomForInstance calls."""
    import orthanc
    store = {}

    def _get(instance_id):
        if instance_id not in store:
            raise KeyError(f'dicom_fake: no fixture for instance_id={instance_id!r}')
        return store[instance_id]

    orthanc.GetDicomForInstance = _get
    return store

# ---------------------------------------------------------------------------
# WADO-RS fake — opt-in fixture covering both shared.wado_retrieval and
# router.wado_utils. Lives here (root unit conftest) so tests in either subtree
# can request it; the two `setattr` calls are individually guarded so the
# fixture works whether the patch target's module is on sys.path or not.
# ---------------------------------------------------------------------------


@pytest.fixture
def wado_fake(monkeypatch) -> Any:
    """Fake DICOMwebClient injected into shared.wado_retrieval and router.wado_utils.

    Usage:
        def test_x(wado_fake):
            wado_fake.series_responses[("1.2.100", "1.2.200")] = [make_dataset()]
            # call code that uses retrieve_via_wado_rs(...) or retrieve_series_metadata
            assert wado_fake.calls == [("retrieve_series", "1.2.100", "1.2.200")]
    """
    calls = []
    series_responses = {}
    metadata_responses = {}
    # MDR H5: tests can bind concrete exception classes (HTTPError, Timeout,
    # InvalidDicomError, etc.) to specific (study, series) keys so production'''s
    # wrap-as-DicomRetrievalError contract is exercised against the real exception surface.
    series_exceptions = {}      # {(study, series): Exception instance}
    metadata_exceptions = {}    # same

    class FakeDICOMwebClient:
        def __init__(self, url=""):
            self.url = url

        def retrieve_series(self, study_instance_uid="", series_instance_uid=""):
            key = (study_instance_uid, series_instance_uid)
            calls.append(("retrieve_series", *key))
            if key in series_exceptions:
                raise series_exceptions[key]
            if key not in series_responses:
                raise ConnectionError(f"wado_fake: no series response for {key}")
            return series_responses[key]

        def retrieve_series_metadata(self, study_instance_uid="", series_instance_uid=""):
            key = (study_instance_uid, series_instance_uid)
            calls.append(("retrieve_series_metadata", *key))
            if key in metadata_exceptions:
                raise metadata_exceptions[key]
            if key not in metadata_responses:
                raise ConnectionError(f"wado_fake: no metadata response for {key}")
            return metadata_responses[key]

    # Each setattr is independently guarded — the target module is only loaded
    # if the test (or its conftest tree) has put it on sys.path.
    for target in ("shared.wado_retrieval.DICOMwebClient", "wado_utils.DICOMwebClient"):
        try:
            monkeypatch.setattr(target, FakeDICOMwebClient)
        except (ImportError, AttributeError):
            pass

    return type("WadoFake", (), {
        "calls": calls,
        "series_responses": series_responses,
        "metadata_responses": metadata_responses,
        "series_exceptions": series_exceptions,
        "metadata_exceptions": metadata_exceptions,
    })()
