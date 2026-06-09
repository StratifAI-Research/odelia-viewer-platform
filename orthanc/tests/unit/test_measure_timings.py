"""Smoke tests for measure_timings.py.

Per spec: this is a profiling driver, not a library. We cover only the
side-effect-free helpers and the TimingProfiler accumulator.
Functions that require docker, network I/O, or running Orthanc instances
(get_study_info, send_to_ai_and_profile, fetch_component_logs, etc.) are
excluded deliberately.
"""
import pytest
import requests




# ---------------------------------------------------------------------------
# create_http_session
# ---------------------------------------------------------------------------

def test_create_http_session_returns_requests_session():
    from measure_timings import create_http_session
    s = create_http_session()
    assert isinstance(s, requests.Session)


def test_create_http_session_mounts_http_and_https():
    from measure_timings import create_http_session
    s = create_http_session()
    # Session should have adapters mounted for http:// and https://
    assert 'http://' in s.adapters
    assert 'https://' in s.adapters


# ---------------------------------------------------------------------------
# TimingProfiler — __init__, record, measurements accumulation
# ---------------------------------------------------------------------------

def test_timing_profiler_construct(tmp_path):
    from measure_timings import TimingProfiler
    prof = TimingProfiler(trace_id='test-trace-01', output_dir=str(tmp_path))
    assert prof is not None
    assert prof.trace_id == 'test-trace-01'


def test_timing_profiler_starts_empty(tmp_path):
    from measure_timings import TimingProfiler
    prof = TimingProfiler(trace_id='t1', output_dir=str(tmp_path))
    assert prof.measurements == []


def test_timing_profiler_record_adds_measurement(tmp_path):
    from measure_timings import TimingProfiler
    prof = TimingProfiler(trace_id='t2', output_dir=str(tmp_path))
    prof.record(component='router', operation='fetch', duration_ms=123.4)
    assert len(prof.measurements) == 1


def test_timing_profiler_record_stores_fields(tmp_path):
    from measure_timings import TimingProfiler
    prof = TimingProfiler(trace_id='t3', output_dir=str(tmp_path))
    prof.record(component='model', operation='infer', duration_ms=250.0)
    m = prof.measurements[0]
    assert m['component'] == 'model'
    assert m['operation'] == 'infer'
    assert m['duration_ms'] == 250.0
    assert m['trace_id'] == 't3'


def test_timing_profiler_record_multiple(tmp_path):
    from measure_timings import TimingProfiler
    prof = TimingProfiler(trace_id='t4', output_dir=str(tmp_path))
    prof.record('router', 'fetch', 100.0)
    prof.record('model', 'infer', 200.0)
    prof.record('router', 'send', 50.0)
    assert len(prof.measurements) == 3


def test_timing_profiler_record_with_metadata(tmp_path):
    from measure_timings import TimingProfiler
    prof = TimingProfiler(trace_id='t5', output_dir=str(tmp_path))
    prof.record('router', 'fetch', 99.9, metadata={'key': 'value'})
    m = prof.measurements[0]
    # metadata field is json-serialized in record()
    assert 'key' in m['metadata']


# ---------------------------------------------------------------------------
# parse_timing_logs — exercises real parsing logic
#
# parse_timing_logs(profiler, container, logs) parses TIMING:/PROFILE: markers
# from docker-style log output. The container name is mapped via component_map.
# Lines without a recognised container name fall through as-is.
# ---------------------------------------------------------------------------

def test_parse_timing_logs_handles_empty_input(tmp_path):
    from measure_timings import TimingProfiler, parse_timing_logs
    prof = TimingProfiler(trace_id='p0', output_dir=str(tmp_path))
    parse_timing_logs(prof, 'some-container', '')
    # Should not raise and should add no measurements
    assert prof.measurements == []


def test_parse_timing_logs_no_timing_markers(tmp_path):
    from measure_timings import TimingProfiler, parse_timing_logs
    prof = TimingProfiler(trace_id='p1', output_dir=str(tmp_path))
    logs = "INFO: loading model\nDEBUG: checkpoint loaded\n"
    parse_timing_logs(prof, 'odelia-orthanc-router', logs)
    assert prof.measurements == []


def test_parse_timing_logs_recognises_timing_marker(tmp_path):
    from measure_timings import TimingProfiler, parse_timing_logs
    prof = TimingProfiler(trace_id='p2', output_dir=str(tmp_path))
    # A plain line (no docker timestamp) containing TIMING: marker
    # parse_timing_logs falls back to using the full line when there is no
    # leading docker timestamp — so a bare TIMING: line is always included.
    logs = "TIMING: dicom_send: 375.50ms\n"
    parse_timing_logs(prof, 'odelia-orthanc-router', logs)
    assert len(prof.measurements) == 1
    m = prof.measurements[0]
    assert m['operation'] == 'dicom_send'
    assert m['duration_ms'] == 375.5


def test_parse_timing_logs_maps_container_to_component(tmp_path):
    from measure_timings import TimingProfiler, parse_timing_logs
    prof = TimingProfiler(trace_id='p3', output_dir=str(tmp_path))
    logs = "TIMING: inference_run: 800.00ms\n"
    parse_timing_logs(prof, 'odelia-breast-cancer-classification', logs)
    assert len(prof.measurements) == 1
    assert prof.measurements[0]['component'] == 'ml-breast-cancer'


def test_parse_timing_logs_profile_marker_also_works(tmp_path):
    from measure_timings import TimingProfiler, parse_timing_logs
    prof = TimingProfiler(trace_id='p4', output_dir=str(tmp_path))
    logs = "PROFILE: weight_load: 1200.00ms\n"
    parse_timing_logs(prof, 'odelia-mst-classifier', logs)
    assert len(prof.measurements) == 1
    assert prof.measurements[0]['duration_ms'] == 1200.0


def test_parse_timing_logs_seconds_unit_converted(tmp_path):
    from measure_timings import TimingProfiler, parse_timing_logs
    prof = TimingProfiler(trace_id='p5', output_dir=str(tmp_path))
    # Parser also handles "N s" (seconds) format and converts to ms
    logs = "TIMING: long_op: 2.5s\n"
    parse_timing_logs(prof, 'odelia-orthanc-router', logs)
    assert len(prof.measurements) == 1
    assert prof.measurements[0]['duration_ms'] == pytest.approx(2500.0)


def test_parse_timing_logs_unknown_container_uses_name_as_is(tmp_path):
    from measure_timings import TimingProfiler, parse_timing_logs
    prof = TimingProfiler(trace_id='p6', output_dir=str(tmp_path))
    logs = "TIMING: some_op: 42.0ms\n"
    parse_timing_logs(prof, 'my-unknown-container', logs)
    assert len(prof.measurements) == 1
    assert prof.measurements[0]['component'] == 'my-unknown-container'


# ---------------------------------------------------------------------------
# T1: docker-timestamp filtering, metadata bracket extraction, non-unit skip,
#     exception-swallow path
# ---------------------------------------------------------------------------

def test_parse_timing_logs_skips_lines_before_profiler_start(tmp_path):
    """Docker-timestamped lines older than profiler.start_datetime are filtered out."""
    import measure_timings
    p = measure_timings.TimingProfiler(trace_id='t1', output_dir=str(tmp_path))
    # Force a known start_datetime so the input timestamps fall before it.
    from datetime import datetime
    p.start_datetime = datetime(2099, 1, 1, 0, 0, 0)
    logs = "2024-11-12T13:56:54.123456789Z TIMING: too_old_op: 100.0ms"
    measure_timings.parse_timing_logs(p, "orthanc-viewer", logs)
    assert p.measurements == []      # nothing recorded - line was filtered


def test_parse_timing_logs_includes_lines_after_profiler_start(tmp_path):
    """Docker-timestamped lines newer than profiler.start_datetime are recorded."""
    import measure_timings
    from datetime import datetime
    p = measure_timings.TimingProfiler(trace_id='t1', output_dir=str(tmp_path))
    p.start_datetime = datetime(2000, 1, 1, 0, 0, 0)
    logs = "2024-11-12T13:56:54.123456789Z TIMING: recent_op: 200.0ms"
    measure_timings.parse_timing_logs(p, "orthanc-viewer", logs)
    assert len(p.measurements) == 1
    assert p.measurements[0]["operation"] == "recent_op"
    assert p.measurements[0]["duration_ms"] == 200.0


def test_parse_timing_logs_extracts_metadata_from_brackets(tmp_path):
    """[key:"val"] after duration is parsed as JSON and stored in metadata."""
    import json
    import measure_timings
    p = measure_timings.TimingProfiler(trace_id='t1', output_dir=str(tmp_path))
    logs = 'TIMING: op_with_meta: 12.5ms ["phase":"warmup","id":7]'
    measure_timings.parse_timing_logs(p, "orthanc-viewer", logs)
    assert len(p.measurements) == 1
    meta = json.loads(p.measurements[0]["metadata"])
    assert meta == {"phase": "warmup", "id": 7}


def test_parse_timing_logs_skips_lines_without_ms_or_s_unit(tmp_path):
    """A TIMING line whose duration contains neither ms nor s hits the else-continue skip."""
    import measure_timings
    p = measure_timings.TimingProfiler(trace_id='t1', output_dir=str(tmp_path))
    logs = "TIMING: op_no_unit: 12.5xyz"
    measure_timings.parse_timing_logs(p, "orthanc-viewer", logs)
    assert p.measurements == []


def test_parse_timing_logs_swallows_unparseable_timing_line(tmp_path):
    """When the inner float parse raises, the outer try swallows it without raising."""
    import measure_timings
    p = measure_timings.TimingProfiler(trace_id='t1', output_dir=str(tmp_path))
    logs = "TIMING: op_garbled: not-a-number-ms"
    # Must NOT raise; nothing should be recorded.
    measure_timings.parse_timing_logs(p, "orthanc-viewer", logs)
    assert p.measurements == []
