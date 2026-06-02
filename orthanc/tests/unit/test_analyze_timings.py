"""Unit tests for analyze_timings.py — timing CSV analyzer.

Skips the CLI entrypoint (main) and print_summary (pure output side-effect).
Tests the library helpers with tight assertions based on the actual return shapes.
"""
import csv
import json





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


_SAMPLE_ROWS_STR = [
    {'trace_id': 'abc123', 'timestamp': '2024-01-01T00:00:00',
     'component': 'router', 'operation': 'fetch', 'duration_ms': '120.0', 'metadata': '{}'},
    {'trace_id': 'abc123', 'timestamp': '2024-01-01T00:00:01',
     'component': 'router', 'operation': 'send', 'duration_ms': '80.0', 'metadata': '{}'},
    {'trace_id': 'abc123', 'timestamp': '2024-01-01T00:00:02',
     'component': 'model', 'operation': 'inference', 'duration_ms': '450.0', 'metadata': '{}'},
]


# ---------------------------------------------------------------------------
# load_timing_csv
# ---------------------------------------------------------------------------

def test_load_timing_csv_returns_list_of_dicts(tmp_path):
    from analyze_timings import load_timing_csv
    csv_path = tmp_path / 'timings.csv'
    _write_csv(csv_path, _SAMPLE_ROWS_STR)
    rows = load_timing_csv(str(csv_path))
    assert len(rows) == 3
    assert rows[0]['component'] == 'router'


def test_load_timing_csv_converts_duration_to_float(tmp_path):
    from analyze_timings import load_timing_csv
    csv_path = tmp_path / 'timings.csv'
    _write_csv(csv_path, _SAMPLE_ROWS_STR)
    rows = load_timing_csv(str(csv_path))
    # load_timing_csv converts duration_ms from string to float
    assert isinstance(rows[0]['duration_ms'], float)
    assert rows[0]['duration_ms'] == 120.0


def test_load_timing_csv_empty_file_yields_empty_list(tmp_path):
    from analyze_timings import load_timing_csv
    csv_path = tmp_path / 'empty.csv'
    csv_path.write_text('trace_id,timestamp,component,operation,duration_ms,metadata\n')
    rows = load_timing_csv(str(csv_path))
    assert rows == []


def test_load_timing_csv_preserves_other_fields(tmp_path):
    from analyze_timings import load_timing_csv
    csv_path = tmp_path / 'timings.csv'
    _write_csv(csv_path, _SAMPLE_ROWS_STR)
    rows = load_timing_csv(str(csv_path))
    assert rows[0]['trace_id'] == 'abc123'
    assert rows[0]['operation'] == 'fetch'


# ---------------------------------------------------------------------------
# analyze_by_component
# Return shape: {component: {'operations': [{'operation': str, 'duration_ms': float}], 'total': float}}
# ---------------------------------------------------------------------------

def _make_float_rows():
    """Return rows with float duration_ms (as produced by load_timing_csv)."""
    return [
        {'component': 'router', 'operation': 'fetch', 'duration_ms': 100.0},
        {'component': 'router', 'operation': 'send',  'duration_ms': 200.0},
        {'component': 'model',  'operation': 'infer', 'duration_ms': 50.0},
    ]


def test_analyze_by_component_returns_all_components():
    from analyze_timings import analyze_by_component
    summary = analyze_by_component(_make_float_rows())
    assert set(summary.keys()) == {'router', 'model'}


def test_analyze_by_component_correct_total():
    from analyze_timings import analyze_by_component
    summary = analyze_by_component(_make_float_rows())
    assert summary['router']['total'] == 300.0
    assert summary['model']['total'] == 50.0


def test_analyze_by_component_operations_list_length():
    from analyze_timings import analyze_by_component
    summary = analyze_by_component(_make_float_rows())
    assert len(summary['router']['operations']) == 2
    assert len(summary['model']['operations']) == 1


def test_analyze_by_component_operations_shape():
    from analyze_timings import analyze_by_component
    summary = analyze_by_component(_make_float_rows())
    op = summary['router']['operations'][0]
    assert 'operation' in op
    assert 'duration_ms' in op


def test_analyze_by_component_operation_names():
    from analyze_timings import analyze_by_component
    summary = analyze_by_component(_make_float_rows())
    op_names = {o['operation'] for o in summary['router']['operations']}
    assert op_names == {'fetch', 'send'}


def test_analyze_by_component_single_row():
    from analyze_timings import analyze_by_component
    rows = [{'component': 'viewer', 'operation': 'load', 'duration_ms': 75.5}]
    summary = analyze_by_component(rows)
    assert summary['viewer']['total'] == 75.5
    assert len(summary['viewer']['operations']) == 1


def test_analyze_by_component_empty_input():
    from analyze_timings import analyze_by_component
    summary = analyze_by_component([])
    assert summary == {}


# ---------------------------------------------------------------------------
# export_to_json
# export_to_json(measurements, output_path) — measurements must have 'trace_id'
# ---------------------------------------------------------------------------

def _make_export_rows():
    return [
        {'trace_id': 'tid42', 'component': 'router', 'operation': 'fetch',
         'duration_ms': 100.0, 'timestamp': '2024-01-01T00:00:00', 'metadata': '{}'},
        {'trace_id': 'tid42', 'component': 'model', 'operation': 'infer',
         'duration_ms': 200.0, 'timestamp': '2024-01-01T00:00:01', 'metadata': '{}'},
    ]


def test_export_to_json_writes_valid_json(tmp_path):
    from analyze_timings import export_to_json
    out_path = tmp_path / 'out.json'
    export_to_json(_make_export_rows(), str(out_path))
    j = json.loads(out_path.read_text())
    assert j  # non-empty


def test_export_to_json_trace_id(tmp_path):
    from analyze_timings import export_to_json
    out_path = tmp_path / 'out.json'
    export_to_json(_make_export_rows(), str(out_path))
    j = json.loads(out_path.read_text())
    assert j['trace_id'] == 'tid42'


def test_export_to_json_total_measurements(tmp_path):
    from analyze_timings import export_to_json
    out_path = tmp_path / 'out.json'
    export_to_json(_make_export_rows(), str(out_path))
    j = json.loads(out_path.read_text())
    assert j['total_measurements'] == 2


def test_export_to_json_has_by_component(tmp_path):
    from analyze_timings import export_to_json
    out_path = tmp_path / 'out.json'
    export_to_json(_make_export_rows(), str(out_path))
    j = json.loads(out_path.read_text())
    assert 'by_component' in j
    assert 'router' in j['by_component']
    assert 'model' in j['by_component']


def test_export_to_json_has_all_measurements(tmp_path):
    from analyze_timings import export_to_json
    out_path = tmp_path / 'out.json'
    export_to_json(_make_export_rows(), str(out_path))
    j = json.loads(out_path.read_text())
    assert len(j['all_measurements']) == 2


# ---------------------------------------------------------------------------
# compare_profiles — print-only; just ensure it doesn't raise
# ---------------------------------------------------------------------------

def test_compare_profiles_does_not_raise(tmp_path):
    from analyze_timings import compare_profiles
    rows_a = [
        {'trace_id': 'r1', 'component': 'router', 'operation': 'fetch',
         'duration_ms': '100.0', 'timestamp': '2024-01-01T00:00:00', 'metadata': '{}'},
        {'trace_id': 'r1', 'component': 'router', 'operation': 'complete_pipeline',
         'duration_ms': '500.0', 'timestamp': '2024-01-01T00:00:01', 'metadata': '{}'},
    ]
    rows_b = [
        {'trace_id': 'r2', 'component': 'router', 'operation': 'fetch',
         'duration_ms': '150.0', 'timestamp': '2024-01-01T00:00:00', 'metadata': '{}'},
        {'trace_id': 'r2', 'component': 'router', 'operation': 'complete_pipeline',
         'duration_ms': '600.0', 'timestamp': '2024-01-01T00:00:01', 'metadata': '{}'},
    ]
    csv_a = tmp_path / 'a.csv'
    csv_b = tmp_path / 'b.csv'
    _write_csv(csv_a, rows_a)
    _write_csv(csv_b, rows_b)
    # compare_profiles prints only; return value is None — just ensure no exception
    compare_profiles([str(csv_a), str(csv_b)])


# ---------------------------------------------------------------------------
# T2: compare_profiles aggregation - capture stdout and verify per-component
#     avg/min/max are computed from the right data.
# ---------------------------------------------------------------------------

def _write_compare_csv(path, rows):
    """Write a minimal timing CSV the loader recognises."""
    headers = ["trace_id", "timestamp", "component", "operation", "duration_ms", "metadata"]
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r.get(h, "")) for h in headers))
    path.write_text("\n".join(lines) + "\n")


def test_compare_profiles_reports_per_component_avg_min_max(tmp_path, capsys):
    """compare_profiles prints avg/min/max for each component across runs."""
    import analyze_timings
    csv1 = tmp_path / "run1.csv"
    csv2 = tmp_path / "run2.csv"
    _write_compare_csv(csv1, [
        {"trace_id": "t1", "component": "orthanc-router", "operation": "infer",
         "duration_ms": "100.0"},
        {"trace_id": "t1", "component": "ml-mst", "operation": "forward",
         "duration_ms": "500.0"},
    ])
    _write_compare_csv(csv2, [
        {"trace_id": "t2", "component": "orthanc-router", "operation": "infer",
         "duration_ms": "200.0"},
        {"trace_id": "t2", "component": "ml-mst", "operation": "forward",
         "duration_ms": "700.0"},
    ])

    analyze_timings.compare_profiles([str(csv1), str(csv2)])
    out = capsys.readouterr().out

    # orthanc-router: avg=150, min=100, max=200; ml-mst: avg=600, min=500, max=700.
    assert "orthanc-router" in out
    assert "ml-mst" in out
    # Values rendered with %.2f formatter.
    assert "150.00ms" in out
    assert "100.00ms" in out
    assert "200.00ms" in out
    assert "600.00ms" in out
    assert "500.00ms" in out
    assert "700.00ms" in out


def test_compare_profiles_reports_end_to_end_times(tmp_path, capsys):
    """A complete_pipeline operation is surfaced as the end-to-end time per run."""
    import analyze_timings
    csv1 = tmp_path / "run1.csv"
    _write_compare_csv(csv1, [
        {"trace_id": "t1", "component": "orthanc-router",
         "operation": "complete_pipeline", "duration_ms": "1234.50"},
    ])
    analyze_timings.compare_profiles([str(csv1)])
    out = capsys.readouterr().out
    assert "1234.50ms" in out
