"""Unit tests for CLI wiring: arg parsing, model resolution, file discovery (ODV-221)."""

from pathlib import Path

import pytest

from batch.cli import discover_files, parse_args, resolve_models
from batch.pipeline import ModelSpec

pytestmark = pytest.mark.unit


def test_resolve_models_none_returns_full_roster_in_order() -> None:
    specs = resolve_models(None)
    assert [s.model_name for s in specs] == [
        "agaldran",
        "BCN_AIM",
        "DivideAndConquer",
        "LME_ABMIL",
        "MST",
        "Pimed",
    ]
    assert all(isinstance(s, ModelSpec) for s in specs)


def test_resolve_models_selects_named_subset_in_given_order() -> None:
    specs = resolve_models(["MST", "agaldran"])
    assert [s.model_name for s in specs] == ["MST", "agaldran"]
    mst = specs[0]
    assert mst.ai_name == "ODELIA MST init weights preview"
    assert mst.router_host == "orthanc-router-odelia-mst"
    assert mst.router_port == 8049
    assert mst.target_url == "http://orthanc-router-odelia-mst:8042/dicom-web"


def test_resolve_models_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        resolve_models(["nope"])


def test_parse_args_splits_models_and_keeps_input_path() -> None:
    ns = parse_args(["--input", "/data/studies", "--models", "MST,agaldran"])
    assert ns.input == Path("/data/studies")
    assert ns.models == ["MST", "agaldran"]


def test_parse_args_models_defaults_to_none_meaning_all() -> None:
    ns = parse_args(["--input", "/data/studies"])
    assert ns.models is None


def test_parse_args_models_all_normalizes_to_none() -> None:
    ns = parse_args(["--input", "/data/studies", "--models", "all"])
    assert ns.models is None


def test_parse_args_mapping_and_data_raw() -> None:
    ns = parse_args(
        ["--input", "/data/studies", "--mapping", "/m.csv", "--data-raw", "/raw"]
    )
    assert ns.mapping == Path("/m.csv")
    assert ns.data_raw == Path("/raw")


def test_parse_args_data_raw_defaults_to_input() -> None:
    ns = parse_args(["--input", "/data/studies", "--mapping", "/m.csv"])
    assert ns.data_raw == Path("/data/studies")


def test_discover_files_walks_recursively_sorted_by_path(tmp_path) -> None:
    (tmp_path / "a.dcm").write_bytes(b"a")
    (tmp_path / "c.dcm").write_bytes(b"c")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.dcm").write_bytes(b"b")

    files = discover_files(tmp_path)

    assert [f.name for f in files] == ["a.dcm", "c.dcm", "b.dcm"]
