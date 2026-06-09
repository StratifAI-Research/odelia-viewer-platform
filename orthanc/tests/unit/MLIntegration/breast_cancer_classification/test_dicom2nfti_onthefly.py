"""Tests for breast-cancer-classification/dicom2nfti_onthefly.py.

dicom2nfti_onthefly.py imports torchio, torch, pydicom, numpy, and pandas at
module level.  torch and torchio are stubbed via shared fixtures; pydicom and
pandas are available in the venv.

Focus: pure helper functions (maybe_convert, sort_dyn) that have no file I/O.
dicom_to_unilateral_nifti is integration-heavy (DICOM files + TorchIO) and is
excluded from this test file; it is exercised transitively by test_dicom_converter.py.
"""
import sys
from typing import Iterator

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _evict_module(torch_stub, torchio_stub) -> Iterator[None]:
    sys.modules.pop('dicom2nfti_onthefly', None)
    yield
    sys.modules.pop('dicom2nfti_onthefly', None)


def test_module_imports_with_stubs(torch_stub, torchio_stub):
    pass  # should not raise


# --- maybe_convert ---

def test_maybe_convert_returns_none_for_pydicom_sequence():
    import pydicom
    import dicom2nfti_onthefly
    seq = pydicom.sequence.Sequence()
    assert dicom2nfti_onthefly.maybe_convert(seq) is None


def test_maybe_convert_converts_multival_to_list():
    import pydicom
    import dicom2nfti_onthefly
    mv = pydicom.multival.MultiValue(str, ["a", "b", "c"])
    result = dicom2nfti_onthefly.maybe_convert(mv)
    assert result == ["a", "b", "c"]
    assert isinstance(result, list)


def test_maybe_convert_converts_dsfloat_to_float():
    import pydicom
    import dicom2nfti_onthefly
    val = pydicom.valuerep.DSfloat("3.14")
    result = dicom2nfti_onthefly.maybe_convert(val)
    assert isinstance(result, float)
    assert abs(result - 3.14) < 0.001


def test_maybe_convert_converts_IS_to_int():
    import pydicom
    import dicom2nfti_onthefly
    val = pydicom.valuerep.IS("42")
    result = dicom2nfti_onthefly.maybe_convert(val)
    assert isinstance(result, int)
    assert result == 42


def test_maybe_convert_passthrough_plain_string():
    import dicom2nfti_onthefly
    assert dicom2nfti_onthefly.maybe_convert("hello") == "hello"


def test_maybe_convert_passthrough_int():
    import dicom2nfti_onthefly
    assert dicom2nfti_onthefly.maybe_convert(7) == 7


# --- sort_dyn ---

def test_sort_dyn_assigns_trigger_index_and_sequence_names():
    import dicom2nfti_onthefly
    df = pd.DataFrame({
        'TriggerTime': [0.0, 0.0, 100.0, 100.0],
        'SeriesInstanceUID': ['uid'] * 4,
    })
    df.name = 'test_series'
    result = dicom2nfti_onthefly.sort_dyn(df)
    assert '_SequenceName' in result.columns
    assert set(result['_SequenceName'].unique()) == {'Pre', 'Post_1'}


def test_sort_dyn_pre_has_lowest_trigger_time():
    import dicom2nfti_onthefly
    df = pd.DataFrame({
        'TriggerTime': [50.0, 50.0, 200.0, 200.0],
        'SeriesInstanceUID': ['uid'] * 4,
    })
    df.name = 'test_series'
    result = dicom2nfti_onthefly.sort_dyn(df)
    pre_times = result[result['_SequenceName'] == 'Pre']['TriggerTime'].unique()
    assert 50.0 in pre_times


def test_sort_dyn_returns_none_on_unequal_slices():
    """sort_dyn returns None when trigger groups have unequal slice counts."""
    import dicom2nfti_onthefly
    df = pd.DataFrame({
        'TriggerTime': [0.0, 100.0, 100.0],  # Pre: 1 slice, Post_1: 2 slices
        'SeriesInstanceUID': ['uid'] * 3,
    })
    df.name = 'test_series'
    result = dicom2nfti_onthefly.sort_dyn(df)
    assert result is None


def test_sort_dyn_adds_number_of_sequences_column():
    import dicom2nfti_onthefly
    df = pd.DataFrame({
        'TriggerTime': [0.0, 0.0, 100.0, 100.0, 200.0, 200.0],
        'SeriesInstanceUID': ['uid'] * 6,
    })
    df.name = 'test_series'
    result = dicom2nfti_onthefly.sort_dyn(df)
    assert '_NumberOfSequences' in result.columns
    assert result['_NumberOfSequences'].iloc[0] == 3
