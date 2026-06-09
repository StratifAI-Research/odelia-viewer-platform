"""Tests for medgemma-mri/retrieval_strategy.py — RetrievalStrategy + WadoRSRetrieval.

medgemma's WadoRSRetrieval only processes the FIRST series (logs a warning
for multiple) and extracts series_uid from the input list (not from the
returned dataset's SeriesInstanceUID).  This differs from BC's implementation.
"""
from typing import Iterator
import sys
import inspect
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture(autouse=True)
def _evict_module() -> Iterator[None]:
    sys.modules.pop('retrieval_strategy', None)
    yield
    sys.modules.pop('retrieval_strategy', None)


def test_retrieval_strategy_is_abstract():
    from retrieval_strategy import RetrievalStrategy
    assert inspect.isabstract(RetrievalStrategy)


def test_retrieval_strategy_has_retrieve_method():
    from retrieval_strategy import RetrievalStrategy
    assert hasattr(RetrievalStrategy, 'retrieve')


def test_wado_rs_retrieval_is_subclass_of_retrieval_strategy():
    from retrieval_strategy import RetrievalStrategy, WadoRSRetrieval
    assert issubclass(WadoRSRetrieval, RetrievalStrategy)


def test_wado_rs_retrieval_stores_retrieval_info(tmp_path):
    from retrieval_strategy import WadoRSRetrieval
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    info = [{'retrieval_url': 'http://host/dicom-web', 'study_uid': 'S1', 'series_uid': 'SE1'}]
    strategy = WadoRSRetrieval(info, cfg)

    assert strategy.wado_rs_retrieval == info
    assert strategy.storage_config == cfg


def test_wado_rs_retrieval_retrieve_uses_only_first_series(tmp_path):
    """medgemma WadoRSRetrieval only passes the first series to retrieve_via_wado_rs."""
    from retrieval_strategy import WadoRSRetrieval
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    info = [
        {'retrieval_url': 'http://host/dicom-web', 'study_uid': 'S1', 'series_uid': 'SE1'},
        {'retrieval_url': 'http://host/dicom-web', 'study_uid': 'S1', 'series_uid': 'SE2'},
    ]

    fake_ds = MagicMock()

    with patch('retrieval_strategy.retrieve_via_wado_rs', return_value=[fake_ds]) as mock_retrieve, \
         patch('retrieval_strategy.save_datasets_to_folder', return_value=tmp_path / 'SE1'):
        strategy = WadoRSRetrieval(info, cfg)
        strategy.retrieve()

    # Only the first series should be passed
    called_with = mock_retrieve.call_args[0][0]
    assert len(called_with) == 1
    assert called_with[0]['series_uid'] == 'SE1'


def test_wado_rs_retrieval_retrieve_returns_folder_and_uid(tmp_path):
    from retrieval_strategy import WadoRSRetrieval
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    info = [{'retrieval_url': 'http://host/dicom-web', 'study_uid': 'S1', 'series_uid': 'SE1'}]
    expected_folder = tmp_path / 'SE1'

    with patch('retrieval_strategy.retrieve_via_wado_rs', return_value=[MagicMock()]), \
         patch('retrieval_strategy.save_datasets_to_folder', return_value=expected_folder):
        strategy = WadoRSRetrieval(info, cfg)
        folder, series_uid = strategy.retrieve()

    assert folder == expected_folder
    assert series_uid == 'SE1'


def test_wado_rs_retrieval_series_uid_from_input_list(tmp_path):
    """series_uid comes from the input list's series_uid key (not from dataset)."""
    from retrieval_strategy import WadoRSRetrieval
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    info = [{'retrieval_url': 'http://host/dicom-web', 'study_uid': 'S1', 'series_uid': 'MY_SERIES_UID'}]

    with patch('retrieval_strategy.retrieve_via_wado_rs', return_value=[MagicMock()]), \
         patch('retrieval_strategy.save_datasets_to_folder', return_value=tmp_path):
        strategy = WadoRSRetrieval(info, cfg)
        _, series_uid = strategy.retrieve()

    assert series_uid == 'MY_SERIES_UID'


def test_wado_rs_retrieval_raises_on_empty_datasets(tmp_path):
    from retrieval_strategy import WadoRSRetrieval
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    info = [{'retrieval_url': 'http://host/dicom-web', 'study_uid': 'S1', 'series_uid': 'SE1'}]

    with patch('retrieval_strategy.retrieve_via_wado_rs', return_value=[]):
        strategy = WadoRSRetrieval(info, cfg)
        with pytest.raises(ValueError, match='No DICOM instances'):
            strategy.retrieve()


def test_wado_rs_retrieval_calls_save_datasets_to_folder(tmp_path):
    from retrieval_strategy import WadoRSRetrieval
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    info = [{'retrieval_url': 'http://host/dicom-web', 'study_uid': 'S1', 'series_uid': 'SE1'}]

    fake_ds = MagicMock()

    with patch('retrieval_strategy.retrieve_via_wado_rs', return_value=[fake_ds]), \
         patch('retrieval_strategy.save_datasets_to_folder', return_value=tmp_path / 'SE1') as mock_save:
        strategy = WadoRSRetrieval(info, cfg)
        strategy.retrieve()

    mock_save.assert_called_once()
