"""Tests for breast-cancer-classification/retrieval_strategy.py — RetrievalStrategy + WadoRSRetrieval.

BC's WadoRSRetrieval differs from MST: it passes all series to retrieve_via_wado_rs
(not just the first) and extracts series_uid from the first returned dataset
rather than from the input list.
"""
import inspect
from unittest.mock import patch

import pytest


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


def test_wado_rs_retrieval_retrieve_calls_wado_and_returns_folder(tmp_path):
    """retrieve() calls retrieve_via_wado_rs and save_datasets_to_folder."""
    from retrieval_strategy import WadoRSRetrieval
    from shared.config import StorageConfig
    import pydicom

    cfg = StorageConfig(image_folder=tmp_path)
    info = [{'retrieval_url': 'http://host/dicom-web', 'study_uid': 'S1', 'series_uid': 'SE1'}]

    fake_ds = pydicom.dataset.Dataset()
    fake_ds.SeriesInstanceUID = '1.2.3.SE1'

    with patch('retrieval_strategy.retrieve_via_wado_rs', return_value=[fake_ds]) as mock_retrieve, \
         patch('retrieval_strategy.save_datasets_to_folder', return_value=tmp_path / 'SE1') as mock_save:
        strategy = WadoRSRetrieval(info, cfg)
        folder, series_uid = strategy.retrieve()

    mock_retrieve.assert_called_once_with(info)
    mock_save.assert_called_once()
    assert series_uid == '1.2.3.SE1'


def test_wado_rs_retrieval_raises_on_empty_datasets(tmp_path):
    from retrieval_strategy import WadoRSRetrieval
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    info = [{'retrieval_url': 'http://host/dicom-web', 'study_uid': 'S1', 'series_uid': 'SE1'}]

    with patch('retrieval_strategy.retrieve_via_wado_rs', return_value=[]):
        strategy = WadoRSRetrieval(info, cfg)
        with pytest.raises(ValueError, match='No DICOM instances'):
            strategy.retrieve()


def test_wado_rs_retrieval_passes_all_series(tmp_path):
    """BC retrieval passes all series to retrieve_via_wado_rs (unlike MST which uses only the first)."""
    from retrieval_strategy import WadoRSRetrieval
    from shared.config import StorageConfig
    import pydicom

    cfg = StorageConfig(image_folder=tmp_path)
    info = [
        {'retrieval_url': 'http://host/dicom-web', 'study_uid': 'S1', 'series_uid': 'SE1'},
        {'retrieval_url': 'http://host/dicom-web', 'study_uid': 'S1', 'series_uid': 'SE2'},
    ]

    fake_ds = pydicom.dataset.Dataset()
    fake_ds.SeriesInstanceUID = '1.2.3.SE1'

    with patch('retrieval_strategy.retrieve_via_wado_rs', return_value=[fake_ds]) as mock_retrieve, \
         patch('retrieval_strategy.save_datasets_to_folder', return_value=tmp_path / 'SE1'):
        strategy = WadoRSRetrieval(info, cfg)
        strategy.retrieve()

    # All series should be passed (not just the first)
    called_with = mock_retrieve.call_args[0][0]
    assert len(called_with) == 2


def test_wado_rs_retrieval_series_uid_from_dataset(tmp_path):
    """series_uid is extracted from the first returned dataset's SeriesInstanceUID."""
    from retrieval_strategy import WadoRSRetrieval
    from shared.config import StorageConfig
    import pydicom

    cfg = StorageConfig(image_folder=tmp_path)
    info = [{'retrieval_url': 'http://host/dicom-web', 'study_uid': 'S1', 'series_uid': 'SE1'}]

    fake_ds = pydicom.dataset.Dataset()
    fake_ds.SeriesInstanceUID = '9.8.7.6.5'

    with patch('retrieval_strategy.retrieve_via_wado_rs', return_value=[fake_ds]), \
         patch('retrieval_strategy.save_datasets_to_folder', return_value=tmp_path / 'SE1'):
        strategy = WadoRSRetrieval(info, cfg)
        folder, series_uid = strategy.retrieve()

    assert series_uid == '9.8.7.6.5'
