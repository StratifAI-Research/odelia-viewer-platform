"""Tests for shared/config.py — StorageConfig dataclass."""
from pathlib import Path



def test_storage_config_stores_image_folder():
    from shared.config import StorageConfig
    cfg = StorageConfig(image_folder=Path('/tmp/dicom'))
    assert cfg.image_folder == Path('/tmp/dicom')


def test_storage_config_cleanup_on_start_default_true():
    from shared.config import StorageConfig
    cfg = StorageConfig(image_folder=Path('/tmp/dicom'))
    assert cfg.cleanup_on_start is True


def test_storage_config_cleanup_on_start_can_be_false():
    from shared.config import StorageConfig
    cfg = StorageConfig(image_folder=Path('/tmp/dicom'), cleanup_on_start=False)
    assert cfg.cleanup_on_start is False


def test_storage_config_image_folder_accepts_string_path():
    from shared.config import StorageConfig
    cfg = StorageConfig(image_folder=Path('/var/data'))
    assert str(cfg.image_folder) == '/var/data'


def test_storage_config_is_dataclass():
    import dataclasses
    from shared.config import StorageConfig
    assert dataclasses.is_dataclass(StorageConfig)


def test_storage_config_equality():
    from shared.config import StorageConfig
    a = StorageConfig(image_folder=Path('/tmp'))
    b = StorageConfig(image_folder=Path('/tmp'))
    assert a == b


def test_storage_config_inequality_different_folder():
    from shared.config import StorageConfig
    a = StorageConfig(image_folder=Path('/tmp/a'))
    b = StorageConfig(image_folder=Path('/tmp/b'))
    assert a != b
