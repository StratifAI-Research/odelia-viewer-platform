"""Tests for shared/dicom_storage.py — DICOM file storage utilities."""
import io

import pydicom
import pytest


def _make_fake_dataset(idx=0):
    """Return a minimal pydicom Dataset compatible with pydicom 3.x."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, UID

    sop_uid = f'1.2.3.{idx}'
    ds = Dataset()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.4')
    meta.MediaStorageSOPInstanceUID = UID(sop_uid)
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = meta
    ds.SOPClassUID = UID('1.2.840.10008.5.1.4.1.1.4')
    ds.SOPInstanceUID = UID(sop_uid)
    ds.PatientName = f'Patient{idx}'
    return ds


def test_save_datasets_to_folder_creates_files(tmp_path):
    from shared.dicom_storage import save_datasets_to_folder
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    datasets = [_make_fake_dataset(i) for i in range(3)]

    result_folder = save_datasets_to_folder(datasets, 'series-001', cfg)

    assert result_folder.is_dir()
    dcm_files = list(result_folder.glob('*.dcm'))
    assert len(dcm_files) == 3


def test_save_datasets_to_folder_returns_path_inside_image_folder(tmp_path):
    from shared.dicom_storage import save_datasets_to_folder
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    datasets = [_make_fake_dataset(0)]

    result = save_datasets_to_folder(datasets, 'series-xyz', cfg)

    assert str(tmp_path) in str(result)
    assert 'series-xyz' in str(result)


def test_save_datasets_to_folder_roundtrip(tmp_path):
    """Files written must be readable as valid pydicom Datasets."""
    from shared.dicom_storage import save_datasets_to_folder
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    original = _make_fake_dataset(99)
    save_datasets_to_folder([original], 'roundtrip-series', cfg)

    saved_files = list((tmp_path / 'roundtrip-series').glob('*.dcm'))
    assert len(saved_files) == 1

    loaded = pydicom.dcmread(str(saved_files[0]), force=True)
    assert str(loaded.SOPInstanceUID) == '1.2.3.99'


def test_save_datasets_raises_on_empty_list(tmp_path):
    from shared.dicom_storage import save_datasets_to_folder
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    with pytest.raises(ValueError, match='No datasets'):
        save_datasets_to_folder([], 'empty-series', cfg)


def test_save_datasets_naming_convention(tmp_path):
    """Each saved file should follow instance_NNNN.dcm naming."""
    from shared.dicom_storage import save_datasets_to_folder
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    datasets = [_make_fake_dataset(i) for i in range(2)]
    folder = save_datasets_to_folder(datasets, 'naming-test', cfg)

    names = {f.name for f in folder.glob('*.dcm')}
    assert 'instance_0000.dcm' in names
    assert 'instance_0001.dcm' in names


def test_save_datasets_cleans_existing_folder(tmp_path):
    """Re-saving to same series_uid replaces existing files (clean=True default)."""
    from shared.dicom_storage import save_datasets_to_folder
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    first = [_make_fake_dataset(i) for i in range(5)]
    save_datasets_to_folder(first, 'clean-series', cfg)

    second = [_make_fake_dataset(0)]
    folder = save_datasets_to_folder(second, 'clean-series', cfg)

    assert len(list(folder.glob('*.dcm'))) == 1


def test_save_dicom_bytes_to_folder_creates_files(tmp_path):
    from shared.dicom_storage import save_dicom_bytes_to_folder
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    ds = _make_fake_dataset(0)
    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, write_like_original=False)
    raw = buf.getvalue()

    folder = save_dicom_bytes_to_folder([raw, raw], 'bytes-series', cfg)
    assert len(list(folder.glob('*.dcm'))) == 2


def test_save_dicom_bytes_raises_on_empty(tmp_path):
    from shared.dicom_storage import save_dicom_bytes_to_folder
    from shared.config import StorageConfig

    cfg = StorageConfig(image_folder=tmp_path)
    with pytest.raises(ValueError, match='No DICOM files'):
        save_dicom_bytes_to_folder([], 'empty', cfg)


# ---------------------------------------------------------------------------
# S4: syscall failures propagate from create_series_folder + save_*_to_folder
# ---------------------------------------------------------------------------

def test_create_series_folder_propagates_rmtree_permission_error(tmp_path, monkeypatch):
    """When shutil.rmtree raises PermissionError, create_series_folder must surface it."""
    from shared.dicom_storage import create_series_folder
    from shared.config import StorageConfig
    cfg = StorageConfig(image_folder=str(tmp_path))
    # Pre-create the target dir so the clean branch is taken.
    (tmp_path / "series.uid").mkdir()
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "rmtree",
                         lambda *a, **kw: (_ for _ in ()).throw(PermissionError("denied")))
    with pytest.raises(PermissionError, match="denied"):
        create_series_folder("series.uid", cfg, clean=True)


def test_create_series_folder_propagates_makedirs_error(tmp_path, monkeypatch):
    """When os.makedirs raises OSError (e.g. read-only mount), it must surface."""
    from shared.dicom_storage import create_series_folder
    from shared.config import StorageConfig
    cfg = StorageConfig(image_folder=str(tmp_path))
    import os as _os
    monkeypatch.setattr(_os, "makedirs",
                         lambda *a, **kw: (_ for _ in ()).throw(OSError("read-only filesystem")))
    with pytest.raises(OSError, match="read-only"):
        create_series_folder("series.uid", cfg, clean=False)


def test_save_dicom_bytes_to_folder_propagates_open_error(tmp_path, monkeypatch):
    """When open(..., 'wb') raises (e.g. disk full), the syscall failure surfaces."""
    from shared.dicom_storage import save_dicom_bytes_to_folder
    from shared.config import StorageConfig
    cfg = StorageConfig(image_folder=str(tmp_path))

    import builtins
    real_open = builtins.open
    def _raising_open(path, mode="r", *a, **kw):
        if "w" in mode and "b" in mode and str(path).endswith(".dcm"):
            raise OSError("No space left on device")
        return real_open(path, mode, *a, **kw)
    monkeypatch.setattr(builtins, "open", _raising_open)

    with pytest.raises(OSError, match="No space left"):
        save_dicom_bytes_to_folder([b"DCMP"], "series.uid", cfg)
