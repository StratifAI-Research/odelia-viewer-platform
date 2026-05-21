"""
Unit tests for shared.dicom_storage.

Covers the security-critical series_uid validator (ODV-190 M2) and the
defence-in-depth behaviour of create_series_folder.
"""
from pathlib import Path

import pytest

from shared.dicom_storage import (
    create_series_folder,
    validate_series_uid,
)
from shared.config import StorageConfig


# ---------------------------------------------------------------------------
# validate_series_uid
# ---------------------------------------------------------------------------

class TestValidateSeriesUid:
    """Strict DICOM-UID allowlist: digits and dots only, 1..64 chars."""

    def test_accepts_legitimate_dicom_uid(self):
        uid = "1.2.826.0.1.3680043.2.1125.123456789"
        assert validate_series_uid(uid) == uid

    def test_accepts_short_uid(self):
        assert validate_series_uid("1") == "1"

    def test_accepts_max_length_uid(self):
        uid = "1." + "2" * 62
        assert len(uid) == 64
        assert validate_series_uid(uid) == uid

    def test_rejects_parent_directory_traversal(self):
        with pytest.raises(ValueError, match="invalid series_uid"):
            validate_series_uid("..")

    def test_rejects_relative_traversal(self):
        with pytest.raises(ValueError, match="invalid series_uid"):
            validate_series_uid("../../etc")

    def test_rejects_absolute_path(self):
        with pytest.raises(ValueError, match="invalid series_uid"):
            validate_series_uid("/etc")

    def test_rejects_path_separator(self):
        with pytest.raises(ValueError, match="invalid series_uid"):
            validate_series_uid("1.2.3/4.5.6")

    def test_rejects_backslash(self):
        with pytest.raises(ValueError, match="invalid series_uid"):
            validate_series_uid("1.2.3\\4")

    def test_rejects_letters(self):
        with pytest.raises(ValueError, match="invalid series_uid"):
            validate_series_uid("1.2.abc")

    def test_rejects_null_byte(self):
        with pytest.raises(ValueError, match="invalid series_uid"):
            validate_series_uid("1.2\x00.3")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError, match="invalid series_uid"):
            validate_series_uid("1" * 65)

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="invalid series_uid"):
            validate_series_uid("")

    def test_rejects_non_string(self):
        with pytest.raises(ValueError, match="invalid series_uid"):
            validate_series_uid(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# create_series_folder
# ---------------------------------------------------------------------------

class TestCreateSeriesFolder:
    """create_series_folder must refuse to touch the filesystem on bad input."""

    def test_creates_folder_for_legitimate_uid(self, tmp_path: Path):
        uid = "1.2.826.0.1.3680043.2.1125.42"
        cfg = StorageConfig(image_folder=tmp_path, cleanup_on_start=False)

        folder = create_series_folder(uid, cfg)

        assert folder == tmp_path / uid
        assert folder.exists()
        assert folder.is_dir()

    def test_rejects_traversal_without_touching_filesystem(self, tmp_path: Path):
        cfg = StorageConfig(image_folder=tmp_path, cleanup_on_start=False)
        # Put a sentinel directory next to image_folder; a successful traversal
        # would target it. The validator must fire before any FS op.
        sibling = tmp_path.parent / "would-be-deleted"
        sibling.mkdir()
        (sibling / "keep.txt").write_text("untouched")

        with pytest.raises(ValueError, match="invalid series_uid"):
            create_series_folder("../would-be-deleted", cfg)

        assert sibling.exists()
        assert (sibling / "keep.txt").read_text() == "untouched"
