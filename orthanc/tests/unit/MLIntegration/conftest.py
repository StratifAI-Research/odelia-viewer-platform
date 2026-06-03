"""MLIntegration test setup.

Prepends `custom/deploy/orthanc/MLIntegration/` to sys.path so that the
`shared` package and the per-service sub-directories can be discovered.
Per-service conftests further prepend their specific dir.

Also adds this directory to sys.path so that _colliders.py is importable as
`from _colliders import ML_SERVICE_COLLIDERS` in any sub-package conftest.
"""
import os
import sys
import types
from types import ModuleType
from unittest.mock import MagicMock

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MLI_DIR = os.path.abspath(os.path.join(_HERE, '..', '..', '..', 'MLIntegration'))

if _MLI_DIR not in sys.path:
    sys.path.insert(0, _MLI_DIR)

# Make _colliders (and other helpers in this dir) importable by sub-package conftests.
if _HERE not in sys.path:
    sys.path.append(_HERE)



# ---------------------------------------------------------------------------
# Stub builder functions — return a ModuleType instance; caller wires it into
# sys.modules (typically via monkeypatch.setitem inside a fixture below).
# ---------------------------------------------------------------------------

def build_torch_stub(cuda_available=False):
    """Return a fake torch module with cuda.is_available()."""
    m = types.ModuleType('torch')
    m.cuda = types.ModuleType('torch.cuda')
    m.cuda.is_available = MagicMock(return_value=cuda_available)
    m.float32 = 'float32'
    m.float16 = 'float16'
    m.long = 'long'
    m.Tensor = MagicMock()
    return m


def build_sitk_stub():
    """Return a fake SimpleITK module covering the surface used by MST tests."""
    import numpy as np

    m = types.ModuleType('SimpleITK')
    m.Image = MagicMock()

    def make_file_reader():
        reader = MagicMock()
        reader.HasMetaDataKey.return_value = False
        reader.GetMetaData.return_value = "0"
        return reader

    def make_series_reader():
        reader = MagicMock()
        reader.GetGDCMSeriesFileNames.return_value = ["file1.dcm", "file2.dcm"]
        mock_img = MagicMock()
        mock_img.GetSize.return_value = (64, 64, 10)
        mock_img.GetSpacing.return_value = (1.0, 1.0, 2.0)
        mock_img.GetDirection.return_value = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        reader.Execute.return_value = mock_img
        return reader

    m.ImageFileReader = MagicMock(side_effect=lambda: make_file_reader())
    m.ImageSeriesReader = MagicMock(side_effect=lambda: make_series_reader())
    mock_result_img = MagicMock()
    m.GetArrayFromImage = MagicMock(return_value=np.ones((10, 64, 64), dtype=np.float32))
    m.GetImageFromArray = MagicMock(return_value=mock_result_img)
    m.WriteImage = MagicMock()
    m.ReadImage = MagicMock(return_value=mock_result_img)
    return m


def build_torchio_stub():
    """Return a fake torchio module.

    Phase 7 extension: adds ZNormalization, CropOrPad, and Compose as real
    subclassable Python classes.  breast-cancer-classification/preprocessing.py
    subclasses tio.ZNormalization and tio.CropOrPad at class-definition time,
    so those attributes must be actual classes (not MagicMock instances).
    """
    m = types.ModuleType('torchio')
    m.Image = MagicMock()
    m.Subject = MagicMock()
    m.ScalarImage = MagicMock(return_value=MagicMock())

    # Real base classes required by BC preprocessing subclassing
    class _ZNormalizationBase:
        """Minimal torchio.ZNormalization stand-in.

        znorm() raises NotImplementedError so tests that need real normalization
        cannot silently pass with identity-substitution data. Tests that need the
        real math must opt out of the torchio_stub fixture and import the real
        torchio (the BC integration tests under test_dicom2nfti_onthefly_integration.py
        and test_preprocessing.py's real-torch block do this).
        """
        def __init__(self, masking_method=None, **kwargs):
            self.masking_method = masking_method

        def znorm(self, image_data, mask):
            raise NotImplementedError(
                "torchio_stub._ZNormalizationBase.znorm() is not implemented; "
                "use the real torchio fixture for tests that exercise normalization"
            )

    class _CropOrPadBase:
        """Minimal torchio.CropOrPad stand-in."""
        def __init__(self, target_shape=None, padding_mode=None, **kwargs):
            self.target_shape = target_shape
            self.padding_mode = padding_mode

    class _ComposeBase:
        """Minimal torchio.Compose stand-in."""
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, x):
            return x

    m.ZNormalization = _ZNormalizationBase
    m.CropOrPad = _CropOrPadBase
    m.Compose = _ComposeBase
    m.Crop = MagicMock(return_value=MagicMock())

    m.transforms = types.ModuleType('torchio.transforms')
    m.transforms.transform = types.ModuleType('torchio.transforms.transform')
    m.transforms.transform.TypeMaskingMethod = MagicMock()
    return m


# ---------------------------------------------------------------------------
# Shared fixtures — available to all MLIntegration tests automatically.
# ---------------------------------------------------------------------------

@pytest.fixture
def torch_stub(monkeypatch) -> ModuleType:
    """Inject a fake torch into sys.modules for the duration of this test."""
    stub = build_torch_stub()
    monkeypatch.setitem(sys.modules, 'torch', stub)
    monkeypatch.setitem(sys.modules, 'torch.cuda', stub.cuda)
    return stub


@pytest.fixture
def sitk_stub(monkeypatch) -> ModuleType:
    """Inject a fake SimpleITK into sys.modules for the duration of this test."""
    stub = build_sitk_stub()
    monkeypatch.setitem(sys.modules, 'SimpleITK', stub)
    return stub


@pytest.fixture
def torchio_stub(monkeypatch) -> ModuleType:
    """Inject a fake torchio into sys.modules for the duration of this test."""
    stub = build_torchio_stub()
    monkeypatch.setitem(sys.modules, 'torchio', stub)
    monkeypatch.setitem(sys.modules, 'torchio.transforms', stub.transforms)
    monkeypatch.setitem(sys.modules, 'torchio.transforms.transform', stub.transforms.transform)
    return stub
