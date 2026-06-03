"""Tests for MST-classification/preprocessing.py.

preprocessing.py imports torch at module level.
torch_stub and torchio_stub fixtures are provided by
tests/unit/MLIntegration/conftest.py.
"""
import sys
import types
from unittest.mock import MagicMock
import numpy as np


def test_prepare_for_inference_returns_scalar_image(tmp_path, torch_stub, torchio_stub, monkeypatch):
    """prepare_for_inference should call tio.ScalarImage and return its result."""
    nifti_path = tmp_path / 'img.nii.gz'
    nifti_path.touch()
    model_path = tmp_path / 'model'
    model_path.mkdir()

    sys.modules.pop('preprocessing', None)
    import preprocessing
    result = preprocessing.prepare_for_inference(nifti_path, model_path)

    torchio_stub.ScalarImage.assert_called_once_with(str(nifti_path))
    assert result is torchio_stub.ScalarImage.return_value


def test_prepare_for_inference_adds_model_path_to_sys_path(tmp_path, torch_stub, torchio_stub, monkeypatch):
    """model_path must be added to sys.path so the reference impl can be found."""
    nifti_path = tmp_path / 'img.nii.gz'
    nifti_path.touch()
    model_path = tmp_path / 'model'
    model_path.mkdir()

    sys.modules.pop('preprocessing', None)
    import preprocessing

    # Clear the model path first
    if str(model_path) in sys.path:
        sys.path.remove(str(model_path))

    preprocessing.prepare_for_inference(nifti_path, model_path)
    assert str(model_path) in sys.path


def test_generate_attention_overlays_returns_dict(tmp_path, torch_stub, torchio_stub, monkeypatch):
    """generate_attention_overlays should return a dict with data/shape/dtype keys."""
    model_path = tmp_path / 'model'
    model_path.mkdir()

    # Stub predict_attention module (needed inside generate_attention_overlays)
    predict_stub = types.ModuleType('predict_attention')
    predict_stub.minmax_norm = MagicMock(side_effect=lambda x: x)
    mock_overlay = MagicMock()
    mock_overlay.shape = (5, 3, 16, 16)
    arr = np.zeros((5, 3, 16, 16), dtype=np.float32)
    mock_overlay.cpu.return_value.numpy.return_value = arr
    predict_stub.tensor_cam2image = MagicMock(return_value=mock_overlay)
    monkeypatch.setitem(sys.modules, 'predict_attention', predict_stub)

    sys.modules.pop('preprocessing', None)
    import preprocessing

    img_tensor = MagicMock()
    img_tensor.swapaxes.return_value = img_tensor
    img_tensor.unsqueeze.return_value = img_tensor
    weight_tensor = MagicMock()
    weight_tensor.swapaxes.return_value = weight_tensor
    weight_tensor.unsqueeze.return_value = weight_tensor

    result = preprocessing.generate_attention_overlays(img_tensor, weight_tensor, model_path)

    assert isinstance(result, dict)
    assert 'data' in result
    assert 'shape' in result
    assert 'dtype' in result


def test_generate_attention_overlays_dtype_is_uint8(tmp_path, torch_stub, torchio_stub, monkeypatch):
    """dtype field must be 'uint8'."""
    model_path = tmp_path / 'model'
    model_path.mkdir()

    predict_stub = types.ModuleType('predict_attention')
    predict_stub.minmax_norm = MagicMock(side_effect=lambda x: x)
    mock_overlay = MagicMock()
    mock_overlay.shape = (5, 3, 16, 16)
    arr = np.zeros((5, 3, 16, 16), dtype=np.float32)
    mock_overlay.cpu.return_value.numpy.return_value = arr
    predict_stub.tensor_cam2image = MagicMock(return_value=mock_overlay)
    monkeypatch.setitem(sys.modules, 'predict_attention', predict_stub)

    sys.modules.pop('preprocessing', None)
    import preprocessing

    img_tensor = MagicMock()
    img_tensor.swapaxes.return_value = img_tensor
    img_tensor.unsqueeze.return_value = img_tensor
    weight_tensor = MagicMock()
    weight_tensor.swapaxes.return_value = weight_tensor
    weight_tensor.unsqueeze.return_value = weight_tensor

    result = preprocessing.generate_attention_overlays(img_tensor, weight_tensor, model_path)
    import base64
    decoded = base64.b64decode(result['data'])
    assert result['dtype'] == 'uint8'
    assert result['shape'] == [5, 16, 16, 3]
    assert len(decoded) == int(np.prod(result['shape']))
