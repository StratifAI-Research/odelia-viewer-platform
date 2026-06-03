"""Tests for MST-classification/response_builder.py — build_bilateral_response."""
import numpy as np


def _make_probs(left_vals, right_vals):
    return {
        'left': np.array(left_vals, dtype=np.float32),
        'right': np.array(right_vals, dtype=np.float32),
    }


def _make_attention_maps(num_slices=5, h=16, w=16):
    data = np.zeros((num_slices, h, w, 3), dtype=np.uint8)
    return {
        'data': 'base64encoded==',
        'shape': [num_slices, h, w, 3],
        'dtype': 'uint8',
    }


def _make_model_info():
    return {'model_name': 'MST', 'architecture': 'ViT', 'version': '1.0'}


def test_build_bilateral_response_returns_dict():
    from response_builder import build_bilateral_response
    probs = _make_probs([0.8, 0.1, 0.1], [0.1, 0.1, 0.8])
    result = build_bilateral_response(probs, _make_attention_maps(), _make_model_info())
    assert isinstance(result, dict)


def test_build_bilateral_response_has_left_right_keys():
    from response_builder import build_bilateral_response
    probs = _make_probs([0.8, 0.1, 0.1], [0.1, 0.1, 0.8])
    result = build_bilateral_response(probs, _make_attention_maps(), _make_model_info())
    assert 'left' in result
    assert 'right' in result


def test_build_bilateral_response_prediction_is_string():
    from response_builder import build_bilateral_response
    probs = _make_probs([0.8, 0.1, 0.1], [0.1, 0.1, 0.8])
    result = build_bilateral_response(probs, _make_attention_maps(), _make_model_info())
    assert isinstance(result['left']['prediction'], str)
    assert isinstance(result['right']['prediction'], str)


def test_build_bilateral_response_correct_class_names():
    """Verify the class name mapping matches the source (No lesion/Benign/Malignant)."""
    from response_builder import build_bilateral_response
    # Index 0 -> No lesion, Index 1 -> Benign, Index 2 -> Malignant
    probs = _make_probs([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    result = build_bilateral_response(probs, _make_attention_maps(), _make_model_info())
    assert result['left']['prediction'] == 'No lesion'
    assert result['right']['prediction'] == 'Malignant'


def test_build_bilateral_response_confidence_is_percentage():
    """Confidence should be in [0, 100] range."""
    from response_builder import build_bilateral_response
    probs = _make_probs([0.75, 0.15, 0.10], [0.10, 0.80, 0.10])
    result = build_bilateral_response(probs, _make_attention_maps(), _make_model_info())
    assert 0.0 <= result['left']['confidence'] <= 100.0
    assert 0.0 <= result['right']['confidence'] <= 100.0


def test_build_bilateral_response_confidence_matches_argmax():
    """Confidence value should equal max probability * 100."""
    from response_builder import build_bilateral_response
    probs = _make_probs([0.6, 0.3, 0.1], [0.2, 0.7, 0.1])
    result = build_bilateral_response(probs, _make_attention_maps(), _make_model_info())
    assert abs(result['left']['confidence'] - 60.0) < 0.1
    assert abs(result['right']['confidence'] - 70.0) < 0.1


def test_build_bilateral_response_model_metadata_present():
    from response_builder import build_bilateral_response
    info = {'model_name': 'MyModel', 'architecture': 'ViT', 'version': '2.0'}
    probs = _make_probs([0.5, 0.3, 0.2], [0.1, 0.8, 0.1])
    result = build_bilateral_response(probs, _make_attention_maps(), info)
    assert 'model_metadata' in result
    assert result['model_metadata']['model_name'] == 'MyModel'


def test_build_bilateral_response_attention_maps_passthrough():
    from response_builder import build_bilateral_response
    attn = _make_attention_maps(num_slices=10)
    probs = _make_probs([0.5, 0.3, 0.2], [0.1, 0.8, 0.1])
    result = build_bilateral_response(probs, attn, _make_model_info())
    assert result['attention_maps'] is attn


def test_build_bilateral_response_benign_class():
    from response_builder import build_bilateral_response
    probs = _make_probs([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    result = build_bilateral_response(probs, _make_attention_maps(), _make_model_info())
    assert result['left']['prediction'] == 'Benign'
