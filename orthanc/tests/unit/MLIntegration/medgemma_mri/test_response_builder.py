"""Tests for medgemma-mri/response_builder.py — build_bilateral_response.

No heavy deps — no stubs required.
"""
from typing import Iterator
import pytest


@pytest.fixture(autouse=True)
def _evict_module() -> Iterator[None]:
    import sys
    sys.modules.pop('response_builder', None)
    yield
    sys.modules.pop('response_builder', None)


def _model_info():
    return {
        "model_name": "MedGemma",
        "architecture": "Vision-Language Model",
        "version": "1.5-4b-it"
    }


def _parsed_result(left_pred='No lesion', left_conf=80.0, right_pred='Benign', right_conf=65.0):
    return {
        "left": {"prediction": left_pred, "confidence": left_conf},
        "right": {"prediction": right_pred, "confidence": right_conf}
    }


def test_build_bilateral_response_returns_dict():
    from response_builder import build_bilateral_response
    result = build_bilateral_response(_parsed_result(), _model_info())
    assert isinstance(result, dict)


def test_build_bilateral_response_has_left_right_keys():
    from response_builder import build_bilateral_response
    result = build_bilateral_response(_parsed_result(), _model_info())
    assert 'left' in result
    assert 'right' in result


def test_build_bilateral_response_has_model_metadata():
    from response_builder import build_bilateral_response
    result = build_bilateral_response(_parsed_result(), _model_info())
    assert 'model_metadata' in result


def test_build_bilateral_response_prediction_passthrough():
    from response_builder import build_bilateral_response
    result = build_bilateral_response(_parsed_result('Malignant', 90.0, 'No lesion', 55.0), _model_info())
    assert result['left']['prediction'] == 'Malignant'
    assert result['right']['prediction'] == 'No lesion'


def test_build_bilateral_response_confidence_rounded_to_1dp():
    from response_builder import build_bilateral_response
    result = build_bilateral_response(_parsed_result(left_conf=82.567, right_conf=61.234), _model_info())
    assert result['left']['confidence'] == round(82.567, 1)
    assert result['right']['confidence'] == round(61.234, 1)


def test_build_bilateral_response_model_metadata_values():
    from response_builder import build_bilateral_response
    info = _model_info()
    result = build_bilateral_response(_parsed_result(), info)
    assert result['model_metadata']['model_name'] == 'MedGemma'
    assert result['model_metadata']['architecture'] == 'Vision-Language Model'
    assert result['model_metadata']['version'] == '1.5-4b-it'


def test_build_bilateral_response_model_metadata_defaults():
    """When model_info is missing keys, defaults are used."""
    from response_builder import build_bilateral_response
    result = build_bilateral_response(_parsed_result(), {})
    assert result['model_metadata']['model_name'] == 'MedGemma'
    assert result['model_metadata']['architecture'] == 'Vision-Language Model'
    assert result['model_metadata']['version'] == '1.5-4b-it'


def test_build_bilateral_response_missing_side_uses_default():
    """If parsed_result lacks a side, default No lesion / 50.0 is used."""
    from response_builder import build_bilateral_response
    result = build_bilateral_response({}, _model_info())
    assert result['left']['prediction'] == 'No lesion'
    assert result['left']['confidence'] == 50.0
    assert result['right']['prediction'] == 'No lesion'
    assert result['right']['confidence'] == 50.0


def test_build_bilateral_response_confidence_zero():
    from response_builder import build_bilateral_response
    result = build_bilateral_response(_parsed_result(left_conf=0.0, right_conf=0.0), _model_info())
    assert result['left']['confidence'] == 0.0
    assert result['right']['confidence'] == 0.0


def test_build_bilateral_response_confidence_100():
    from response_builder import build_bilateral_response
    result = build_bilateral_response(_parsed_result(left_conf=100.0, right_conf=100.0), _model_info())
    assert result['left']['confidence'] == 100.0
    assert result['right']['confidence'] == 100.0
