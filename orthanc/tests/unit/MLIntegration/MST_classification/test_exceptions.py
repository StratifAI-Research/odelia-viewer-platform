"""Tests for MST-classification/exceptions.py."""
import pytest


def test_model_not_loaded_error_is_exception():
    from exceptions import ModelNotLoadedError
    assert issubclass(ModelNotLoadedError, Exception)


def test_inference_error_is_exception():
    from exceptions import InferenceError
    assert issubclass(InferenceError, Exception)


def test_preprocessing_error_is_exception():
    from exceptions import PreprocessingError
    assert issubclass(PreprocessingError, Exception)


def test_model_not_loaded_error_carries_message():
    from exceptions import ModelNotLoadedError
    err = ModelNotLoadedError("model not loaded yet")
    assert "model not loaded yet" in str(err)


def test_inference_error_carries_message():
    from exceptions import InferenceError
    err = InferenceError("inference failed: OOM")
    assert "inference failed: OOM" in str(err)


def test_preprocessing_error_carries_message():
    from exceptions import PreprocessingError
    err = PreprocessingError("bad input shape")
    assert "bad input shape" in str(err)


def test_model_not_loaded_error_can_be_raised_and_caught():
    from exceptions import ModelNotLoadedError
    with pytest.raises(ModelNotLoadedError, match="not ready"):
        raise ModelNotLoadedError("not ready")


def test_inference_error_can_be_raised_and_caught():
    from exceptions import InferenceError
    with pytest.raises(InferenceError):
        raise InferenceError("failed")


def test_errors_are_distinct_types():
    from exceptions import ModelNotLoadedError, InferenceError, PreprocessingError
    assert ModelNotLoadedError is not InferenceError
    assert InferenceError is not PreprocessingError
    assert ModelNotLoadedError is not PreprocessingError
