"""Tests for medgemma-mri/exceptions.py."""
import pytest


def test_model_not_loaded_error_is_exception():
    from exceptions import ModelNotLoadedError
    assert issubclass(ModelNotLoadedError, Exception)


def test_model_authentication_error_is_exception():
    from exceptions import ModelAuthenticationError
    assert issubclass(ModelAuthenticationError, Exception)


def test_inference_error_is_exception():
    from exceptions import InferenceError
    assert issubclass(InferenceError, Exception)


def test_response_parsing_error_is_exception():
    from exceptions import ResponseParsingError
    assert issubclass(ResponseParsingError, Exception)


def test_model_not_loaded_error_default_message():
    from exceptions import ModelNotLoadedError
    err = ModelNotLoadedError()
    assert "not loaded" in str(err).lower()


def test_model_not_loaded_error_custom_message():
    from exceptions import ModelNotLoadedError
    err = ModelNotLoadedError("model offline")
    assert "model offline" in str(err)


def test_model_authentication_error_default_message():
    from exceptions import ModelAuthenticationError
    err = ModelAuthenticationError()
    assert "authentication" in str(err).lower()


def test_model_authentication_error_custom_message():
    from exceptions import ModelAuthenticationError
    err = ModelAuthenticationError("bad token")
    assert "bad token" in str(err)


def test_inference_error_default_message():
    from exceptions import InferenceError
    err = InferenceError()
    assert "inference" in str(err).lower()


def test_inference_error_custom_message():
    from exceptions import InferenceError
    err = InferenceError("OOM during forward pass")
    assert "OOM" in str(err)


def test_response_parsing_error_default_message():
    from exceptions import ResponseParsingError
    err = ResponseParsingError()
    assert "parse" in str(err).lower()


def test_response_parsing_error_raw_response_stored():
    from exceptions import ResponseParsingError
    raw = '{"left": broken json}'
    err = ResponseParsingError("bad json", raw_response=raw)
    assert err.raw_response == raw


def test_response_parsing_error_raw_response_none_by_default():
    from exceptions import ResponseParsingError
    err = ResponseParsingError()
    assert err.raw_response is None


def test_all_exceptions_can_be_raised_and_caught():
    from exceptions import ModelNotLoadedError, ModelAuthenticationError, InferenceError, ResponseParsingError
    for cls in [ModelNotLoadedError, ModelAuthenticationError, InferenceError, ResponseParsingError]:
        with pytest.raises(cls):
            raise cls("test")


def test_exceptions_are_distinct_types():
    from exceptions import ModelNotLoadedError, ModelAuthenticationError, InferenceError, ResponseParsingError
    types_ = [ModelNotLoadedError, ModelAuthenticationError, InferenceError, ResponseParsingError]
    assert len(set(types_)) == 4


# ---------------------------------------------------------------------------
# G1: PreprocessingError parity with the other four exception classes
# ---------------------------------------------------------------------------

def test_preprocessing_error_is_exception():
    from exceptions import PreprocessingError
    assert issubclass(PreprocessingError, Exception)


def test_preprocessing_error_default_message():
    from exceptions import PreprocessingError
    err = PreprocessingError()
    assert "preprocessing" in str(err).lower()


def test_preprocessing_error_custom_message():
    from exceptions import PreprocessingError
    err = PreprocessingError("dcm read failed")
    assert "dcm read failed" in str(err)


def test_preprocessing_error_can_be_raised_and_caught():
    from exceptions import PreprocessingError
    with pytest.raises(PreprocessingError):
        raise PreprocessingError("test")


def test_all_five_exceptions_are_distinct_types():
    """Includes PreprocessingError — the previous distinct-count was 4 and silently
    excluded this class. With G1, all 5 service exceptions must be distinct."""
    from exceptions import (
        ModelNotLoadedError, ModelAuthenticationError, InferenceError,
        PreprocessingError, ResponseParsingError,
    )
    types_ = [ModelNotLoadedError, ModelAuthenticationError, InferenceError,
              PreprocessingError, ResponseParsingError]
    assert len(set(types_)) == 5
