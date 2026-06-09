"""Tests for breast-cancer-classification/response_builder.py — build_bilateral_classification."""


def test_build_bilateral_classification_returns_dict():
    from response_builder import build_bilateral_classification
    left = {"prediction": "Cancerous", "confidence": 87.5}
    right = {"prediction": "Not Cancerous", "confidence": 62.3}
    result = build_bilateral_classification(left, right)
    assert isinstance(result, dict)


def test_build_bilateral_classification_has_left_right_keys():
    from response_builder import build_bilateral_classification
    left = {"prediction": "Cancerous", "confidence": 80.0}
    right = {"prediction": "Not Cancerous", "confidence": 55.0}
    result = build_bilateral_classification(left, right)
    assert "left" in result
    assert "right" in result


def test_build_bilateral_classification_passes_through_left():
    from response_builder import build_bilateral_classification
    left = {"prediction": "Cancerous", "confidence": 92.0}
    right = {"prediction": "Not Cancerous", "confidence": 60.0}
    result = build_bilateral_classification(left, right)
    assert result["left"] is left


def test_build_bilateral_classification_passes_through_right():
    from response_builder import build_bilateral_classification
    left = {"prediction": "Not Cancerous", "confidence": 65.0}
    right = {"prediction": "Cancerous", "confidence": 88.5}
    result = build_bilateral_classification(left, right)
    assert result["right"] is right


def test_build_bilateral_classification_no_extra_keys():
    from response_builder import build_bilateral_classification
    left = {"prediction": "Cancerous", "confidence": 75.0}
    right = {"prediction": "Not Cancerous", "confidence": 50.0}
    result = build_bilateral_classification(left, right)
    assert set(result.keys()) == {"left", "right"}


def test_build_bilateral_classification_with_error_result():
    from response_builder import build_bilateral_classification
    left = {"error": "Missing Pre contrast image for left side"}
    right = {"prediction": "Not Cancerous", "confidence": 60.0}
    result = build_bilateral_classification(left, right)
    assert result["left"]["error"] == "Missing Pre contrast image for left side"
    assert result["right"]["prediction"] == "Not Cancerous"


def test_build_bilateral_classification_both_cancerous():
    from response_builder import build_bilateral_classification
    left = {"prediction": "Cancerous", "confidence": 91.0}
    right = {"prediction": "Cancerous", "confidence": 85.0}
    result = build_bilateral_classification(left, right)
    assert result["left"]["prediction"] == "Cancerous"
    assert result["right"]["prediction"] == "Cancerous"
