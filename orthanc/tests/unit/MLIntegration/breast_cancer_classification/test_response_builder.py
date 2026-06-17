"""Tests for breast-cancer-classification/response_builder.py — build_bilateral_classification."""


def test_build_bilateral_classification_returns_dict():
    from response_builder import build_bilateral_classification
    left = {"prediction": "Malignant", "confidence": 87.5}
    right = {"prediction": "Benign", "confidence": 62.3}
    result = build_bilateral_classification(left, right)
    assert isinstance(result, dict)


def test_build_bilateral_classification_has_left_right_keys():
    from response_builder import build_bilateral_classification
    left = {"prediction": "Malignant", "confidence": 80.0}
    right = {"prediction": "Benign", "confidence": 55.0}
    result = build_bilateral_classification(left, right)
    assert "left" in result
    assert "right" in result


def test_build_bilateral_classification_passes_through_left():
    from response_builder import build_bilateral_classification
    left = {"prediction": "Malignant", "confidence": 92.0}
    right = {"prediction": "Benign", "confidence": 60.0}
    result = build_bilateral_classification(left, right)
    assert result["left"] is left


def test_build_bilateral_classification_passes_through_right():
    from response_builder import build_bilateral_classification
    left = {"prediction": "Benign", "confidence": 65.0}
    right = {"prediction": "Malignant", "confidence": 88.5}
    result = build_bilateral_classification(left, right)
    assert result["right"] is right


def test_build_bilateral_classification_no_extra_keys():
    from response_builder import build_bilateral_classification
    left = {"prediction": "Malignant", "confidence": 75.0}
    right = {"prediction": "Benign", "confidence": 50.0}
    result = build_bilateral_classification(left, right)
    assert set(result.keys()) == {"left", "right"}


def test_build_bilateral_classification_with_error_result():
    from response_builder import build_bilateral_classification
    left = {"error": "Missing Pre contrast image for left side"}
    right = {"prediction": "Benign", "confidence": 60.0}
    result = build_bilateral_classification(left, right)
    assert result["left"]["error"] == "Missing Pre contrast image for left side"
    assert result["right"]["prediction"] == "Benign"


def test_build_bilateral_classification_both_malignant():
    from response_builder import build_bilateral_classification
    left = {"prediction": "Malignant", "confidence": 91.0}
    right = {"prediction": "Malignant", "confidence": 85.0}
    result = build_bilateral_classification(left, right)
    assert result["left"]["prediction"] == "Malignant"
    assert result["right"]["prediction"] == "Malignant"
