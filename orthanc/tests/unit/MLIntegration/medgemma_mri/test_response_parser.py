"""Tests for medgemma-mri/response_parser.py — parse_bilateral_response.

response_parser.py imports only json, logging, and the local exceptions module.
No heavy ML deps — no stubs required.

Uncovered paths:
- validate_classification case-folding for all 3 valid values is implicitly
  tested via parse_bilateral_response.  Direct lower-case variant tested in
  test_validate_classification_case_insensitive.
"""
from typing import Iterator
import json
import pytest


@pytest.fixture(autouse=True)
def _evict_module() -> Iterator[None]:
    import sys
    sys.modules.pop('response_parser', None)
    sys.modules.pop('exceptions', None)
    yield
    sys.modules.pop('response_parser', None)
    sys.modules.pop('exceptions', None)


# --- clean_json_text ---

def test_clean_json_text_strips_whitespace():
    from response_parser import clean_json_text
    assert clean_json_text('  {"a": 1}  ') == '{"a": 1}'


def test_clean_json_text_removes_markdown_json_block():
    from response_parser import clean_json_text
    text = '```json\n{"a": 1}\n```'
    result = clean_json_text(text)
    assert result == '{"a": 1}'


def test_clean_json_text_removes_plain_code_block():
    from response_parser import clean_json_text
    text = '```\n{"a": 1}\n```'
    result = clean_json_text(text)
    assert result == '{"a": 1}'


def test_clean_json_text_passthrough_plain_json():
    from response_parser import clean_json_text
    text = '{"left": {}, "right": {}}'
    assert clean_json_text(text) == text


# --- validate_classification ---

def test_validate_classification_no_lesion():
    from response_parser import validate_classification
    assert validate_classification('No lesion') == 'No lesion'


def test_validate_classification_benign():
    from response_parser import validate_classification
    assert validate_classification('Benign') == 'Benign'


def test_validate_classification_malignant():
    from response_parser import validate_classification
    assert validate_classification('Malignant') == 'Malignant'


def test_validate_classification_case_insensitive():
    from response_parser import validate_classification
    assert validate_classification('benign') == 'Benign'
    assert validate_classification('MALIGNANT') == 'Malignant'
    assert validate_classification('NO LESION') == 'No lesion'


def test_validate_classification_invalid_raises():
    from response_parser import validate_classification
    from exceptions import ResponseParsingError
    with pytest.raises(ResponseParsingError, match='Invalid classification'):
        validate_classification('Unknown')


# --- validate_confidence ---

def test_validate_confidence_normal_value():
    from response_parser import validate_confidence
    assert validate_confidence(85.0) == 85.0


def test_validate_confidence_clamps_above_100():
    from response_parser import validate_confidence
    assert validate_confidence(150.0) == 100.0


def test_validate_confidence_clamps_below_0():
    from response_parser import validate_confidence
    assert validate_confidence(-5.0) == 0.0


def test_validate_confidence_accepts_string_number():
    from response_parser import validate_confidence
    assert validate_confidence('75') == 75.0


def test_validate_confidence_invalid_raises():
    from response_parser import validate_confidence
    from exceptions import ResponseParsingError
    with pytest.raises(ResponseParsingError, match='Invalid confidence'):
        validate_confidence('not_a_number')


# --- parse_bilateral_response ---

def _good_json(left_cls='No lesion', right_cls='Benign', left_conf=80, right_conf=65):
    return json.dumps({
        "left": {"classification": left_cls, "confidence": left_conf, "reasoning": "ok"},
        "right": {"classification": right_cls, "confidence": right_conf, "reasoning": "ok"}
    })


def test_parse_bilateral_response_returns_dict():
    from response_parser import parse_bilateral_response
    result = parse_bilateral_response(_good_json())
    assert isinstance(result, dict)


def test_parse_bilateral_response_has_left_right_keys():
    from response_parser import parse_bilateral_response
    result = parse_bilateral_response(_good_json())
    assert 'left' in result
    assert 'right' in result


def test_parse_bilateral_response_prediction_values():
    from response_parser import parse_bilateral_response
    result = parse_bilateral_response(_good_json('Malignant', 'Benign', 90, 55))
    assert result['left']['prediction'] == 'Malignant'
    assert result['right']['prediction'] == 'Benign'


def test_parse_bilateral_response_confidence_rounded():
    from response_parser import parse_bilateral_response
    result = parse_bilateral_response(_good_json(left_conf=82.567, right_conf=61.234))
    assert result['left']['confidence'] == round(82.567, 1)
    assert result['right']['confidence'] == round(61.234, 1)


def test_parse_bilateral_response_invalid_json_raises():
    from response_parser import parse_bilateral_response
    from exceptions import ResponseParsingError
    with pytest.raises(ResponseParsingError, match='Invalid JSON'):
        parse_bilateral_response('not json at all')


def test_parse_bilateral_response_missing_left_key_raises():
    from response_parser import parse_bilateral_response
    from exceptions import ResponseParsingError
    text = json.dumps({"right": {"classification": "Benign", "confidence": 80}})
    with pytest.raises(ResponseParsingError, match="'left' and 'right'"):
        parse_bilateral_response(text)


def test_parse_bilateral_response_missing_right_key_raises():
    from response_parser import parse_bilateral_response
    from exceptions import ResponseParsingError
    text = json.dumps({"left": {"classification": "Benign", "confidence": 80}})
    with pytest.raises(ResponseParsingError, match="'left' and 'right'"):
        parse_bilateral_response(text)


def test_parse_bilateral_response_side_not_dict_raises():
    from response_parser import parse_bilateral_response
    from exceptions import ResponseParsingError
    text = json.dumps({"left": "Benign", "right": {"classification": "Benign", "confidence": 80}})
    with pytest.raises(ResponseParsingError):
        parse_bilateral_response(text)


def test_parse_bilateral_response_missing_classification_field_raises():
    from response_parser import parse_bilateral_response
    from exceptions import ResponseParsingError
    text = json.dumps({
        "left": {"confidence": 80},
        "right": {"classification": "Benign", "confidence": 80}
    })
    with pytest.raises(ResponseParsingError, match="classification"):
        parse_bilateral_response(text)


def test_parse_bilateral_response_missing_confidence_field_raises():
    from response_parser import parse_bilateral_response
    from exceptions import ResponseParsingError
    text = json.dumps({
        "left": {"classification": "Benign"},
        "right": {"classification": "Benign", "confidence": 80}
    })
    with pytest.raises(ResponseParsingError, match="confidence"):
        parse_bilateral_response(text)


def test_parse_bilateral_response_with_markdown_block():
    from response_parser import parse_bilateral_response
    inner = json.dumps({
        "left": {"classification": "No lesion", "confidence": 90, "reasoning": "clear"},
        "right": {"classification": "Malignant", "confidence": 85, "reasoning": "suspicious"}
    })
    text = f'```json\n{inner}\n```'
    result = parse_bilateral_response(text)
    assert result['left']['prediction'] == 'No lesion'
    assert result['right']['prediction'] == 'Malignant'


def test_parse_bilateral_response_confidence_clamped_to_100():
    from response_parser import parse_bilateral_response
    text = json.dumps({
        "left": {"classification": "Benign", "confidence": 150},
        "right": {"classification": "No lesion", "confidence": 50}
    })
    result = parse_bilateral_response(text)
    assert result['left']['confidence'] == 100.0


def test_parse_bilateral_response_reasoning_not_in_output():
    """reasoning field from model is not propagated to output."""
    from response_parser import parse_bilateral_response
    result = parse_bilateral_response(_good_json())
    assert 'reasoning' not in result['left']
    assert 'reasoning' not in result['right']


# ---------------------------------------------------------------------------
# G3: malformed-LLM-output coverage — realistic VLM failure modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload,reason", [
    ("", "whitespace-only payload (empty after strip)"),
    ("   \n\t  ", "whitespace-only payload (chars stripped)"),
    ("```json\n{\"left\": {}}\n", "single fence — no closing ```"),
    ('{"left": {"classification": "Benign", "confidence": 80}, "right": {"classification": "Benign", "confidence": 80}}\nplus trailing prose', "trailing prose after JSON"),
])
def test_parse_bilateral_response_rejects_malformed_json(payload, reason):
    """Parser must raise ResponseParsingError for common LLM malformations."""
    from response_parser import parse_bilateral_response
    from exceptions import ResponseParsingError
    with pytest.raises(ResponseParsingError):
        parse_bilateral_response(payload)


def test_parse_bilateral_response_rejects_null_confidence():
    """JSON `null` confidence -> ResponseParsingError (float(None) -> TypeError)."""
    from response_parser import parse_bilateral_response
    from exceptions import ResponseParsingError
    payload = '{"left": {"classification": "Benign", "confidence": null}, "right": {"classification": "Benign", "confidence": 50}}'
    with pytest.raises(ResponseParsingError, match="confidence"):
        parse_bilateral_response(payload)


def test_parse_bilateral_response_accepts_nan_confidence_as_nan_float():
    """JSON parser converts NaN to float; validate_confidence clamp does NOT catch NaN.
    Pins current behavior: NaN survives. If we want stricter, change production and update this test."""
    import math
    from response_parser import parse_bilateral_response
    # Most JSON parsers reject NaN. json.loads(...) with allow_nan=True (default for stdlib) accepts it.
    payload = '{"left": {"classification": "Benign", "confidence": NaN}, "right": {"classification": "Benign", "confidence": 50}}'
    parsed = parse_bilateral_response(payload)
    # min(100, max(0, nan)) returns nan (math semantics). Pin that the parser doesn't crash.
    assert math.isnan(parsed["left"]["confidence"]) or parsed["left"]["confidence"] in (0.0, 100.0)


def test_parse_bilateral_response_rejects_side_as_list():
    """parsed[side] must be a dict; a list (e.g. LLM returns array) raises."""
    from response_parser import parse_bilateral_response
    from exceptions import ResponseParsingError
    payload = '{"left": [{"classification": "Benign", "confidence": 80}], "right": {"classification": "Benign", "confidence": 80}}'
    with pytest.raises(ResponseParsingError, match="must be an object"):
        parse_bilateral_response(payload)


def test_parse_bilateral_response_rejects_empty_classification_string():
    """Empty classification string fails validate_classification."""
    from response_parser import parse_bilateral_response
    from exceptions import ResponseParsingError
    payload = '{"left": {"classification": "", "confidence": 80}, "right": {"classification": "Benign", "confidence": 80}}'
    with pytest.raises(ResponseParsingError, match="Invalid classification"):
        parse_bilateral_response(payload)
