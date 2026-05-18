"""
Response parser for MedGemma text output
Single Responsibility: Parse generated JSON into structured classification
"""
import json
import logging
from typing import Dict, Any

from exceptions import ResponseParsingError

logger = logging.getLogger(__name__)

# Valid classification values
VALID_CLASSIFICATIONS = ["No lesion", "Benign", "Malignant"]


def clean_json_text(text: str) -> str:
    """
    Clean raw text to extract JSON content.

    Handles:
    - Leading/trailing whitespace
    - Markdown code blocks (```json ... ```)

    Args:
        text: Raw text output from MedGemma

    Returns:
        Cleaned JSON string
    """
    cleaned = text.strip()

    # Remove markdown code block wrapper if present
    if cleaned.startswith("```"):
        # Find the end of opening fence
        first_newline = cleaned.find('\n')
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]

        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    return cleaned


def validate_classification(value: str) -> str:
    """
    Validate classification value against allowed values.

    Args:
        value: Classification string from model

    Returns:
        Normalized classification string

    Raises:
        ResponseParsingError: If classification is invalid
    """
    for valid in VALID_CLASSIFICATIONS:
        if valid.lower() == value.lower().strip():
            return valid

    raise ResponseParsingError(
        f"Invalid classification '{value}'. Must be one of: {VALID_CLASSIFICATIONS}"
    )


def validate_confidence(value: Any) -> float:
    """
    Validate and normalize confidence value.

    Args:
        value: Confidence value from model

    Returns:
        Float confidence clamped to 0-100

    Raises:
        ResponseParsingError: If confidence is not a number
    """
    try:
        confidence = float(value)
        return max(0.0, min(100.0, confidence))
    except (ValueError, TypeError):
        raise ResponseParsingError(f"Invalid confidence value '{value}'. Must be a number 0-100")


def parse_bilateral_response(text: str) -> Dict[str, Any]:
    """
    Parse MedGemma JSON output into bilateral classification format.

    Expects JSON in format:
    {
        "left": {"classification": "...", "confidence": 0-100, "reasoning": "..."},
        "right": {"classification": "...", "confidence": 0-100, "reasoning": "..."}
    }

    Args:
        text: Raw text output from MedGemma

    Returns:
        Dictionary with structure:
        {
            "left": {"prediction": str, "confidence": float},
            "right": {"prediction": str, "confidence": float}
        }

    Raises:
        ResponseParsingError: If parsing fails
    """
    logger.info(f"Parsing MedGemma response ({len(text)} chars)")

    # Clean and parse JSON directly
    json_str = clean_json_text(text)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ResponseParsingError(f"Invalid JSON: {e}", raw_response=json_str)

    # Validate structure
    if "left" not in parsed or "right" not in parsed:
        raise ResponseParsingError(
            "Response must contain 'left' and 'right' keys",
            raw_response=text
        )

    result = {}

    for side in ["left", "right"]:
        side_data = parsed[side]

        if not isinstance(side_data, dict):
            raise ResponseParsingError(
                f"'{side}' must be an object with classification and confidence",
                raw_response=text
            )

        if "classification" not in side_data:
            raise ResponseParsingError(
                f"'{side}' missing required 'classification' field",
                raw_response=text
            )

        if "confidence" not in side_data:
            raise ResponseParsingError(
                f"'{side}' missing required 'confidence' field",
                raw_response=text
            )

        result[side] = {
            "prediction": validate_classification(side_data["classification"]),
            "confidence": round(validate_confidence(side_data["confidence"]), 1)
        }

    logger.info(f"Parsed response: left={result['left']}, right={result['right']}")

    return result
