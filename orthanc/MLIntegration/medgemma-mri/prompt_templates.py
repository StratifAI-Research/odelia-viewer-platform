"""
Prompt templates for MedGemma breast MRI analysis
Single Responsibility: Build structured prompts for consistent model output
"""
from typing import List, Tuple
from PIL import Image


def build_breast_mri_prompt() -> Tuple[str, str]:
    """
    Returns (instruction, query_text) for breast MRI analysis.

    The prompt is designed to get structured JSON output from MedGemma
    with bilateral classification results.

    Returns:
        Tuple of (instruction_text, query_text)
    """
    instruction = (
        "You are a radiologist analyzing a breast MRI scan. "
        "You are reviewing a contiguous block of axial slices from "
        "the breast region. Please examine each slice carefully for "
        "masses, enhancement patterns, and morphological features."
    )

    query_text = (
        "\n\nBased on the visual evidence in the slices provided above, "
        "analyze both breasts for lesions. For each breast (left and right), "
        "classify findings as: 'No lesion', 'Benign', or 'Malignant'.\n\n"
        "Respond ONLY with a JSON object in this exact format:\n"
        "{\n"
        '  "left": {"classification": "No lesion" or "Benign" or "Malignant", "confidence": 0-100, "reasoning": "brief explanation"},\n'
        '  "right": {"classification": "No lesion" or "Benign" or "Malignant", "confidence": 0-100, "reasoning": "brief explanation"}\n'
        "}\n\n"
        "Important: Output ONLY the JSON object, no additional text."
    )

    return instruction, query_text


def build_message_content(slices: List[Image.Image]) -> List[dict]:
    """
    Build MedGemma message content with interleaved images and text.

    Follows the MedGemma multi-slice input pattern:
    [instruction, img1, "SLICE 1", img2, "SLICE 2", ..., query_text]

    Args:
        slices: List of PIL Images extracted from MRI volume

    Returns:
        List of content items for MedGemma message format
    """
    instruction, query_text = build_breast_mri_prompt()

    content = [{"type": "text", "text": instruction}]

    for slice_number, mri_slice in enumerate(slices, 1):
        content.append({"type": "image", "image": mri_slice})
        content.append({"type": "text", "text": f"SLICE {slice_number}"})

    content.append({"type": "text", "text": query_text})

    return content


def build_messages(slices: List[Image.Image]) -> List[dict]:
    """
    Build complete message structure for MedGemma inference.

    Args:
        slices: List of PIL Images extracted from MRI volume

    Returns:
        List with single user message containing interleaved content
    """
    content = build_message_content(slices)

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    return messages
