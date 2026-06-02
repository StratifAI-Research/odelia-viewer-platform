"""Tests for medgemma-mri/prompt_templates.py — build_messages and helpers.

No heavy deps — PIL is available in venv.  No stubs required.
"""
from typing import Iterator
from PIL import Image as PILImage
import pytest


@pytest.fixture(autouse=True)
def _evict_module() -> Iterator[None]:
    import sys
    sys.modules.pop('prompt_templates', None)
    yield
    sys.modules.pop('prompt_templates', None)


def _make_pil_image():
    return PILImage.new('RGB', (64, 64), color=(128, 128, 128))


# --- build_breast_mri_prompt ---

def test_build_breast_mri_prompt_returns_two_strings():
    from prompt_templates import build_breast_mri_prompt
    instruction, query = build_breast_mri_prompt()
    assert isinstance(instruction, str)
    assert isinstance(query, str)


def test_build_breast_mri_prompt_instruction_mentions_radiologist():
    from prompt_templates import build_breast_mri_prompt
    instruction, _ = build_breast_mri_prompt()
    assert 'radiologist' in instruction.lower()


def test_build_breast_mri_prompt_query_mentions_json():
    from prompt_templates import build_breast_mri_prompt
    _, query = build_breast_mri_prompt()
    assert 'json' in query.lower()


def test_build_breast_mri_prompt_query_mentions_valid_classifications():
    from prompt_templates import build_breast_mri_prompt
    _, query = build_breast_mri_prompt()
    for cls in ['No lesion', 'Benign', 'Malignant']:
        assert cls in query


def test_build_breast_mri_prompt_query_mentions_left_right():
    from prompt_templates import build_breast_mri_prompt
    _, query = build_breast_mri_prompt()
    assert 'left' in query.lower()
    assert 'right' in query.lower()


# --- build_message_content ---

def test_build_message_content_returns_list():
    from prompt_templates import build_message_content
    slices = [_make_pil_image()]
    result = build_message_content(slices)
    assert isinstance(result, list)


def test_build_message_content_first_item_is_text():
    from prompt_templates import build_message_content
    slices = [_make_pil_image()]
    result = build_message_content(slices)
    assert result[0]['type'] == 'text'


def test_build_message_content_last_item_is_text():
    from prompt_templates import build_message_content
    slices = [_make_pil_image()]
    result = build_message_content(slices)
    assert result[-1]['type'] == 'text'


def test_build_message_content_interleaves_image_and_slice_label():
    from prompt_templates import build_message_content
    slices = [_make_pil_image(), _make_pil_image()]
    result = build_message_content(slices)
    # items 1,2 should be image, text(SLICE 1); items 3,4 should be image, text(SLICE 2)
    assert result[1]['type'] == 'image'
    assert result[2]['type'] == 'text' and 'SLICE 1' in result[2]['text']
    assert result[3]['type'] == 'image'
    assert result[4]['type'] == 'text' and 'SLICE 2' in result[4]['text']


def test_build_message_content_image_items_are_pil_images():
    from prompt_templates import build_message_content
    img = _make_pil_image()
    slices = [img]
    result = build_message_content(slices)
    image_items = [item for item in result if item['type'] == 'image']
    assert len(image_items) == 1
    assert isinstance(image_items[0]['image'], PILImage.Image)


def test_build_message_content_count_for_n_slices():
    """For N slices: 1 instruction + N*(image+text) + 1 query = 2N+2 items."""
    from prompt_templates import build_message_content
    n = 5
    slices = [_make_pil_image() for _ in range(n)]
    result = build_message_content(slices)
    assert len(result) == 2 * n + 2


# --- build_messages ---

def test_build_messages_returns_list_with_one_message():
    from prompt_templates import build_messages
    slices = [_make_pil_image()]
    result = build_messages(slices)
    assert isinstance(result, list)
    assert len(result) == 1


def test_build_messages_message_role_is_user():
    from prompt_templates import build_messages
    slices = [_make_pil_image()]
    result = build_messages(slices)
    assert result[0]['role'] == 'user'


def test_build_messages_message_has_content():
    from prompt_templates import build_messages
    slices = [_make_pil_image()]
    result = build_messages(slices)
    assert 'content' in result[0]
    assert isinstance(result[0]['content'], list)


def test_build_messages_content_for_3_slices():
    from prompt_templates import build_messages
    slices = [_make_pil_image() for _ in range(3)]
    result = build_messages(slices)
    content = result[0]['content']
    assert len(content) == 3 * 2 + 2  # 3 slices * (image+text) + instruction + query
