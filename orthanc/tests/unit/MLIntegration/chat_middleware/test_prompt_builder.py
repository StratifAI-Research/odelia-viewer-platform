"""Tests for chat-middleware/prompt_builder.py — OpenAI-format message assembly."""
import pytest


def _config_env(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_FOLDER", str(tmp_path / "pb-img"))
    import config
    config.config = None
    import runtime_config
    runtime_config._runtime_config = None


def test_build_messages_text_only_user_content_is_plain_string(tmp_path, monkeypatch):
    """No images -> user content is the raw text, not an interleaved array."""
    _config_env(tmp_path, monkeypatch)
    from prompt_builder import PromptBuilder
    pb = PromptBuilder()
    msgs, user_content = pb.build_messages(
        conversation_history=[], new_user_message="hello", series_images={},
    )
    assert isinstance(user_content, str)
    assert user_content == "hello"
    # Total: system + user = 2
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "hello"


def test_build_messages_with_single_series_interleaves_image_and_slice_labels(tmp_path, monkeypatch):
    """images interleave as [image_url, 'SLICE 1', image_url, 'SLICE 2', ..., user_text]."""
    _config_env(tmp_path, monkeypatch)
    from prompt_builder import PromptBuilder
    pb = PromptBuilder()
    msgs, user_content = pb.build_messages(
        conversation_history=[],
        new_user_message="describe the slices",
        series_images={"1.2.3.SE1": ["data:image/png;base64,AAA", "data:image/png;base64,BBB"]},
    )
    assert isinstance(user_content, list)
    # 2 images x 2 parts each (image_url + slice label) + user text = 5
    assert len(user_content) == 5
    assert user_content[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    assert user_content[1] == {"type": "text", "text": "SLICE 1"}
    assert user_content[2] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}}
    assert user_content[3] == {"type": "text", "text": "SLICE 2"}
    assert user_content[4] == {"type": "text", "text": "describe the slices"}


def test_build_messages_multiple_series_concatenate_image_lists(tmp_path, monkeypatch):
    """All series''' images flattened in dict-insertion order; slice labels keep ascending."""
    _config_env(tmp_path, monkeypatch)
    from prompt_builder import PromptBuilder
    pb = PromptBuilder()
    _, user_content = pb.build_messages(
        conversation_history=[],
        new_user_message="all of these",
        series_images={
            "SE1": ["img.a"],
            "SE2": ["img.b", "img.c"],
        },
    )
    # 3 images x 2 + user text = 7
    assert len(user_content) == 7
    slice_labels = [c["text"] for c in user_content if c.get("type") == "text" and c["text"].startswith("SLICE")]
    assert slice_labels == ["SLICE 1", "SLICE 2", "SLICE 3"]
    image_urls = [c["image_url"]["url"] for c in user_content if c.get("type") == "image_url"]
    assert image_urls == ["img.a", "img.b", "img.c"]


def test_build_messages_replays_conversation_history_as_is(tmp_path, monkeypatch):
    """History messages preserved in order between system prompt and new user message."""
    _config_env(tmp_path, monkeypatch)
    from prompt_builder import PromptBuilder
    pb = PromptBuilder()
    history = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]
    msgs, _ = pb.build_messages(
        conversation_history=history, new_user_message="follow-up", series_images={},
    )
    assert len(msgs) == 4         # system + 2 history + 1 new
    assert msgs[1]["content"] == "previous question"
    assert msgs[2]["content"] == "previous answer"
    assert msgs[3]["content"] == "follow-up"


def test_build_messages_system_prompt_from_runtime_config(tmp_path, monkeypatch):
    """The system message uses runtime_config.system_prompt at call time."""
    _config_env(tmp_path, monkeypatch)
    from runtime_config import get_runtime_config
    rc = get_runtime_config()
    rc.system_prompt = "you are a quiet doctor"
    from prompt_builder import PromptBuilder
    pb = PromptBuilder()
    msgs, _ = pb.build_messages(conversation_history=[], new_user_message="hi", series_images={})
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "you are a quiet doctor"


def test_prompt_builder_uses_explicit_config_when_provided(tmp_path, monkeypatch):
    """If a RuntimeConfig is passed, the singleton is not consulted (the singleton is not consulted)."""
    _config_env(tmp_path, monkeypatch)
    from runtime_config import RuntimeConfig
    explicit = RuntimeConfig()
    explicit.system_prompt = "explicit-prompt-marker"
    from prompt_builder import PromptBuilder
    pb = PromptBuilder(runtime_config=explicit)
    msgs, _ = pb.build_messages(conversation_history=[], new_user_message="hi", series_images={})
    assert msgs[0]["content"] == "explicit-prompt-marker"


def test_prompt_builder_falls_back_to_singleton_when_no_config(tmp_path, monkeypatch):
    """No explicit config -> .config falls through to get_runtime_config()."""
    _config_env(tmp_path, monkeypatch)
    from prompt_builder import PromptBuilder
    from runtime_config import get_runtime_config
    pb = PromptBuilder()
    assert pb.config is get_runtime_config()


def test_get_prompt_builder_returns_singleton(tmp_path, monkeypatch):
    _config_env(tmp_path, monkeypatch)
    import prompt_builder as _pb_mod
    _pb_mod._prompt_builder = None
    a = _pb_mod.get_prompt_builder()
    b = _pb_mod.get_prompt_builder()
    assert a is b
