"""Tests for chat-middleware/models.py — pydantic schemas + enums."""


def test_client_message_type_enum_values():
    import models
    assert models.ClientMessageType.CHAT.value == "chat"
    assert models.ClientMessageType.CANCEL.value == "cancel"


def test_server_message_type_enum_values():
    import models
    assert models.ServerMessageType.CONNECTED.value == "connected"
    assert models.ServerMessageType.TOKEN.value == "token"
    assert models.ServerMessageType.THINKING_TOKEN.value == "thinking_token"
    assert models.ServerMessageType.DONE.value == "done"
    assert models.ServerMessageType.ERROR.value == "error"
    assert models.ServerMessageType.PREPROCESSING.value == "preprocessing"


def test_slice_strategy_enum_values():
    import models
    assert models.SliceStrategy.CENTRAL.value == "central"
    assert models.SliceStrategy.UNIFORM.value == "uniform"
    assert models.SliceStrategy.FIRST_N.value == "first_n"
    assert models.SliceStrategy.LAST_N.value == "last_n"


def test_client_message_round_trip_chat():
    import models
    msg = models.ClientMessage(type="chat", content="hi", study_uid="1.2.3",
                                 series_uids=["1.2.3.1", "1.2.3.2"])
    assert msg.type == models.ClientMessageType.CHAT
    assert msg.content == "hi"
    assert msg.series_uids == ["1.2.3.1", "1.2.3.2"]
    # Round-trip through JSON
    payload = msg.model_dump_json()
    parsed = models.ClientMessage.model_validate_json(payload)
    assert parsed == msg


def test_client_message_cancel_no_content_required():
    import models
    msg = models.ClientMessage(type="cancel")
    assert msg.type == models.ClientMessageType.CANCEL
    assert msg.content is None


def test_server_message_token_with_content():
    import models
    msg = models.ServerMessage(type="token", content="lorem")
    assert msg.type == models.ServerMessageType.TOKEN
    assert msg.content == "lorem"


def test_server_message_connected_with_session_id():
    import models
    msg = models.ServerMessage(type="connected", session_id="sess-1")
    assert msg.session_id == "sess-1"


def test_preprocessing_config_accepts_partial_fields():
    import models
    cfg = models.PreprocessingConfig(num_slices=7)
    assert cfg.num_slices == 7
    assert cfg.slice_strategy is None


def test_preprocessing_config_with_slice_strategy_enum():
    import models
    cfg = models.PreprocessingConfig(slice_strategy="uniform")
    assert cfg.slice_strategy == models.SliceStrategy.UNIFORM


def test_ollama_options_config_all_fields_optional():
    import models
    cfg = models.OllamaOptionsConfig()
    assert cfg.max_tokens is None
    assert cfg.temperature is None


def test_debug_config_update_accepts_nested_preprocessing():
    import models
    upd = models.DebugConfigUpdate(
        system_prompt="be terse",
        preprocessing=models.PreprocessingConfig(num_slices=3),
    )
    assert upd.system_prompt == "be terse"
    assert upd.preprocessing.num_slices == 3


def test_debug_config_update_rejects_overlong_system_prompt():
    import pydantic
    import pytest

    import models
    too_long = "x" * (models.MAX_SYSTEM_PROMPT_LEN + 1)
    with pytest.raises(pydantic.ValidationError):
        models.DebugConfigUpdate(system_prompt=too_long)


def test_debug_config_update_accepts_max_length_system_prompt():
    import models
    at_limit = "x" * models.MAX_SYSTEM_PROMPT_LEN
    upd = models.DebugConfigUpdate(system_prompt=at_limit)
    assert upd.system_prompt == at_limit


def test_session_info_required_fields():
    import models
    info = models.SessionInfo(
        session_id="s1", created_at="2026-01-01T00:00:00Z",
        last_activity="2026-01-01T00:01:00Z", message_count=3,
    )
    assert info.message_count == 3


def test_session_list_response_with_two_sessions():
    import models
    payload = models.SessionListResponse(sessions=[
        models.SessionInfo(session_id="s1", created_at="2026-01-01T00:00:00Z",
                            last_activity="2026-01-01T00:01:00Z", message_count=1),
        models.SessionInfo(session_id="s2", created_at="2026-01-02T00:00:00Z",
                            last_activity="2026-01-02T00:05:00Z", message_count=5),
    ])
    assert len(payload.sessions) == 2


def test_cache_clear_response_fields():
    import models
    resp = models.CacheClearResponse(cleared_entries=10, message="done")
    assert resp.cleared_entries == 10


# ---------- SliceSelection ----------

def test_slice_selection_defaults_to_naming_no_instances():
    """No named instances means 'use the configured recipe', the pre-existing behaviour."""
    import models
    sel = models.SliceSelection(series_uid="1.2.3")
    assert sel.sop_instance_uids == []
    assert sel.range_start is None


def test_slice_selection_round_trips_named_instances_in_order():
    import models
    sel = models.SliceSelection(
        series_uid="1.2.3", sop_instance_uids=["1.1", "1.2", "1.3"],
        range_start=18, range_end=62, total_slices=103,
    )
    parsed = models.SliceSelection.model_validate_json(sel.model_dump_json())
    assert parsed.sop_instance_uids == ["1.1", "1.2", "1.3"]
    assert (parsed.range_start, parsed.range_end, parsed.total_slices) == (18, 62, 103)


def test_slice_selection_rejects_an_unbounded_instance_list():
    """The list decides how many images go into one LLM call, so it is bounded."""
    import models
    import pydantic
    import pytest
    too_many = [f"1.{i}" for i in range(models.MAX_SLICES_PER_SERIES + 1)]
    with pytest.raises(pydantic.ValidationError):
        models.SliceSelection(series_uid="1.2.3", sop_instance_uids=too_many)


def test_slice_selection_rejects_an_empty_series_uid():
    import models
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        models.SliceSelection(series_uid="")


def test_slice_selection_rejects_a_zero_based_range():
    """The range is 1-based, matching what the panel shows the user."""
    import models
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        models.SliceSelection(series_uid="1.2.3", range_start=0)


def test_client_message_carries_per_series_slice_selections():
    import models
    msg = models.ClientMessage(
        type="chat", content="hi", study_uid="1.2.3", series_uids=["SE1", "SE2"],
        slice_selections=[
            {"series_uid": "SE1", "sop_instance_uids": ["1.1", "1.2"]},
            {"series_uid": "SE2"},
        ],
    )
    assert [s.series_uid for s in msg.slice_selections] == ["SE1", "SE2"]
    assert msg.slice_selections[0].sop_instance_uids == ["1.1", "1.2"]
    assert msg.slice_selections[1].sop_instance_uids == []


def test_client_message_without_slice_selections_still_parses():
    """A viewer predating the field must keep working."""
    import models
    msg = models.ClientMessage(type="chat", content="hi", series_uids=["SE1"])
    assert msg.slice_selections is None


def test_client_message_rejects_too_many_series_selections():
    import models
    import pydantic
    import pytest
    too_many = [
        {"series_uid": f"SE{i}"} for i in range(models.MAX_SERIES_PER_MESSAGE + 1)
    ]
    with pytest.raises(pydantic.ValidationError):
        models.ClientMessage(type="chat", content="hi", slice_selections=too_many)
