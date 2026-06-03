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
