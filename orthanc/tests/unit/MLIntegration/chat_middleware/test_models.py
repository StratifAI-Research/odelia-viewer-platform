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


# ---------- RegionOfInterest ----------

def _roi(**kw):
    import models
    return models.RegionOfInterest(**{"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4, **kw})


def test_roi_round_trips_its_fractions():
    import models
    roi = _roi()
    parsed = models.RegionOfInterest.model_validate_json(roi.model_dump_json())
    assert (parsed.x, parsed.y, parsed.width, parsed.height) == (0.1, 0.2, 0.3, 0.4)


def test_roi_accepts_the_whole_image():
    roi = _roi(x=0.0, y=0.0, width=1.0, height=1.0)
    assert roi.width == 1.0


def test_roi_rejects_a_zero_width_rectangle():
    """A degenerate rectangle would crop to nothing; refuse rather than widen it."""
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        _roi(width=0.0)


def test_roi_rejects_a_negative_origin():
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        _roi(x=-0.1)


def test_roi_rejects_a_rectangle_running_past_the_edge():
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        _roi(x=0.8, width=0.5)


def test_roi_tolerates_a_rectangle_drawn_flush_to_the_edge():
    """World-coordinate arithmetic lands a hair over 1.0; that is not an error."""
    roi = _roi(x=0.5, width=0.5 + 5e-7)
    assert roi.x == 0.5


def test_slice_selection_carries_an_optional_roi():
    import models
    sel = models.SliceSelection(series_uid="SE1", roi=_roi())
    assert sel.roi.width == 0.3
    assert models.SliceSelection(series_uid="SE1").roi is None


# ---------- WindowLevel ----------

def _voi(**kw):
    import models
    return models.WindowLevel(**{"lower": 0.0, "upper": 1000.0, **kw})


def test_window_round_trips_its_bounds():
    import models
    voi = _voi(invert=True)
    parsed = models.WindowLevel.model_validate_json(voi.model_dump_json())
    assert (parsed.lower, parsed.upper, parsed.invert) == (0.0, 1000.0, True)


def test_window_rejects_a_zero_width():
    """Every pixel would map to one value; refuse rather than substitute a window."""
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        _voi(lower=500.0, upper=500.0)


def test_window_rejects_an_inverted_window():
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        _voi(lower=1000.0, upper=0.0)


def test_window_rejects_an_infinite_bound():
    """An infinity satisfies `upper > lower` and then blanks the slice.

    `(x - lower) / (upper - lower)` is 0.0 for every finite pixel, so the model
    would receive an all-black PNG with nothing logged and nothing raised.
    """
    import pydantic
    import pytest
    for kw in ({"upper": float("inf")}, {"lower": float("-inf")}):
        with pytest.raises(pydantic.ValidationError):
            _voi(**kw)


def test_window_rejects_a_nan_bound():
    """Held by the field rather than by comparison semantics.

    `nan > x` and `x > nan` are both False, so `_is_a_window` already refused
    these -- but only as a side effect. Pinned so the guarantee survives a
    rewrite of that comparison.
    """
    import pydantic
    import pytest
    for kw in ({"upper": float("nan")}, {"lower": float("nan")}):
        with pytest.raises(pydantic.ValidationError):
            _voi(**kw)


def test_window_rejects_a_width_that_overflows_to_infinity():
    """Finite bounds are not enough: the width is what the pixels are divided by."""
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        _voi(lower=-1e308, upper=1e308)


def test_window_accepts_a_window_far_outside_the_pixel_range():
    """A window wider than the data, or clear of it, is a real reader action.

    Clipping the viewport to black is deliberate and well defined for any finite
    window. Only a non-finite one is meaningless, so the bound stays on
    finiteness and not on the volume's value range -- which this model could not
    consult anyway, being validated before any DICOM is retrieved.
    """
    voi = _voi(lower=100_000.0, upper=200_000.0)
    assert (voi.lower, voi.upper) == (100_000.0, 200_000.0)


def test_window_from_a_json_payload_carrying_infinity_is_rejected():
    """The transport really can deliver this.

    `websocket.iter_json()` parses with `json.loads`, which accepts the
    non-standard `Infinity` literal, so the guard has to hold at the model rather
    than relying on JSON being unable to express it. A browser cannot send this
    (`JSON.stringify(Infinity)` is `null`), but `/ws/chat/` is proxied without
    authentication, so a hand-written frame is in scope.
    """
    import json

    import models
    import pydantic
    import pytest

    payload = json.loads('{"lower": 0, "upper": Infinity}')
    assert payload["upper"] == float("inf")  # json.loads accepted it
    with pytest.raises(pydantic.ValidationError):
        models.WindowLevel(**payload)


def test_slice_selection_carries_an_optional_voi():
    import models
    sel = models.SliceSelection(series_uid="SE1", voi=_voi())
    assert sel.voi.upper == 1000.0
    assert models.SliceSelection(series_uid="SE1").voi is None
