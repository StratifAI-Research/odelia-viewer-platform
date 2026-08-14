"""Tests for chat-middleware/image_cache.py — LRU image cache."""
from datetime import datetime


def _make_cached(uid="1.2.3", n_images=2):
    from image_cache import CachedSeries
    return CachedSeries(series_uid=uid, base64_images=[f"img{i}" for i in range(n_images)])


def test_cached_series_timestamps_are_tz_aware():
    from image_cache import CachedSeries
    s = CachedSeries(series_uid="u", base64_images=["i"])
    assert s.created_at.tzinfo is not None
    assert s.last_accessed.tzinfo is not None


def test_cache_get_sets_tz_aware_last_accessed():
    from image_cache import ImageCache
    c = ImageCache(max_entries=3)
    c.put("uid.1", _make_cached("uid.1"))
    out = c.get("uid.1")
    assert out.last_accessed.tzinfo is not None


def test_image_cache_empty_initially():
    from image_cache import ImageCache
    c = ImageCache(max_entries=3)
    assert c.size() == 0
    assert not c.has("anything")
    assert c.get("anything") is None


def test_put_then_get_returns_same_object():
    from image_cache import ImageCache
    c = ImageCache(max_entries=3)
    s = _make_cached("uid.1")
    c.put("uid.1", s)
    out = c.get("uid.1")
    assert out is s
    assert c.has("uid.1")
    assert c.size() == 1


def test_put_updates_existing_entry_in_place():
    from image_cache import ImageCache
    c = ImageCache(max_entries=3)
    c.put("uid.1", _make_cached("uid.1", n_images=2))
    s2 = _make_cached("uid.1", n_images=5)
    c.put("uid.1", s2)
    out = c.get("uid.1")
    assert out is s2
    assert c.size() == 1


def test_lru_evicts_oldest_when_full():
    from image_cache import ImageCache
    c = ImageCache(max_entries=2)
    c.put("uid.a", _make_cached("uid.a"))
    c.put("uid.b", _make_cached("uid.b"))
    c.put("uid.c", _make_cached("uid.c"))         # evicts uid.a
    assert not c.has("uid.a")
    assert c.has("uid.b")
    assert c.has("uid.c")
    assert c.size() == 2


def test_get_moves_entry_to_most_recently_used():
    """After get(uid.a), the LRU becomes uid.b — next put() should evict b, not a."""
    from image_cache import ImageCache
    c = ImageCache(max_entries=2)
    c.put("uid.a", _make_cached("uid.a"))
    c.put("uid.b", _make_cached("uid.b"))
    _ = c.get("uid.a")                            # uid.a -> MRU, uid.b -> LRU
    c.put("uid.c", _make_cached("uid.c"))         # evicts uid.b
    assert c.has("uid.a")
    assert not c.has("uid.b")
    assert c.has("uid.c")






def test_clear_returns_count_and_empties_cache():
    from image_cache import ImageCache
    c = ImageCache(max_entries=5)
    for i in range(3):
        c.put(f"uid.{i}", _make_cached(f"uid.{i}"))
    n = c.clear()
    assert n == 3
    assert c.size() == 0


def test_stats_reports_size_capacity_and_keys():
    from image_cache import ImageCache
    c = ImageCache(max_entries=5)
    c.put("uid.a", _make_cached("uid.a"))
    c.put("uid.b", _make_cached("uid.b"))
    st = c.stats()
    assert st["size"] == 2
    assert st["max_entries"] == 5
    assert sorted(st["series_uids"]) == ["uid.a", "uid.b"]


def test_evict_on_empty_cache_returns_none():
    from image_cache import ImageCache
    c = ImageCache()
    assert c._evict_lru() is None


def test_get_image_cache_returns_singleton():
    import image_cache
    image_cache.reset_image_cache(max_entries=10)
    a = image_cache.get_image_cache()
    b = image_cache.get_image_cache()
    assert a is b


def test_get_image_cache_warns_on_max_entries_mismatch(caplog):
    """Calling get_image_cache(max_entries=N) with an existing singleton at a different N
    must log a warning and return the existing one (semantics: max_entries is sticky)."""
    import image_cache
    image_cache.reset_image_cache(max_entries=10)
    with caplog.at_level("WARNING"):
        c = image_cache.get_image_cache(max_entries=99)
    assert c.max_entries == 10
    assert any("already initialized" in r.message for r in caplog.records)


def test_reset_image_cache_replaces_singleton():
    import image_cache
    first = image_cache.reset_image_cache(max_entries=5)
    second = image_cache.reset_image_cache(max_entries=20)
    assert first is not second
    assert second.max_entries == 20


def test_get_updates_last_accessed_timestamp():
    """M6: construct CachedSeries with an explicit OLD timestamp so the post-get now()
    is unambiguously greater. No sleep, no monkeypatch — deterministic against CI clock skew."""
    from datetime import datetime, timezone
    from image_cache import ImageCache, CachedSeries
    c = ImageCache(max_entries=2)
    ancient = datetime(2020, 1, 1, tzinfo=timezone.utc)
    s = CachedSeries(series_uid="uid.x", base64_images=["i"], last_accessed=ancient)
    c.put("uid.x", s)
    assert c._cache["uid.x"].last_accessed == ancient
    c.get("uid.x")
    assert c._cache["uid.x"].last_accessed > ancient


# ---------- make_cache_key ----------
#
# The key decides whether two messages share preprocessed pixels. Keying on the
# series alone — as this cache originally did — answers the second message in a
# conversation with the first message's slices as soon as they differ.

def test_make_cache_key_is_prefixed_with_the_series_uid():
    """Keys stay greppable in logs and in /debug/cache/stats."""
    from image_cache import make_cache_key
    key = make_cache_key("1.2.840.SE1", ("recipe", "5", "central", "60"))
    assert key.startswith("1.2.840.SE1#")


def test_make_cache_key_is_stable_for_the_same_recipe():
    from image_cache import make_cache_key
    a = make_cache_key("SE1", ("recipe", "5", "central", "60"))
    b = make_cache_key("SE1", ("recipe", "5", "central", "60"))
    assert a == b


def test_make_cache_key_differs_when_the_recipe_differs():
    from image_cache import make_cache_key
    a = make_cache_key("SE1", ("recipe", "5", "central", "60"))
    b = make_cache_key("SE1", ("recipe", "8", "central", "60"))
    assert a != b


def test_make_cache_key_differs_when_the_series_differs():
    from image_cache import make_cache_key
    a = make_cache_key("SE1", ("recipe", "5"))
    b = make_cache_key("SE2", ("recipe", "5"))
    assert a != b


def test_make_cache_key_is_order_sensitive():
    """Slice order is part of what was sent, so it must be part of the key."""
    from image_cache import make_cache_key
    a = make_cache_key("SE1", ("instances", "1.1", "1.2"))
    b = make_cache_key("SE1", ("instances", "1.2", "1.1"))
    assert a != b


def test_make_cache_key_stays_short_for_a_long_selection():
    """64 named instances must not produce an unloggable key."""
    from image_cache import make_cache_key
    key = make_cache_key("SE1", ("instances", *[f"1.2.840.113619.2.{i}" for i in range(64)]))
    assert len(key) < 80


def test_make_cache_key_separates_a_selection_from_a_recipe():
    """A selection of 5 named slices is not the same entry as 'the recipe says 5'."""
    from image_cache import make_cache_key
    a = make_cache_key("SE1", ("instances", "1.1", "1.2", "1.3", "1.4", "1.5"))
    b = make_cache_key("SE1", ("recipe", "5", "central", "60"))
    assert a != b


def test_stats_lists_entry_keys_and_the_series_behind_them():
    """One series can hold several entries — one per recipe — and stats says so."""
    from image_cache import ImageCache, make_cache_key
    c = ImageCache(max_entries=5)
    k1 = make_cache_key("SE1", ("instances", "1.1"))
    k2 = make_cache_key("SE1", ("instances", "1.2"))
    c.put(k1, _make_cached("SE1"))
    c.put(k2, _make_cached("SE1"))
    st = c.stats()
    assert st["size"] == 2
    assert sorted(st["keys"]) == sorted([k1, k2])
    assert st["series_uids"] == ["SE1"]
