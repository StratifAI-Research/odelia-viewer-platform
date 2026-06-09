"""Tests for chat-middleware/image_cache.py — LRU image cache."""
from datetime import datetime


def _make_cached(uid="1.2.3", n_images=2):
    from image_cache import CachedSeries
    return CachedSeries(series_uid=uid, base64_images=[f"img{i}" for i in range(n_images)])


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
    from datetime import datetime
    from image_cache import ImageCache, CachedSeries
    c = ImageCache(max_entries=2)
    ancient = datetime(2020, 1, 1)
    s = CachedSeries(series_uid="uid.x", base64_images=["i"], last_accessed=ancient)
    c.put("uid.x", s)
    assert c._cache["uid.x"].last_accessed == ancient
    c.get("uid.x")
    assert c._cache["uid.x"].last_accessed > ancient


