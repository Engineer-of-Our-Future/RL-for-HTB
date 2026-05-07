"""Tests for the preference store + FastAPI feedback server."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from htbrl.data.preference_dataset import PreferenceStore
from htbrl.feedback.server import create_app


# ---- PreferenceStore ---------------------------------------------------------


def test_store_creates_schema(tmp_path: Path):
    p = tmp_path / "prefs.db"
    with PreferenceStore(p) as store:
        assert store.n_snippets() == 0
        assert store.n_preferences() == 0


def test_add_snippet_returns_increasing_ids(tmp_path: Path):
    with PreferenceStore(tmp_path / "p.db") as store:
        a = store.add_snippet("enterprise", "obs A")
        b = store.add_snippet("enterprise", "obs B")
        assert b == a + 1
        assert store.n_snippets() == 2


def test_add_snippet_rejects_unknown_matrix(tmp_path: Path):
    with PreferenceStore(tmp_path / "p.db") as store:
        with pytest.raises(ValueError, match="invalid matrix"):
            store.add_snippet("windows", "x")


def test_random_pair_requires_two_snippets(tmp_path: Path):
    with PreferenceStore(tmp_path / "p.db") as store:
        store.add_snippet("enterprise", "only one")
        with pytest.raises(RuntimeError, match="at least 2"):
            store.random_pair()


def test_random_pair_returns_distinct_ids(tmp_path: Path):
    with PreferenceStore(tmp_path / "p.db") as store:
        for i in range(5):
            store.add_snippet("enterprise", f"obs {i}")
        a, b = store.random_pair(matrix="enterprise")
        assert a.id != b.id


def test_add_preference_records_label(tmp_path: Path):
    with PreferenceStore(tmp_path / "p.db") as store:
        a = store.add_snippet("enterprise", "x")
        b = store.add_snippet("enterprise", "y")
        store.add_preference(a, b, "left")
        assert store.n_preferences() == 1


def test_add_preference_rejects_invalid_label(tmp_path: Path):
    with PreferenceStore(tmp_path / "p.db") as store:
        a = store.add_snippet("enterprise", "x")
        b = store.add_snippet("enterprise", "y")
        with pytest.raises(ValueError, match="not in"):
            store.add_preference(a, b, "banana")


def test_add_preference_rejects_self_pair(tmp_path: Path):
    with PreferenceStore(tmp_path / "p.db") as store:
        a = store.add_snippet("enterprise", "x")
        with pytest.raises(ValueError, match="must be different"):
            store.add_preference(a, a, "left")


def test_iter_training_pairs_orients_by_label(tmp_path: Path):
    """Whichever side was preferred ends up first in the (preferred, other) tuple."""
    with PreferenceStore(tmp_path / "p.db") as store:
        s1 = store.add_snippet("enterprise", "ALPHA")
        s2 = store.add_snippet("enterprise", "BETA")
        store.add_preference(s1, s2, "left")  # ALPHA is preferred
        store.add_preference(s1, s2, "right")  # BETA is preferred

        pairs = list(store.iter_training_pairs())
        assert pairs[0][0].content_text == "ALPHA"   # preferred
        assert pairs[0][1].content_text == "BETA"    # other
        assert pairs[1][0].content_text == "BETA"    # preferred (right)
        assert pairs[1][1].content_text == "ALPHA"


def test_iter_training_pairs_skips_discard(tmp_path: Path):
    with PreferenceStore(tmp_path / "p.db") as store:
        s1 = store.add_snippet("enterprise", "x")
        s2 = store.add_snippet("enterprise", "y")
        store.add_preference(s1, s2, "discard")
        assert list(store.iter_training_pairs()) == []


# ---- FastAPI server ----------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    db = tmp_path / "ui.db"
    app = create_app(db)
    with TestClient(app) as c:
        yield c, app
    app.state.store.close()


def test_health_endpoint(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_root_redirects_to_label(client):
    c, _ = client
    r = c.get("/", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/label"


def test_label_page_shows_warning_when_empty(client):
    c, _ = client
    r = c.get("/label")
    assert r.status_code == 200
    assert "at least 2 snippets" in r.text


def test_label_page_renders_pair(client):
    c, app = client
    app.state.store.add_snippet("enterprise", "ALPHA-content")
    app.state.store.add_snippet("enterprise", "BETA-content")
    r = c.get("/label")
    assert r.status_code == 200
    assert "ALPHA-content" in r.text or "BETA-content" in r.text


def test_label_post_records_preference(client):
    c, app = client
    a = app.state.store.add_snippet("enterprise", "x")
    b = app.state.store.add_snippet("enterprise", "y")
    r = c.post(
        "/label",
        data={"left_id": a, "right_id": b, "label": "left", "notes": "test", "matrix": ""},
    )
    assert r.status_code == 200
    assert app.state.store.n_preferences() == 1


def test_label_post_rejects_invalid_label(client):
    c, app = client
    a = app.state.store.add_snippet("enterprise", "x")
    b = app.state.store.add_snippet("enterprise", "y")
    r = c.post(
        "/label",
        data={"left_id": a, "right_id": b, "label": "banana", "notes": "", "matrix": ""},
    )
    assert r.status_code == 400


def test_stats_endpoint(client):
    c, app = client
    app.state.store.add_snippet("enterprise", "x")
    app.state.store.add_snippet("ics", "y")
    r = c.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["snippets"]["total"] == 2
    assert body["snippets"]["enterprise"] == 1
    assert body["snippets"]["ics"] == 1
