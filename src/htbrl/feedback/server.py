"""FastAPI feedback UI for pairwise preference labeling (Phase 8).

Routes:
- ``GET  /``                    -> redirect to /label
- ``GET  /label[?matrix=...]``   -> render two random snippets side-by-side
- ``POST /label``                -> record a preference + go to next pair
- ``GET  /stats``                -> JSON counts (snippets, preferences)
- ``GET  /health``               -> 200 OK heartbeat

The UI is intentionally minimal HTML+forms (no JS framework). One Python
process, one sqlite file, one user. Run via ``scripts/serve_feedback.py``.
"""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from htbrl.data.preference_dataset import PreferenceStore


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _render_label_page(
    store: PreferenceStore,
    matrix: str | None,
    msg: str = "",
) -> str:
    """Render the labeling page with two random snippets side-by-side."""
    try:
        left, right = store.random_pair(matrix=matrix)
    except RuntimeError as e:
        body = f"""
        <h1>htbrl feedback — labeling</h1>
        <p style="color:#a00">{_esc(str(e))}</p>
        <p>Add at least two snippets to <code>{_esc(str(store.path))}</code> first
        (use <code>scripts/serve_feedback.py --add-snippet ...</code> or insert via the
        Python API).</p>
        """
        return _wrap(body)

    n_p = store.n_preferences(exclude_discard=False)
    n_s = store.n_snippets()
    matrix_select = _matrix_select(matrix)
    msg_html = f'<p style="color:#070">{_esc(msg)}</p>' if msg else ""
    body = f"""
    <h1>htbrl feedback — pairwise preference labeling</h1>
    <p>Snippets in store: <b>{n_s}</b>. Preferences logged: <b>{n_p}</b>.</p>
    {matrix_select}
    {msg_html}
    <form method="post" action="/label" style="margin-top:1em">
      <input type="hidden" name="left_id"  value="{left.id}">
      <input type="hidden" name="right_id" value="{right.id}">
      <input type="hidden" name="matrix"   value="{_esc(matrix or '')}">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1em">
        <div style="border:1px solid #888;padding:1em">
          <h3>LEFT (id={left.id}, matrix={_esc(left.matrix)})</h3>
          <small>source: {_esc(left.source)}</small>
          <pre style="white-space:pre-wrap;background:#f4f4f4;padding:.5em">{_esc(left.content_text)}</pre>
        </div>
        <div style="border:1px solid #888;padding:1em">
          <h3>RIGHT (id={right.id}, matrix={_esc(right.matrix)})</h3>
          <small>source: {_esc(right.source)}</small>
          <pre style="white-space:pre-wrap;background:#f4f4f4;padding:.5em">{_esc(right.content_text)}</pre>
        </div>
      </div>
      <div style="margin-top:1em">
        <textarea name="notes" rows="2" cols="80" placeholder="optional notes (why?)"></textarea>
      </div>
      <div style="margin-top:.5em">
        <button name="label" value="left"    type="submit" style="background:#cef">left is better</button>
        <button name="label" value="right"   type="submit" style="background:#fec">right is better</button>
        <button name="label" value="tie"     type="submit">tie</button>
        <button name="label" value="discard" type="submit" style="background:#fdd">discard pair</button>
      </div>
    </form>
    """
    return _wrap(body)


def _matrix_select(current: str | None) -> str:
    options = []
    for m in (None, "enterprise", "mobile", "ics"):
        sel = " selected" if m == current else ""
        label = m if m is not None else "(any)"
        val = m if m is not None else ""
        options.append(f'<option value="{_esc(val)}"{sel}>{_esc(label)}</option>')
    opts = "\n".join(options)
    return f"""
    <form method="get" action="/label">
      <label>matrix: <select name="matrix" onchange="this.form.submit()">{opts}</select></label>
      <noscript> <button type="submit">apply</button></noscript>
    </form>
    """


def _wrap(body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>htbrl feedback</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:1em auto;padding:0 1em}}</style>
</head>
<body>{body}</body></html>"""


def create_app(db_path: Path | str = "data/preferences.db") -> FastAPI:
    """Build a FastAPI app bound to a sqlite preference store at ``db_path``."""
    store = PreferenceStore(db_path)
    app = FastAPI(title="htbrl feedback", version="0.1")
    app.state.store = store

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse("/label", status_code=303)

    @app.get("/label", response_class=HTMLResponse)
    def label_get(matrix: str | None = Query(default=None)) -> HTMLResponse:
        m = matrix or None
        return HTMLResponse(_render_label_page(app.state.store, m))

    @app.post("/label", response_class=HTMLResponse)
    def label_post(
        left_id: int = Form(...),
        right_id: int = Form(...),
        label: str = Form(...),
        notes: str = Form(""),
        matrix: str = Form(""),
    ) -> HTMLResponse:
        try:
            app.state.store.add_preference(
                left_id=left_id, right_id=right_id, label=label, notes=notes
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return HTMLResponse(
            _render_label_page(
                app.state.store,
                matrix or None,
                msg=f"recorded {label} for ({left_id} vs {right_id}).",
            )
        )

    @app.get("/stats")
    def stats() -> JSONResponse:
        s = app.state.store
        return JSONResponse({
            "snippets": {
                "total": s.n_snippets(),
                "enterprise": s.n_snippets("enterprise"),
                "mobile": s.n_snippets("mobile"),
                "ics": s.n_snippets("ics"),
            },
            "preferences": {
                "total": s.n_preferences(),
                "non_discard": s.n_preferences(exclude_discard=True),
            },
            "db_path": str(s.path),
        })

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    return app
