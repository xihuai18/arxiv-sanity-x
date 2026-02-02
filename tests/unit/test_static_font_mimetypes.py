from __future__ import annotations


def test_static_mathjax_font_served_with_font_mime(client):
    resp = client.get("/static/lib/es5/output/chtml/fonts/woff-v2/MathJax_Calligraphic-Regular.woff")
    assert resp.status_code == 200
    assert (resp.headers.get("Content-Type") or "").startswith("font/woff")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
