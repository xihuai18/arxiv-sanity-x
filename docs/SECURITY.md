# Security Notes

This project is designed primarily for **personal / intranet** usage.

## Deployment Model

- Login is username-only (no password). Do not expose this service directly to the public internet.
- Recommended: put it behind VPN / reverse-proxy auth / SSO.
- See also: `README.md` → "Deployment & Security Notes".

## API Docs (Swagger)

- Swagger UI is disabled by default.
- Enable only in trusted environments: `ARXIV_SANITY_ENABLE_SWAGGER=true` (serves under `/apidocs/`).

## Session Secret Key

- Recommended: set `ARXIV_SANITY_SECRET_KEY` (strong random string).
- `secret_key.txt` is a local fallback for convenience and is ignored by git. Treat it as a secret:
  - Do not commit it
  - Rotate it if you suspect it was leaked

## CSRF

- State-changing endpoints require CSRF (header `X-CSRF-Token`) for browser sessions.
- For internal service-to-service calls (e.g. `tools/send_emails.py`), use `ARXIV_SANITY_RECO_API_KEY` and send:
  - `X-ARXIV-SANITY-API-KEY: ...` (or `Authorization: Bearer ...`)

## Health/Metrics Exposure

- `GET /health` is safe for basic readiness checks; still avoid exposing internal status to untrusted networks.
- `GET /metrics` should be protected with `ARXIV_SANITY_METRICS_KEY` if reachable from untrusted networks.

## Uploaded PDFs (Experimental)

- Upload PIDs are treated as private per-user resources.
- Avoid sharing upload PIDs publicly.
- Consider tightening `ARXIV_SANITY_MAX_CONTENT_LENGTH` (default 50MB) and monitoring `data/uploads/` growth.

## Frontend Asset CDN (Optional)

The app can load third-party frontend libraries from a public CDN with a local fallback.

- Enable/disable: `ARXIV_SANITY_ASSET_CDN_ENABLED=true|false`
- CDN base: `ARXIV_SANITY_ASSET_NPM_CDN_BASE=https://cdn.jsdelivr.net/npm`

If your deployment requires strict offline / no-exfil policies, disable the CDN and serve all assets locally.

## Error Reporting (Sentry)

- Disabled by default and requires explicit enable + DSN.
- Still review what you send to Sentry for compliance (paper content, user identifiers, etc.).
