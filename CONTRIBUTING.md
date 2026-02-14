# Contributing

Thanks for considering a contribution!

## Development setup

- Python 3.10+ recommended
- (Optional) `conda activate sanity`
- Install:
  - `pip install -e ".[dev]"`

## Run tests

Recommended (fast):

```bash
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit tests/integration -q -k "not daemon"
```

## Frontend build

```bash
npm ci
npm run build:static
```

## Security / privacy

Do not commit:

- `.env*`, API keys, private URLs, `secret_key.txt`
- `data/` (DBs, caches, uploads, summaries)
- `.venv/`, `node_modules/`, `static/dist/`

If you're maintaining a private fork with a public mirror, see `docs/OPEN_SOURCE.md`.
