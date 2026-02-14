#!/bin/bash
# Pre-commit hook to ensure frontend build still works.
#
# Policy:
# - `static/dist/` is gitignored (rebuildable). We do NOT require committing dist.
# - If a repo chooses to track `static/dist/` (opt-in), we enforce it being in sync.

set -e

# Get the list of staged JS source files (excluding dist)
STAGED_JS=$(git diff --cached --name-only --diff-filter=ACM | grep -E '^static/.*\.js$' | grep -v '^static/dist/' || true)

if [ -z "$STAGED_JS" ]; then
    exit 0
fi

echo "JS source files changed, checking frontend build..."

# Rebuild dist
npm run build:static

if [ ! -f "static/dist/manifest.json" ]; then
    echo "ERROR: static/dist/manifest.json not found after build."
    exit 1
fi

# If dist files are tracked (opt-in), enforce they are in sync.
TRACKED_DIST_COUNT=$(git ls-files static/dist/ | wc -l | tr -d ' ')
if [ "$TRACKED_DIST_COUNT" != "0" ]; then
    DIST_CHANGED=$(git diff --name-only static/dist/ || true)
    if [ -n "$DIST_CHANGED" ]; then
        echo "ERROR: static/dist/ is out of sync with source files."
        echo "The following dist files need to be updated:"
        echo "$DIST_CHANGED"
        echo ""
        echo "Please run 'npm run build:static' and stage the dist files."
        exit 1
    fi
fi

echo "frontend build looks OK."
