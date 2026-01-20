#!/bin/bash
# Pre-commit hook to ensure static/dist is in sync with source files
# This script checks if any JS source files have been modified and ensures dist is rebuilt

set -e

# Get the list of staged JS source files (excluding dist)
STAGED_JS=$(git diff --cached --name-only --diff-filter=ACM | grep -E '^static/.*\.js$' | grep -v '^static/dist/' || true)

if [ -z "$STAGED_JS" ]; then
    exit 0
fi

echo "JS source files changed, checking dist sync..."

# Rebuild dist
npm run build:static

# Check if dist files changed after rebuild
DIST_CHANGED=$(git diff --name-only static/dist/ || true)

if [ -n "$DIST_CHANGED" ]; then
    echo "ERROR: static/dist/ is out of sync with source files."
    echo "The following dist files need to be updated:"
    echo "$DIST_CHANGED"
    echo ""
    echo "Please run 'npm run build:static' and stage the dist files."
    exit 1
fi

echo "dist files are in sync."
