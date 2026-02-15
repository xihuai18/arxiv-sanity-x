#!/usr/bin/env bash
# =============================================================================
# Sync script: Sync code to open source repository, strictly excluding private files and content not intended for open source
# =============================================================================
#
# Usage:
#   ./sync_to_opensource.sh            # Actual sync (--delete enabled by default)
#   ./sync_to_opensource.sh --dry-run  # Preview changes only, no actual writes
#   ./sync_to_opensource.sh --no-delete # Don't delete extra files in target
#   ./sync_to_opensource.sh --purge-excluded # Also delete excluded files in target (more destructive)
#   ./sync_to_opensource.sh --verbose  # Show detailed sync information
#
# Environment variables:
#   SOURCE_DIR  Source directory (default: parent directory of script location)
#   TARGET_DIR  Target directory (default: ../arxiv-sanity-x)
#   DUMMY_SUBMODULE_URL  Dummy/public URL for the data backup submodule in open source mirror
#   DUMMY_SUBMODULE_NAME Submodule name to rewrite (default: data-repo)
#   DUMMY_SUBMODULE_PATH Submodule path to rewrite (default: data-repo)
#
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Path settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SOURCE_DIR:-$(dirname "$SCRIPT_DIR")}"
TARGET_DIR="${TARGET_DIR:-$(dirname "$SOURCE_DIR")/arxiv-sanity-x}"

# Submodule URL sanitization (avoid leaking private SSH URLs into open source mirror)
DUMMY_SUBMODULE_URL="${DUMMY_SUBMODULE_URL:-https://github.com/arxiv-sanity/arxiv-sanity-data-dummy.git}"
DUMMY_SUBMODULE_NAME="${DUMMY_SUBMODULE_NAME:-data-repo}"
DUMMY_SUBMODULE_PATH="${DUMMY_SUBMODULE_PATH:-data-repo}"

# Default parameter values
DRY_RUN=0
DELETE=1
PURGE_EXCLUDED=0
VERBOSE=0

usage() {
    cat <<'EOF'
Usage:
    ./sync_to_opensource.sh [options]

Options:
    --dry-run    Preview changes only, no actual writes
    --no-delete  Don't delete extra files in target
    --purge-excluded  Also delete excluded files in target (more destructive)
    --verbose    Show detailed sync information
    -h, --help   Show help

Description:
    - Uses rsync to sync code to open source directory
    - Auto-excludes: private files, data files, cache, model weights, etc.
    - --delete enabled by default: extra files in target will be deleted
    - Security check after sync to ensure no privacy leaks
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-delete)
            DELETE=0
            shift
            ;;
        --purge-excluded)
            PURGE_EXCLUDED=1
            shift
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            usage >&2
            exit 2
            ;;
    esac
done

# Pre-checks
if [[ ! -d "$SOURCE_DIR" ]]; then
    log_error "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

if [[ ! -d "$TARGET_DIR" ]]; then
    log_error "Target directory does not exist: $TARGET_DIR"
    log_info "Please create target directory or clone open source repository first"
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    log_error "rsync not found, please install it first"
    exit 1
fi

# Show configuration
echo "=============================================="
echo "  arxiv-sanity-X Open Source Sync Script"
echo "=============================================="
log_info "Source:   $SOURCE_DIR"
log_info "Target:   $TARGET_DIR"
log_info "dry-run:  $([[ $DRY_RUN -eq 1 ]] && echo 'yes' || echo 'no')"
log_info "delete:   $([[ $DELETE -eq 1 ]] && echo 'yes' || echo 'no')"
log_info "purge:    $([[ $PURGE_EXCLUDED -eq 1 ]] && echo 'yes' || echo 'no')"
log_info "submod:   ${DUMMY_SUBMODULE_NAME} -> ${DUMMY_SUBMODULE_URL}"
echo ""

# =============================================================================
# rsync exclude rules (organized by category)
# =============================================================================
RSYNC_OPTS=(
    -a                      # Archive mode (preserve permissions, timestamps, etc.)
    --human-readable        # Human-readable output
    --stats                 # Show statistics

    # Protect target directory's .git (not deleted by --delete)
    --filter='P .git/'

    # 1. Version control and IDE config (exclude from source)
    --exclude='.git/'
    # Exclude .git *files* too (submodules may store gitdir pointers in a .git file)
    --exclude='.git'
    --exclude='**/.git'
    --exclude='.vscode/'
    --exclude='.idea/'

    # 2. Python cache and compiled files
    --exclude='__pycache__/'
    --exclude='**/__pycache__/'
    --exclude='.pytest_cache/'
    --exclude='.mypy_cache/'
    --exclude='.ruff_cache/'
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='*.pyd'
    --exclude='.coverage'
    --exclude='htmlcov/'
    --exclude='.venv/'
    --exclude='venv/'
    --exclude='*.egg-info/'

    # 3. Node.js related
    --exclude='node_modules/'

    # 4. Data directory
    --exclude='data/'
    --exclude='data-repo/'

    # 5. Model weights and large files
    --exclude='*.safetensors'
    --exclude='*.pt'
    --exclude='*.pth'
    --exclude='*.bin'
    --exclude='*.onnx'
    --exclude='*.h5'

    # 6. Database and serialization files
    --exclude='*.db'
    --exclude='*.db-shm'
    --exclude='*.db-wal'
    --exclude='*.sqlite'
    --exclude='*.sqlite3'
    --exclude='*.p'
    --exclude='*.pkl'
    --exclude='*.pickle'

    # 7. Privacy and key files (most important!)
    --include='.env.example'    # Example config file needs to be synced
    --exclude='.env'
    --exclude='.env.*'
    --exclude='*.env'
    --exclude='secret_key.txt'
    --exclude='sendgrid_api_key.txt'
    --exclude='vars.py'
    --exclude='config/llm.yml'
    --exclude='*.pem'
    --exclude='*.key'
    --exclude='*.crt'
    --exclude='id_rsa'
    --exclude='id_rsa.pub'
    --exclude='*.secret'
    --exclude='*_secret*'
    --exclude='*credentials*'

    # 8. Logs and temporary files
    --exclude='*.log'
    --exclude='nohup.out'
    --exclude='*.tmp'
    --exclude='*.temp'
    --exclude='tmp_*'

    # 9. AI/IDE agent local config (do not publish)
    --exclude='.claude/'
    --exclude='**/.claude/'
    --exclude='.factory/'
    --exclude='**/.factory/'
    --exclude='.skills/'
    --exclude='**/.skills/'
    --exclude='*.local.json'
    --exclude='*.local.yml'
    --exclude='*.local.yaml'
    --exclude='*.local.toml'

    # 10. System files
    --exclude='.DS_Store'
    --exclude='Thumbs.db'
    --exclude='*.swp'
    --exclude='*.swo'
    --exclude='*~'

    # 11. Build output
    --exclude='static/dist/'

    # 12. Local-only files
    --exclude='AGENTS.md'

    # 13. PDF and other large media files
    --exclude='*.pdf'
)

# Add --delete option
if [[ $DELETE -eq 1 ]]; then
    RSYNC_OPTS+=(--delete)
    if [[ $PURGE_EXCLUDED -eq 1 ]]; then
        RSYNC_OPTS+=(--delete-excluded)
    fi
fi

# Add --dry-run option
if [[ $DRY_RUN -eq 1 ]]; then
    RSYNC_OPTS+=(--dry-run)
    log_warn "DRY-RUN mode: no files will be written"
fi

# Add verbose option
if [[ $VERBOSE -eq 1 ]]; then
    RSYNC_OPTS+=(-v --progress)
fi

# =============================================================================
# Execute sync
# =============================================================================
log_info "Starting sync..."
echo ""

rsync "${RSYNC_OPTS[@]}" "$SOURCE_DIR/" "$TARGET_DIR/"

echo ""

# =============================================================================
# Post-sync sanitization (only in non dry-run mode)
# =============================================================================
sanitize_target() {
    # 1) Rewrite submodule URL to a dummy/public link
    local gitmodules="$TARGET_DIR/.gitmodules"
    if [[ -f "$gitmodules" ]]; then
        log_info "Sanitizing .gitmodules (rewrite submodule URL)..."
        if command -v git >/dev/null 2>&1; then
            git config -f "$gitmodules" "submodule.${DUMMY_SUBMODULE_NAME}.path" "$DUMMY_SUBMODULE_PATH" || true
            git config -f "$gitmodules" "submodule.${DUMMY_SUBMODULE_NAME}.url" "$DUMMY_SUBMODULE_URL" || true
        else
            log_warn "git not found; skipping .gitmodules rewrite"
        fi
    fi

    # 2) Remove any nested .git files that could have been copied from submodules
    #    (rsync excludes .git/ directories, but submodules may contain a .git *file*)
    local nested_git_files
    nested_git_files=$(find "$TARGET_DIR" -type f -name .git 2>/dev/null | head -20 || true)
    if [[ -n "$nested_git_files" ]]; then
        log_warn "Removing nested .git files in target (copied from submodules):"
        echo "$nested_git_files" | while read -r f; do
            echo "  - $f"
            rm -f "$f" || true
        done
    fi

    # 3) Remove forbidden directories that may already exist in the target repo.
    #    rsync excludes prevent copying, but won't delete pre-existing excluded dirs
    #    unless --delete-excluded is used.
    local forbidden
    for forbidden in ".skills" ".claude" ".factory" "node_modules" "__pycache__" ".pytest_cache" "htmlcov" "data"; do
        if [[ -e "$TARGET_DIR/$forbidden" ]]; then
            log_warn "Removing forbidden path in target: $TARGET_DIR/$forbidden"
            rm -rf "$TARGET_DIR/$forbidden" || true
        fi
    done
}

# =============================================================================
# Post-sync security check (only in non dry-run mode)
# =============================================================================
if [[ $DRY_RUN -eq 1 ]]; then
    log_warn "Skipping security check in DRY-RUN mode (target directory not actually modified)"
else
    sanitize_target
    log_info "Running security check..."

    # Files forbidden in target repository
    FORBIDDEN_FILES=(
        'vars.py'
        'llm.yml'
        'secret_key.txt'
        'sendgrid_api_key.txt'
        'AGENTS.md'
        '.coverage'
    )

    # Forbidden file patterns
    FORBIDDEN_PATTERNS=(
        '.env'
        '.env.*'
        '*.pem'
        '*.key'
        '*.secret'
        '*.local.json'
        '*.local.yml'
        '*.local.yaml'
        '*.local.toml'
        '.envrc'
        '.netrc'
        '.npmrc'
        '.pypirc'
    )

    # Forbidden directories
    FORBIDDEN_DIRS=(
        '__pycache__'
        'node_modules'
        '.venv'
        'venv'
        '.pytest_cache'
        'htmlcov'
        'data'
        '.claude'
        '.factory'
        '.skills'
    )

    FOUND_ISSUES=0

    # Check forbidden files
    for name in "${FORBIDDEN_FILES[@]}"; do
        found=$(find "$TARGET_DIR" -type f -name "$name" 2>/dev/null | head -5)
        if [[ -n "$found" ]]; then
            log_error "Found forbidden file: $name"
            echo "$found" | while read -r f; do echo "  - $f"; done
            FOUND_ISSUES=1
        fi
    done

    # Check forbidden patterns (exclude allowed files like .env.example)
    for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
        found=$(find "$TARGET_DIR" -type f -name "$pattern" ! -name ".env.example" 2>/dev/null | head -5)
        if [[ -n "$found" ]]; then
            log_error "Found forbidden pattern file: $pattern"
            echo "$found" | while read -r f; do echo "  - $f"; done
            FOUND_ISSUES=1
        fi
    done

    # Check forbidden directories
    for dir in "${FORBIDDEN_DIRS[@]}"; do
        found=$(find "$TARGET_DIR" -type d -name "$dir" 2>/dev/null | head -3)
        if [[ -n "$found" ]]; then
            log_error "Found forbidden directory: $dir"
            echo "$found" | while read -r f; do echo "  - $f"; done
            FOUND_ISSUES=1
        fi
    done

    # Check nested .git files (submodule metadata leakage)
    nested_git_files=$(find "$TARGET_DIR" -type f -name .git 2>/dev/null | head -5 || true)
    if [[ -n "$nested_git_files" ]]; then
        log_error "Found nested .git files (submodule metadata leakage):"
        echo "$nested_git_files" | while read -r f; do echo "  - $f"; done
        FOUND_ISSUES=1
    fi

    # Check .gitmodules does not contain private SSH URLs
    if [[ -f "$TARGET_DIR/.gitmodules" ]]; then
        if grep -n "git@github.com:" "$TARGET_DIR/.gitmodules" >/dev/null 2>&1; then
            log_error "Found private SSH URL in .gitmodules (git@github.com:...)"
            grep -n "git@github.com:" "$TARGET_DIR/.gitmodules" | head -5 || true
            FOUND_ISSUES=1
        fi
        if grep -n "xihuai18/arxiv-sanity-data" "$TARGET_DIR/.gitmodules" >/dev/null 2>&1; then
            log_error "Found private data repo reference in .gitmodules"
            grep -n "xihuai18/arxiv-sanity-data" "$TARGET_DIR/.gitmodules" | head -5 || true
            FOUND_ISSUES=1
        fi
    fi

    # Check database files
    db_files=$(find "$TARGET_DIR" -type f \( -name '*.db' -o -name '*.sqlite' -o -name '*.sqlite3' \) 2>/dev/null | head -5)
    if [[ -n "$db_files" ]]; then
        log_error "Found database files:"
        echo "$db_files" | while read -r f; do echo "  - $f"; done
        FOUND_ISSUES=1
    fi

    # Check pickle files
    pickle_files=$(find "$TARGET_DIR" -type f \( -name '*.p' -o -name '*.pkl' -o -name '*.pickle' \) 2>/dev/null | head -5)
    if [[ -n "$pickle_files" ]]; then
        log_error "Found serialization files:"
        echo "$pickle_files" | while read -r f; do echo "  - $f"; done
        FOUND_ISSUES=1
    fi

    # Check model weight files
    model_files=$(find "$TARGET_DIR" -type f \( -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' \) 2>/dev/null | head -5)
    if [[ -n "$model_files" ]]; then
        log_error "Found model weight files:"
        echo "$model_files" | while read -r f; do echo "  - $f"; done
        FOUND_ISSUES=1
    fi

    # Check for private IP addresses / local user paths in content (best-effort)
    if command -v grep >/dev/null 2>&1; then
        ip_hits=$(grep -RInE "\\b172\\.16\\.([0-9]{1,3}\\.)[0-9]{1,3}\\b|\\b192\\.168\\.([0-9]{1,3}\\.)[0-9]{1,3}\\b|\\b10\\.([0-9]{1,3}\\.){2}[0-9]{1,3}\\b" \
            "$TARGET_DIR" \
            --exclude-dir=".git" \
            --exclude-dir="node_modules" \
            --exclude-dir="static/dist" \
            --exclude-dir="data" \
            --exclude="package-lock.json" 2>/dev/null | head -5 || true)
        if [[ -n "$ip_hits" ]]; then
            log_error "Found private IP-like patterns in target content (possible privacy leak):"
            echo "$ip_hits"
            FOUND_ISSUES=1
        fi

	        user_hits=$(grep -RIn "$TARGET_DIR" \
	            --exclude-dir=".git" \
	            --exclude-dir="node_modules" \
	            --exclude-dir="static/dist" \
	            --exclude-dir="data" \
	            --exclude="sync_to_opensource.sh" \
	            -e "/xhwang/" -e "\\bxhwang\\b" 2>/dev/null | head -5 || true)
	        if [[ -n "$user_hits" ]]; then
	            log_error "Found local username/path hints in target content (possible privacy leak):"
	            echo "$user_hits"
	            FOUND_ISSUES=1
	        fi
	    fi

    # Check for unexpectedly large files (avoid publishing big artifacts)
    large_files=$(find "$TARGET_DIR" -type f -size +20M -not -path "*/.git/*" 2>/dev/null | head -5 || true)
    if [[ -n "$large_files" ]]; then
        log_error "Found unexpectedly large files (>20MB) in target:"
        echo "$large_files" | while read -r f; do echo "  - $f"; done
        FOUND_ISSUES=1
    fi

    echo ""

    if [[ $FOUND_ISSUES -eq 1 ]]; then
        log_error "Security check failed! Please fix exclude rules and re-sync."
        exit 3
    fi

    log_ok "Security check passed!"
fi

# =============================================================================
# Completion message
# =============================================================================
echo ""
echo "=============================================="
if [[ $DRY_RUN -eq 1 ]]; then
    log_warn "This was DRY-RUN mode, no files were written"
    log_info "Remove --dry-run argument to perform actual sync"
else
    log_ok "Sync completed!"
fi
echo "=============================================="
echo ""
log_info "Next steps:"
echo "  1. cd $TARGET_DIR"
echo "  2. git status"
echo "  3. git diff"
echo "  4. git add -A && git commit -m 'sync from private repo'"
echo "  5. git push"
echo ""
