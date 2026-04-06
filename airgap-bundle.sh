#!/usr/bin/env bash
# ============================================================================
# SwiftLLM Air-Gap Bundle Creator
# ============================================================================
#
# Run this script on a CONNECTED machine to create a self-contained archive
# that can be transferred to an air-gapped host and installed offline.
#
# Usage:
#   ./airgap-bundle.sh                        # Bundle with defaults
#   ./airgap-bundle.sh --model org/repo:f.gguf  # Include a model
#   ./airgap-bundle.sh --cpu                  # CPU-only wheels
#   ./airgap-bundle.sh -o /tmp/bundle.tar.gz  # Custom output path
#
# The resulting archive contains:
#   swiftllm/              – source tree (for maturin build)
#   wheels/                – pip wheels for all Python deps
#   rust/                  – standalone Rust installer (rustup-init)
#   models/                – (optional) pre-downloaded models
#
# On the air-gapped host, extract and run:
#   tar xzf swiftllm-airgap-bundle.tar.gz
#   cd swiftllm-airgap-bundle
#   ./install.sh --airgap
# ============================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()    { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
step()    { echo -e "\n${BOLD}${CYAN}=> $1${NC}"; }

# ----------------------------
# Defaults
# ----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR=""
OUTPUT=""
MODELS=()
CPU_ONLY=false
PLATFORM=""

# ----------------------------
# Parse arguments
# ----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|-m)     MODELS+=("$2"); shift 2 ;;
        --cpu)          CPU_ONLY=true; shift ;;
        -o|--output)    OUTPUT="$2"; shift 2 ;;
        --platform)     PLATFORM="$2"; shift 2 ;;
        -h|--help)
            echo "SwiftLLM Air-Gap Bundle Creator"
            echo ""
            echo "Usage: ./airgap-bundle.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model, -m MODEL   Include a model (repeatable)"
            echo "  --cpu               Download CPU-only wheels"
            echo "  --platform PLAT     pip platform tag (e.g. manylinux2014_x86_64)"
            echo "  -o, --output PATH   Output archive path"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *) fail "Unknown option: $1" ;;
    esac
done

# ----------------------------
# Detect Python
# ----------------------------
PYTHON=""
for py in python3 python; do
    if command -v "$py" &>/dev/null; then
        PY_MAJOR=$("$py" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        PY_MINOR=$("$py" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [[ "$PY_MAJOR" -ge 3 ]] && [[ "$PY_MINOR" -ge 8 ]]; then
            PYTHON="$py"
            break
        fi
    fi
done
[[ -z "$PYTHON" ]] && fail "Python 3.8+ required"
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# ----------------------------
# Set up bundle directory
# ----------------------------
BUNDLE_DIR=$(mktemp -d)
BUNDLE_NAME="swiftllm-airgap-bundle"
DEST="$BUNDLE_DIR/$BUNDLE_NAME"
mkdir -p "$DEST"/{wheels,rust,models}

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$SCRIPT_DIR/$BUNDLE_NAME.tar.gz"
fi

step "Creating air-gap bundle"
info "Temporary workspace: $BUNDLE_DIR"
info "Output: $OUTPUT"

# ----------------------------
# 1. Copy source tree (excluding target/ and venv/)
# ----------------------------
step "Copying source tree..."
rsync -a --exclude='target/' --exclude='venv/' --exclude='.venv/' \
    --exclude='*.pyc' --exclude='__pycache__/' --exclude='.git/' \
    "$SCRIPT_DIR/" "$DEST/swiftllm/"
success "Source tree copied"

# ----------------------------
# 2. Download pip wheels for all dependencies
# ----------------------------
step "Downloading Python wheels..."

PLATFORM_FLAG=""
if [[ -n "$PLATFORM" ]]; then
    PLATFORM_FLAG="--platform $PLATFORM --only-binary=:all:"
fi

# Core build tool
$PYTHON -m pip download maturin -d "$DEST/wheels" $PLATFORM_FLAG 2>&1 | tail -3
success "maturin wheels downloaded"

# Runtime dependencies from pyproject.toml
$PYTHON -m pip download \
    "numpy>=1.20" "transformers>=4.30" "torch>=2.0" "safetensors>=0.3" \
    "tokenizers>=0.13" "tqdm>=4.60" "requests>=2.25" "aiohttp>=3.8" \
    "pydantic>=2.0" "huggingface-hub>=0.14" \
    -d "$DEST/wheels" $PLATFORM_FLAG 2>&1 | tail -5
success "Runtime dependency wheels downloaded"

# llama-cpp-python
if $CPU_ONLY; then
    $PYTHON -m pip download llama-cpp-python -d "$DEST/wheels" $PLATFORM_FLAG 2>&1 | tail -3
else
    $PYTHON -m pip download llama-cpp-python -d "$DEST/wheels" $PLATFORM_FLAG 2>&1 | tail -3
fi
success "llama-cpp-python wheel downloaded"

# pip itself (for bootstrapping)
$PYTHON -m pip download pip setuptools wheel -d "$DEST/wheels" $PLATFORM_FLAG 2>&1 | tail -3
success "pip/setuptools/wheel downloaded"

WHEEL_COUNT=$(ls "$DEST/wheels/" | wc -l | tr -d ' ')
WHEEL_SIZE=$(du -sh "$DEST/wheels/" | cut -f1)
info "$WHEEL_COUNT wheel(s), $WHEEL_SIZE total"

# ----------------------------
# 3. Download Rust standalone installer
# ----------------------------
step "Downloading Rust installer..."

# Detect target triple
ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$OS" in
    linux)  RUST_TARGET="${ARCH}-unknown-linux-gnu" ;;
    darwin) RUST_TARGET="${ARCH}-apple-darwin" ;;
    *)      warn "Unsupported OS ($OS). Skipping Rust installer." ; RUST_TARGET="" ;;
esac

if [[ -n "$RUST_TARGET" ]]; then
    RUSTUP_URL="https://static.rust-lang.org/rustup/dist/${RUST_TARGET}/rustup-init"
    if curl -sSfL "$RUSTUP_URL" -o "$DEST/rust/rustup-init"; then
        chmod +x "$DEST/rust/rustup-init"
        success "Downloaded rustup-init for $RUST_TARGET"
    else
        warn "Failed to download rustup-init. Rust must be pre-installed on the target."
    fi
fi

# ----------------------------
# 4. (Optional) Download models
# ----------------------------
if [[ ${#MODELS[@]} -gt 0 ]]; then
    step "Downloading models..."
    for model in "${MODELS[@]}"; do
        info "Downloading: $model"
        $PYTHON -c "
from swiftllm.model_resolver import resolve_model
import shutil, os
path = resolve_model('$model')
dest = os.path.join('$DEST/models', os.path.basename(path))
if os.path.isfile(path):
    shutil.copy2(path, dest)
elif os.path.isdir(path):
    shutil.copytree(path, dest)
print(f'Saved to: {dest}')
" 2>&1
    done
    MODEL_SIZE=$(du -sh "$DEST/models/" | cut -f1)
    success "Models downloaded ($MODEL_SIZE)"
fi

# ----------------------------
# 5. Create the archive
# ----------------------------
step "Creating archive..."
(cd "$BUNDLE_DIR" && tar czf "$OUTPUT" "$BUNDLE_NAME")

ARCHIVE_SIZE=$(du -sh "$OUTPUT" | cut -f1)
success "Bundle created: $OUTPUT ($ARCHIVE_SIZE)"

# Cleanup
rm -rf "$BUNDLE_DIR"

# ----------------------------
# Summary
# ----------------------------
echo ""
echo -e "${BOLD}${CYAN}============================================${NC}"
echo -e "${BOLD}${GREEN}  Air-gap bundle ready!${NC}"
echo -e "${BOLD}${CYAN}============================================${NC}"
echo ""
echo -e "  ${BOLD}Archive:${NC} $OUTPUT ($ARCHIVE_SIZE)"
echo -e "  ${BOLD}Contents:${NC}"
echo "    swiftllm/    Source code"
echo "    wheels/      $WHEEL_COUNT Python wheels"
if [[ -n "$RUST_TARGET" ]]; then
echo "    rust/        rustup-init ($RUST_TARGET)"
fi
if [[ ${#MODELS[@]} -gt 0 ]]; then
echo "    models/      ${#MODELS[@]} model(s)"
fi
echo ""
echo -e "  ${BOLD}On the air-gapped host:${NC}"
echo "    tar xzf $(basename "$OUTPUT")"
echo "    cd $BUNDLE_NAME/swiftllm"
echo "    ./install.sh --airgap"
echo ""
