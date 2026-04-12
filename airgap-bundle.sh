#!/usr/bin/env bash
# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      airgap-bundle.sh
# PATH:      /airgap-bundle.sh
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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

set -eo pipefail

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
TARGET_ARCH=""

# ----------------------------
# Parse arguments
# ----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|-m)
            [[ -z "${2:-}" || "$2" == --* ]] && fail "--model requires an argument"
            MODELS+=("$2"); shift 2 ;;
        --cpu)          CPU_ONLY=true; shift ;;
        -o|--output)
            [[ -z "${2:-}" || "$2" == --* ]] && fail "-o/--output requires a path argument"
            OUTPUT="$2"; shift 2 ;;
        --platform)
            [[ -z "${2:-}" || "$2" == --* ]] && fail "--platform requires a platform tag argument"
            PLATFORM="$2"; shift 2 ;;
        --arch)
            [[ -z "${2:-}" || "$2" == --* ]] && fail "--arch requires an argument (x86_64, aarch64, arm64)"
            TARGET_ARCH="$2"; shift 2 ;;
        -h|--help)
            echo "SwiftLLM Air-Gap Bundle Creator"
            echo ""
            echo "Usage: ./airgap-bundle.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model, -m MODEL   Include a model (repeatable)"
            echo "  --cpu               Download CPU-only wheels"
            echo "  --arch ARCH         Target architecture: x86_64, aarch64, arm64 (auto)"
            echo "  --platform PLAT     Explicit pip platform tag (overrides --arch)"
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
        if [[ "$PY_MAJOR" -gt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -ge 8 ]]; }; then
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
BUNDLE_DIR=$(mktemp -d) || fail "Failed to create temporary directory"
trap 'rm -rf "$BUNDLE_DIR"' EXIT
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
    --exclude='.env*' --exclude='*.pem' --exclude='*.key' \
    --exclude='*.log' --exclude='models/' \
    "$SCRIPT_DIR/" "$DEST/swiftllm/"
success "Source tree copied"

# ----------------------------
# 2. Download pip wheels for all dependencies
# ----------------------------
step "Downloading Python wheels..."

# If --arch was given but --platform wasn't, map arch → manylinux/macOS platform tag.
# Users can still pass --platform explicitly to override.
if [[ -z "$PLATFORM" && -n "$TARGET_ARCH" ]]; then
    HOST_OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    case "$TARGET_ARCH" in
        x86_64)          PIP_ARCH="x86_64" ;;
        aarch64|arm64)   PIP_ARCH="aarch64" ;;
        *)               fail "Unsupported --arch '$TARGET_ARCH' (expected x86_64, aarch64, arm64)" ;;
    esac
    case "$HOST_OS" in
        linux)
            # manylinux2014 covers glibc >=2.17 which is modern enough for
            # Ubuntu 20.04+, RHEL 8+, Debian 10+.
            PLATFORM="manylinux2014_${PIP_ARCH}"
            ;;
        darwin)
            # macOS 11+ is the baseline for Apple Silicon; pip uses macosx_11_0_arm64
            if [[ "$PIP_ARCH" == "aarch64" ]]; then
                PLATFORM="macosx_11_0_arm64"
            else
                PLATFORM="macosx_10_15_x86_64"
            fi
            ;;
        *)
            warn "Cannot auto-map --arch on OS '$HOST_OS'; pass --platform explicitly."
            ;;
    esac
    [[ -n "$PLATFORM" ]] && info "Auto-selected pip platform: $PLATFORM (from --arch $TARGET_ARCH)"
fi

PLATFORM_FLAG=()
if [[ -n "$PLATFORM" ]]; then
    # Validate platform tag: only allow alphanumeric, underscores, dots, hyphens
    if [[ ! "$PLATFORM" =~ ^[a-zA-Z0-9._-]+$ ]]; then
        fail "Invalid platform tag: $PLATFORM"
    fi
    PLATFORM_FLAG=(--platform "$PLATFORM" --only-binary=:all: --python-version "$PY_VERSION")
fi

# Core build tool
$PYTHON -m pip download maturin -d "$DEST/wheels" "${PLATFORM_FLAG[@]}" 2>&1 | tail -3
success "maturin wheels downloaded"

# Runtime dependencies from pyproject.toml
$PYTHON -m pip download \
    "numpy>=1.20" "transformers>=4.30" "torch>=2.0" "safetensors>=0.3" \
    "tokenizers>=0.13" "tqdm>=4.60" "requests>=2.25" "aiohttp>=3.8" \
    "pydantic>=2.0" "huggingface-hub>=0.14" \
    "fastapi>=0.100" "uvicorn>=0.23" \
    -d "$DEST/wheels" "${PLATFORM_FLAG[@]}" 2>&1 | tail -5
success "Runtime dependency wheels downloaded"

# llama-cpp-python
if $CPU_ONLY; then
    $PYTHON -m pip download llama-cpp-python -d "$DEST/wheels" "${PLATFORM_FLAG[@]}" 2>&1 | tail -3
else
    $PYTHON -m pip download llama-cpp-python -d "$DEST/wheels" "${PLATFORM_FLAG[@]}" 2>&1 | tail -3
fi
success "llama-cpp-python wheel downloaded"

# pip itself (for bootstrapping)
$PYTHON -m pip download pip setuptools wheel -d "$DEST/wheels" "${PLATFORM_FLAG[@]}" 2>&1 | tail -3
success "pip/setuptools/wheel downloaded"

WHEEL_COUNT=$(ls "$DEST/wheels/" | wc -l | tr -d ' ')
WHEEL_SIZE=$(du -sh "$DEST/wheels/" | cut -f1)
info "$WHEEL_COUNT wheel(s), $WHEEL_SIZE total"

# ----------------------------
# 3. Download Rust standalone installer
# ----------------------------
step "Downloading Rust installer..."

# Detect target triple. Rust uses `aarch64` for 64-bit ARM on both Linux and
# macOS; `uname -m` reports `arm64` on Apple Silicon, so normalize.
# If --arch was specified, use that instead of the host arch.
ARCH="${TARGET_ARCH:-$(uname -m)}"
case "$ARCH" in
    arm64)   RUST_ARCH="aarch64" ;;   # Apple Silicon
    aarch64) RUST_ARCH="aarch64" ;;   # Linux ARM64 (Graviton, RPi, etc.)
    x86_64)  RUST_ARCH="x86_64" ;;
    i686|i386) RUST_ARCH="i686" ;;
    *)       RUST_ARCH="$ARCH" ;;
esac
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$OS" in
    linux)  RUST_TARGET="${RUST_ARCH}-unknown-linux-gnu" ;;
    darwin) RUST_TARGET="${RUST_ARCH}-apple-darwin" ;;
    *)      warn "Unsupported OS ($OS). Skipping Rust installer." ; RUST_TARGET="" ;;
esac
info "Target triple: ${RUST_TARGET:-unknown}"

if [[ -n "$RUST_TARGET" ]]; then
    RUSTUP_URL="https://static.rust-lang.org/rustup/dist/${RUST_TARGET}/rustup-init"
    RUSTUP_SHA_URL="${RUSTUP_URL}.sha256"
    if curl -sSfL "$RUSTUP_URL" -o "$DEST/rust/rustup-init"; then
        # Verify SHA256 checksum
        if curl -sSfL "$RUSTUP_SHA_URL" -o "$DEST/rust/rustup-init.sha256" 2>/dev/null; then
            EXPECTED_SHA=$(awk '{print $1}' "$DEST/rust/rustup-init.sha256")
            # Portable SHA256: sha256sum on Linux, shasum on macOS
            if command -v sha256sum &>/dev/null; then
                ACTUAL_SHA=$(sha256sum "$DEST/rust/rustup-init" | awk '{print $1}')
            elif command -v shasum &>/dev/null; then
                ACTUAL_SHA=$(shasum -a 256 "$DEST/rust/rustup-init" | awk '{print $1}')
            else
                fail "Neither sha256sum nor shasum available — cannot verify rustup-init"
            fi
            if [[ "$EXPECTED_SHA" != "$ACTUAL_SHA" ]]; then
                fail "SHA256 checksum mismatch for rustup-init (expected $EXPECTED_SHA, got $ACTUAL_SHA)"
            fi
            rm -f "$DEST/rust/rustup-init.sha256"
            success "Downloaded and verified rustup-init for $RUST_TARGET"
        else
            warn "Could not download checksum file. Proceeding without verification."
            success "Downloaded rustup-init for $RUST_TARGET (unverified)"
        fi
        chmod +x "$DEST/rust/rustup-init"
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
import sys, shutil, os
from swiftllm.model_resolver import resolve_model
model_id = sys.argv[1]
dest_dir = sys.argv[2]
path = resolve_model(model_id)
dest = os.path.join(dest_dir, os.path.basename(path))
if os.path.isfile(path):
    shutil.copy2(path, dest)
elif os.path.isdir(path):
    shutil.copytree(path, dest)
print(f'Saved to: {dest}')
" "$model" "$DEST/models" 2>&1
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

# Cleanup (trap handles this, but clear it to avoid double-rm)
trap - EXIT
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

# ------------------------------------------------------------------------------
# END OF FILE: airgap-bundle.sh
# REPO PATH:   /swiftllm/airgap-bundle.sh
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
