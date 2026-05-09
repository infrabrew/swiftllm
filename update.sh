#!/usr/bin/env bash
# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      update.sh
# PATH:      /update.sh
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
# SwiftLLM Updater
# ============================================================================
#
# Updates an existing SwiftLLM installation by pulling the latest source,
# rebuilding the Rust backend + Python wheel, and reinstalling.
#
# Usage:
#   ./update.sh              # Update from git (auto-detects GPU)
#   ./update.sh --cpu        # Force CPU-only rebuild
#   ./update.sh --gpu        # Force GPU/CUDA rebuild
#   ./update.sh --venv DIR   # Use a specific venv directory
#   ./update.sh --no-venv    # Update into current Python environment
#   ./update.sh --branch BR  # Checkout a specific branch before building
#   ./update.sh --tag TAG    # Checkout a specific tag before building
#   ./update.sh --no-pull    # Rebuild from current source (skip git pull)
#   ./update.sh --clean      # Clean build artifacts before rebuilding
#
# ============================================================================

set -eo pipefail

# ----------------------------
# Colors
# ----------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ----------------------------
# Defaults
# ----------------------------
VENV_DIR=""
NO_VENV=false
FORCE_CPU=false
FORCE_GPU=false
BRANCH=""
TAG=""
NO_PULL=false
CLEAN_BUILD=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------
# Helpers
# ----------------------------
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()    { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
step()    { echo -e "\n${BOLD}${CYAN}=> $1${NC}"; }

command_exists() { command -v "$1" &>/dev/null; }

# ----------------------------
# Parse arguments
# ----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)       FORCE_CPU=true; shift ;;
        --gpu)       FORCE_GPU=true; shift ;;
        --venv)
            [[ -z "${2:-}" || "$2" == --* ]] && fail "--venv requires a directory argument"
            VENV_DIR="$2"; shift 2 ;;
        --no-venv)   NO_VENV=true; shift ;;
        --branch)
            [[ -z "${2:-}" || "$2" == --* ]] && fail "--branch requires an argument"
            BRANCH="$2"; shift 2 ;;
        --tag)
            [[ -z "${2:-}" || "$2" == --* ]] && fail "--tag requires an argument"
            TAG="$2"; shift 2 ;;
        --no-pull)   NO_PULL=true; shift ;;
        --clean)     CLEAN_BUILD=true; shift ;;
        -h|--help)
            echo "SwiftLLM Updater"
            echo ""
            echo "Usage: ./update.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu          Rebuild CPU-only wheel"
            echo "  --gpu          Force GPU/CUDA rebuild"
            echo "  --venv DIR     Use virtual environment at DIR"
            echo "  --no-venv      Update in current Python environment"
            echo "  --branch BR    Checkout branch BR before building"
            echo "  --tag TAG      Checkout tag TAG before building"
            echo "  --no-pull      Skip git pull (rebuild from current source)"
            echo "  --clean        Clean build artifacts before rebuilding"
            echo "  -h, --help     Show this help message"
            exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done

# ----------------------------
# Banner
# ----------------------------
echo ""
echo -e "${BOLD}${CYAN}"
echo "  ____          _  __ _   _     _     __  __ "
echo " / ___|_      _(_)/ _| |_| |   | |   |  \/  |"
echo " \___ \ \ /\ / / | |_| __| |   | |   | |\/| |"
echo "  ___) \ V  V /| |  _| |_| |___| |___| |  | |"
echo " |____/ \_/\_/ |_|_|  \__|_____|_____|_|  |_|"
echo -e "${NC}"
echo -e "${BOLD} Update${NC}"
echo ""

# ----------------------------
# Step 1: Capture current version
# ----------------------------
step "Checking current installation..."

cd "$SCRIPT_DIR"

PYTHON=""
for py in python3 python; do
    if command_exists "$py"; then
        PY_MAJOR=$("$py" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        PY_MINOR=$("$py" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [[ "$PY_MAJOR" -gt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -ge 8 ]]; }; then
            PYTHON="$py"
            break
        fi
    fi
done
[[ -z "$PYTHON" ]] && fail "Python 3.8+ required"

# Activate venv if specified
if ! $NO_VENV; then
    if [[ -z "$VENV_DIR" ]]; then
        VENV_DIR="$SCRIPT_DIR/venv"
    fi
    if [[ -d "$VENV_DIR" ]] && [[ -f "$VENV_DIR/bin/activate" ]]; then
        source "$VENV_DIR/bin/activate"
        PYTHON="$VENV_DIR/bin/python"
        PIP="$VENV_DIR/bin/pip"
        success "Activated venv: $VENV_DIR"
    else
        warn "Venv not found at $VENV_DIR — will install without venv"
        PIP="$PYTHON -m pip"
    fi
else
    PIP="$PYTHON -m pip"
fi

# Get old version
OLD_VERSION=$($PYTHON -c "
try:
    import swiftllm
    print(swiftllm.__version__)
except Exception:
    print('not installed')
" 2>/dev/null)
info "Current version: $OLD_VERSION"

# ----------------------------
# Step 2: Update source code
# ----------------------------
step "Updating source code..."

if $NO_PULL; then
    info "Skipping git pull (--no-pull)"
else
    if [[ -d "$SCRIPT_DIR/.git" ]]; then
        # Stash any local changes
        STASHED=false
        if [[ -n "$(git -C "$SCRIPT_DIR" status --porcelain 2>/dev/null)" ]]; then
            info "Stashing local changes..."
            git -C "$SCRIPT_DIR" stash push -m "swiftllm-update-$(date +%s)" --quiet 2>/dev/null && STASHED=true
        fi

        # Checkout branch/tag if specified
        if [[ -n "$TAG" ]]; then
            info "Checking out tag: $TAG"
            git -C "$SCRIPT_DIR" fetch --tags --quiet 2>/dev/null
            git -C "$SCRIPT_DIR" checkout "$TAG" --quiet 2>/dev/null || fail "Failed to checkout tag $TAG"
            success "Checked out tag: $TAG"
        elif [[ -n "$BRANCH" ]]; then
            info "Checking out branch: $BRANCH"
            git -C "$SCRIPT_DIR" fetch origin --quiet 2>/dev/null
            git -C "$SCRIPT_DIR" checkout "$BRANCH" --quiet 2>/dev/null || fail "Failed to checkout branch $BRANCH"
            git -C "$SCRIPT_DIR" pull origin "$BRANCH" --quiet 2>/dev/null
            success "Checked out and pulled branch: $BRANCH"
        else
            CURRENT_BRANCH=$(git -C "$SCRIPT_DIR" branch --show-current 2>/dev/null)
            info "Pulling latest on: $CURRENT_BRANCH"
            git -C "$SCRIPT_DIR" pull --quiet 2>/dev/null || warn "git pull failed (may be detached HEAD or no remote)"
        fi

        # Pop stash if we stashed
        if $STASHED; then
            info "Restoring local changes..."
            git -C "$SCRIPT_DIR" stash pop --quiet 2>/dev/null || warn "Could not restore stashed changes"
        fi

        NEW_COMMIT=$(git -C "$SCRIPT_DIR" log --oneline -1 2>/dev/null)
        success "Source updated: $NEW_COMMIT"
    else
        warn "Not a git repository — skipping source update"
    fi
fi

# ----------------------------
# Step 3: Detect GPU
# ----------------------------
step "Detecting GPU..."

HAS_NVIDIA=false
HAS_CUDA=false
NVCC_PATH=""

if command_exists nvidia-smi; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    [[ -n "$GPU_NAME" ]] && HAS_NVIDIA=true && success "Found GPU: $GPU_NAME"
fi

for nvcc_candidate in nvcc /usr/local/cuda/bin/nvcc /usr/lib/cuda/bin/nvcc; do
    if [[ -x "$(command -v "$nvcc_candidate" 2>/dev/null || echo "$nvcc_candidate")" ]] || [[ -f "$nvcc_candidate" ]]; then
        NVCC_PATH="$nvcc_candidate"
        HAS_CUDA=true
        break
    fi
done

USE_GPU=false
if $FORCE_GPU; then
    USE_GPU=true
elif $FORCE_CPU; then
    USE_GPU=false
elif $HAS_NVIDIA && $HAS_CUDA; then
    USE_GPU=true
fi

if $USE_GPU; then
    info "GPU mode enabled"
else
    info "CPU-only mode"
fi

# ----------------------------
# Step 4: Clean (optional)
# ----------------------------
if $CLEAN_BUILD; then
    step "Cleaning build artifacts..."
    cargo clean --manifest-path "$SCRIPT_DIR/Cargo.toml" 2>/dev/null || true
    rm -rf "$SCRIPT_DIR/target/wheels" 2>/dev/null
    success "Build artifacts cleaned"
fi

# ----------------------------
# Step 5: Check toolchain
# ----------------------------
step "Checking build tools..."

command_exists rustc || fail "Rust not found. Run install.sh first or install Rust: https://rustup.rs"
RUST_VERSION=$(rustc --version | awk '{print $2}')
success "Rust $RUST_VERSION"

command_exists maturin || $PIP install --quiet maturin 2>/dev/null || fail "maturin not available"
success "maturin available"

# ----------------------------
# Step 6: Rebuild
# ----------------------------
step "Rebuilding SwiftLLM..."

cd "$SCRIPT_DIR"

if $USE_GPU; then
    BUILD_FEATURES="cuda"
else
    BUILD_FEATURES="cpu"
fi

info "Building with features: $BUILD_FEATURES"
maturin build --release --no-default-features --features "$BUILD_FEATURES" 2>&1 | tail -5

WHEEL=$(ls -t "$SCRIPT_DIR/target/wheels/swiftllm-"*.whl 2>/dev/null | head -1)
[[ -z "$WHEEL" ]] && fail "Build failed — no wheel file produced"
success "Built: $(basename "$WHEEL")"

# ----------------------------
# Step 7: Reinstall
# ----------------------------
step "Installing updated wheel..."

PIP_EXTRA=()
if $NO_VENV; then
    if $PYTHON -c "
import sysconfig, os
p = sysconfig.get_path('stdlib')
marker = os.path.join(os.path.dirname(p), 'EXTERNALLY-MANAGED')
import sys
sys.exit(0 if os.path.exists(marker) else 1)
" 2>/dev/null; then
        PIP_EXTRA=(--break-system-packages)
    fi
fi

$PIP install --force-reinstall "$WHEEL[serve]" --quiet "${PIP_EXTRA[@]}" || fail "Wheel install failed"
success "SwiftLLM reinstalled"

# ----------------------------
# Step 8: Verify
# ----------------------------
step "Verifying update..."

NEW_VERSION=$($PYTHON -c "
try:
    import swiftllm
    print(swiftllm.__version__)
except Exception:
    print('unknown')
" 2>/dev/null)

ERRORS=0

if $PYTHON -c "from swiftllm import LLM, SamplingParams, resolve_model" 2>/dev/null; then
    success "Python imports OK"
else
    warn "Python import check failed"
    ERRORS=$((ERRORS + 1))
fi

if $PYTHON -c "from llama_cpp import Llama" 2>/dev/null; then
    success "llama-cpp-python OK"
else
    warn "llama-cpp-python not found (GGUF support unavailable)"
fi

# ----------------------------
# Summary
# ----------------------------
echo ""
echo -e "${BOLD}${CYAN}============================================${NC}"
echo -e "${BOLD}${GREEN}  SwiftLLM update complete!${NC}"
echo -e "${BOLD}${CYAN}============================================${NC}"
echo ""
echo -e "  ${BOLD}Previous version:${NC} $OLD_VERSION"
echo -e "  ${BOLD}New version:${NC}      $NEW_VERSION"
echo ""

if $USE_GPU; then
    echo -e "  ${BOLD}GPU acceleration:${NC} ${GREEN}Enabled${NC}"
else
    echo -e "  ${BOLD}GPU acceleration:${NC} CPU only"
fi

if [[ $ERRORS -gt 0 ]]; then
    warn "$ERRORS verification check(s) had warnings."
fi

echo ""

# ------------------------------------------------------------------------------
# END OF FILE: update.sh
# REPO PATH:   /swiftllm/update.sh
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
