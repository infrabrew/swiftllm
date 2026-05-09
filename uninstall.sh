#!/usr/bin/env bash
# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      uninstall.sh
# PATH:      /uninstall.sh
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
# SwiftLLM Uninstaller
# ============================================================================
#
# Removes SwiftLLM from the system, including the Python package, optional
# dependencies, cached models, and build artifacts.
#
# Usage:
#   ./uninstall.sh                    # Interactive uninstall
#   ./uninstall.sh --venv DIR         # Uninstall from a specific venv
#   ./uninstall.sh --no-venv          # Uninstall from current Python
#   ./uninstall.sh --keep-models      # Keep downloaded models
#   ./uninstall.sh --keep-venv        # Keep the venv directory
#   ./uninstall.sh --purge            # Remove everything (models, venv, build)
#   ./uninstall.sh --yes              # Skip confirmation prompts
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
KEEP_MODELS=false
KEEP_VENV=false
PURGE=false
YES=false
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

confirm() {
    if $YES; then
        return 0
    fi
    local prompt="$1 [y/N] "
    echo -en "$prompt"
    read -r answer
    case "$answer" in
        [yY][eE][sS]|[yY]) return 0 ;;
        *) return 1 ;;
    esac
}

# ----------------------------
# Parse arguments
# ----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            [[ -z "${2:-}" || "$2" == --* ]] && fail "--venv requires a directory argument"
            VENV_DIR="$2"; shift 2 ;;
        --no-venv)    NO_VENV=true; shift ;;
        --keep-models) KEEP_MODELS=true; shift ;;
        --keep-venv)  KEEP_VENV=true; shift ;;
        --purge)      PURGE=true; shift ;;
        --yes|-y)     YES=true; shift ;;
        -h|--help)
            echo "SwiftLLM Uninstaller"
            echo ""
            echo "Usage: ./uninstall.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --venv DIR       Uninstall from venv at DIR"
            echo "  --no-venv        Uninstall from current Python (no venv)"
            echo "  --keep-models    Keep downloaded models in ~/.cache/swiftllm"
            echo "  --keep-venv      Keep the virtual environment directory"
            echo "  --purge          Remove everything (models, venv, build artifacts)"
            echo "  --yes, -y        Skip confirmation prompts"
            echo "  -h, --help       Show this help message"
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
echo -e "${BOLD} Uninstall${NC}"
echo ""

# ----------------------------
# Detect Python
# ----------------------------
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

# Activate venv if exists
if ! $NO_VENV; then
    if [[ -z "$VENV_DIR" ]]; then
        VENV_DIR="$SCRIPT_DIR/venv"
    fi
    if [[ -d "$VENV_DIR" ]] && [[ -f "$VENV_DIR/bin/activate" ]]; then
        source "$VENV_DIR/bin/activate"
        PYTHON="$VENV_DIR/bin/python"
        PIP="$VENV_DIR/bin/pip"
        info "Using venv: $VENV_DIR"
    else
        PIP="$PYTHON -m pip"
        info "No venv found at $VENV_DIR — using system Python"
    fi
else
    PIP="$PYTHON -m pip"
fi

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

# ----------------------------
# Check what's installed
# ----------------------------
step "Checking installation..."

VERSION=$($PYTHON -c "
try:
    import swiftllm
    print(swiftllm.__version__)
except Exception:
    print('')
" 2>/dev/null)

if [[ -z "$VERSION" ]]; then
    warn "SwiftLLM does not appear to be installed"
else
    info "Found SwiftLLM version: $VERSION"
fi

# Model cache location
MODEL_DIR="${SWIFTLLM_MODEL_DIR:-$HOME/.cache/swiftllm}"
MODEL_SIZE=""
if [[ -d "$MODEL_DIR" ]]; then
    MODEL_SIZE=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
    info "Model cache: $MODEL_DIR ($MODEL_SIZE)"
fi

# Build artifacts
BUILD_SIZE=""
if [[ -d "$SCRIPT_DIR/target" ]]; then
    BUILD_SIZE=$(du -sh "$SCRIPT_DIR/target" 2>/dev/null | cut -f1)
    info "Build artifacts: $SCRIPT_DIR/target ($BUILD_SIZE)"
fi

# ----------------------------
# Confirm
# ----------------------------
echo ""
echo -e "${BOLD}The following will be removed:${NC}"
echo ""
echo "  1. swiftllm Python package"
echo "  2. llama-cpp-python package (optional)"

if $PURGE || ! $KEEP_MODELS; then
    if [[ -d "$MODEL_DIR" ]]; then
        echo "  3. Model cache: $MODEL_DIR ($MODEL_SIZE)"
    fi
fi

if $PURGE || ! $KEEP_VENV; then
    if [[ -d "$VENV_DIR" ]]; then
        echo "  4. Virtual environment: $VENV_DIR"
    fi
fi

if $PURGE; then
    if [[ -d "$SCRIPT_DIR/target" ]]; then
        echo "  5. Build artifacts: $SCRIPT_DIR/target ($BUILD_SIZE)"
    fi
fi

echo ""

if ! confirm "Proceed with uninstall?"; then
    echo -e "${YELLOW}Uninstall cancelled.${NC}"
    exit 0
fi

# ----------------------------
# Step 1: Uninstall Python packages
# ----------------------------
step "Removing Python packages..."

if $PIP show swiftllm &>/dev/null 2>&1; then
    $PIP uninstall swiftllm --yes "${PIP_EXTRA[@]}" 2>/dev/null
    success "swiftllm package removed"
else
    info "swiftllm package not found (already removed)"
fi

# Ask about llama-cpp-python
if $PIP show llama-cpp-python &>/dev/null 2>&1; then
    if $YES || confirm "Remove llama-cpp-python as well?"; then
        $PIP uninstall llama-cpp-python --yes "${PIP_EXTRA[@]}" 2>/dev/null
        success "llama-cpp-python removed"
    else
        info "Keeping llama-cpp-python"
    fi
fi

# ----------------------------
# Step 2: Remove model cache
# ----------------------------
if $PURGE && ! $KEEP_MODELS; then
    step "Removing model cache..."
    if [[ -d "$MODEL_DIR" ]]; then
        rm -rf "$MODEL_DIR"
        success "Model cache removed: $MODEL_DIR"
    else
        info "No model cache to remove"
    fi
elif ! $KEEP_MODELS; then
    step "Model cache..."
    if [[ -d "$MODEL_DIR" ]]; then
        if confirm "Remove model cache at $MODEL_DIR ($MODEL_SIZE)?"; then
            rm -rf "$MODEL_DIR"
            success "Model cache removed"
        else
            info "Keeping model cache"
        fi
    fi
else
    info "Keeping model cache (--keep-models)"
fi

# ----------------------------
# Step 3: Remove venv
# ----------------------------
if ! $NO_VENV && ! $KEEP_VENV; then
    step "Virtual environment..."
    if [[ -d "$VENV_DIR" ]]; then
        if $PURGE || confirm "Remove virtual environment at $VENV_DIR?"; then
            # Deactivate first if active
            deactivate 2>/dev/null || true
            rm -rf "$VENV_DIR"
            success "Virtual environment removed: $VENV_DIR"
        else
            info "Keeping virtual environment"
        fi
    fi
fi

# ----------------------------
# Step 4: Remove build artifacts
# ----------------------------
if $PURGE; then
    step "Removing build artifacts..."
    if [[ -d "$SCRIPT_DIR/target" ]]; then
        rm -rf "$SCRIPT_DIR/target"
        success "Build artifacts removed"
    fi
    # Clean any .egg-info or __pycache__
    find "$SCRIPT_DIR/python" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$SCRIPT_DIR/python" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    success "Cache files cleaned"
fi

# ----------------------------
# Summary
# ----------------------------
echo ""
echo -e "${BOLD}${CYAN}============================================${NC}"
echo -e "${BOLD}${GREEN}  SwiftLLM uninstall complete!${NC}"
echo -e "${BOLD}${CYAN}============================================${NC}"
echo ""

echo -e "  ${BOLD}Removed:${NC}"
echo "    - swiftllm Python package"
if ! $KEEP_MODELS && [[ -n "$MODEL_SIZE" ]]; then
    echo "    - Model cache ($MODEL_SIZE)"
fi
if ! $KEEP_VENV && [[ -d "$VENV_DIR" ]] 2>/dev/null; then
    echo "    - Virtual environment"
fi
if $PURGE && [[ -n "$BUILD_SIZE" ]]; then
    echo "    - Build artifacts ($BUILD_SIZE)"
fi

echo ""
echo -e "  ${BOLD}To reinstall:${NC}"
echo "    ./install.sh"
echo ""

# ------------------------------------------------------------------------------
# END OF FILE: uninstall.sh
# REPO PATH:   /swiftllm/uninstall.sh
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
