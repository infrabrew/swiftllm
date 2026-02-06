#!/usr/bin/env bash
# ============================================================================
# SwiftLLM Installer
# ============================================================================
#
# Usage:
#   ./install.sh              # Interactive install (auto-detects GPU)
#   ./install.sh --cpu        # CPU-only install (skip CUDA)
#   ./install.sh --gpu        # Force GPU/CUDA install
#   ./install.sh --venv DIR   # Use a specific venv directory
#   ./install.sh --no-venv    # Install into current Python environment
#   ./install.sh --model-dir  # Set default model download directory
#
# ============================================================================

set -e

# ----------------------------
# Colors
# ----------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ----------------------------
# Defaults
# ----------------------------
VENV_DIR=""
NO_VENV=false
FORCE_CPU=false
FORCE_GPU=false
MODEL_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------
# Parse arguments
# ----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            FORCE_CPU=true
            shift
            ;;
        --gpu)
            FORCE_GPU=true
            shift
            ;;
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        --no-venv)
            NO_VENV=true
            shift
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "SwiftLLM Installer"
            echo ""
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu          CPU-only install (skip CUDA/GPU support)"
            echo "  --gpu          Force GPU/CUDA install"
            echo "  --venv DIR     Create/use virtual environment at DIR"
            echo "  --no-venv      Install into current Python (no venv)"
            echo "  --model-dir    Set default model download directory"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

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
echo -e "${BOLD} High-Performance LLM Inference Engine${NC}"
echo ""

# ----------------------------
# Step 1: Check Python
# ----------------------------
step "Checking Python..."

PYTHON=""
for py in python3 python; do
    if command_exists "$py"; then
        PY_VERSION=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        PY_MAJOR=$("$py" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        PY_MINOR=$("$py" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [[ "$PY_MAJOR" -ge 3 ]] && [[ "$PY_MINOR" -ge 8 ]]; then
            PYTHON="$py"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    fail "Python 3.8+ is required but not found. Please install Python first."
fi

success "Found Python $PY_VERSION ($PYTHON)"

# ----------------------------
# Step 2: Detect GPU / CUDA
# ----------------------------
step "Detecting GPU..."

HAS_NVIDIA=false
HAS_CUDA=false
NVCC_PATH=""
CUDA_VERSION=""

# Check for NVIDIA GPU
if command_exists nvidia-smi; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ -n "$GPU_NAME" ]]; then
        HAS_NVIDIA=true
        success "Found GPU: $GPU_NAME (${GPU_VRAM} MB VRAM)"
    fi
else
    info "No NVIDIA GPU detected (nvidia-smi not found)"
fi

# Check for CUDA toolkit (nvcc)
if command_exists nvcc; then
    NVCC_PATH="$(which nvcc)"
    HAS_CUDA=true
elif [[ -f /usr/local/cuda/bin/nvcc ]]; then
    NVCC_PATH="/usr/local/cuda/bin/nvcc"
    HAS_CUDA=true
elif [[ -f /usr/lib/cuda/bin/nvcc ]]; then
    NVCC_PATH="/usr/lib/cuda/bin/nvcc"
    HAS_CUDA=true
fi

if $HAS_CUDA; then
    CUDA_VERSION=$("$NVCC_PATH" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
    success "Found CUDA toolkit: $CUDA_VERSION ($NVCC_PATH)"
else
    info "CUDA toolkit (nvcc) not found"
fi

# Decide GPU mode
USE_GPU=false
if $FORCE_GPU; then
    if ! $HAS_NVIDIA; then
        warn "No NVIDIA GPU detected, but --gpu was specified. Will attempt GPU install."
    fi
    if ! $HAS_CUDA; then
        warn "CUDA toolkit not found. llama-cpp-python GPU build may fail."
        warn "Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
    fi
    USE_GPU=true
elif $FORCE_CPU; then
    USE_GPU=false
    info "CPU-only mode selected"
elif $HAS_NVIDIA && $HAS_CUDA; then
    USE_GPU=true
    info "GPU mode auto-detected"
else
    USE_GPU=false
    if $HAS_NVIDIA && ! $HAS_CUDA; then
        warn "NVIDIA GPU found but CUDA toolkit is missing. Falling back to CPU."
        warn "For GPU support, install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
    fi
    info "Installing in CPU-only mode"
fi

# ----------------------------
# Step 3: Set up virtual environment
# ----------------------------
step "Setting up Python environment..."

if $NO_VENV; then
    info "Skipping virtual environment (--no-venv)"
    PIP="$PYTHON -m pip"
else
    # Default venv location
    if [[ -z "$VENV_DIR" ]]; then
        VENV_DIR="$SCRIPT_DIR/venv"
    fi

    if [[ -d "$VENV_DIR" ]] && [[ -f "$VENV_DIR/bin/activate" ]]; then
        info "Using existing venv: $VENV_DIR"
    else
        info "Creating virtual environment at $VENV_DIR"
        "$PYTHON" -m venv "$VENV_DIR" || fail "Failed to create virtual environment. Install python3-venv."
        success "Virtual environment created"
    fi

    source "$VENV_DIR/bin/activate"
    PYTHON="$VENV_DIR/bin/python"
    PIP="$VENV_DIR/bin/pip"
    success "Activated venv: $VENV_DIR"
fi

# Upgrade pip
info "Upgrading pip..."
$PYTHON -m pip install --upgrade pip --quiet 2>/dev/null
success "pip is up to date"

# ----------------------------
# Step 4: Check Rust (for building from source)
# ----------------------------
step "Checking Rust toolchain..."

if command_exists rustc; then
    RUST_VERSION=$(rustc --version | awk '{print $2}')
    success "Found Rust $RUST_VERSION"
else
    info "Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet
    source "$HOME/.cargo/env" 2>/dev/null || true
    if command_exists rustc; then
        RUST_VERSION=$(rustc --version | awk '{print $2}')
        success "Installed Rust $RUST_VERSION"
    else
        fail "Failed to install Rust. Install manually: https://rustup.rs"
    fi
fi

# ----------------------------
# Step 5: Install maturin (Rust-Python build tool)
# ----------------------------
step "Installing build tools..."

$PIP install --quiet maturin 2>/dev/null
success "maturin installed"

# ----------------------------
# Step 6: Build SwiftLLM
# ----------------------------
step "Building SwiftLLM from source..."

cd "$SCRIPT_DIR"

info "Running maturin build (this may take a few minutes)..."
maturin build --release 2>&1 | tail -5

# Find the built wheel
WHEEL=$(ls -t "$SCRIPT_DIR/target/wheels/swiftllm-"*.whl 2>/dev/null | head -1)

if [[ -z "$WHEEL" ]]; then
    fail "Build failed - no wheel file found in target/wheels/"
fi

success "Built: $(basename "$WHEEL")"

# ----------------------------
# Step 7: Install SwiftLLM wheel
# ----------------------------
step "Installing SwiftLLM..."

$PIP install --force-reinstall "$WHEEL" --quiet 2>/dev/null
success "SwiftLLM installed"

# ----------------------------
# Step 8: Install llama-cpp-python (GGUF support)
# ----------------------------
step "Installing GGUF backend (llama-cpp-python)..."

if $USE_GPU; then
    info "Building llama-cpp-python with CUDA support..."

    export CUDACXX="$NVCC_PATH"
    export CMAKE_ARGS="-DGGML_CUDA=on"

    $PIP install llama-cpp-python --force-reinstall --no-cache-dir 2>&1 | tail -3

    if $PYTHON -c "from llama_cpp import Llama; print('ok')" 2>/dev/null | grep -q ok; then
        success "llama-cpp-python installed with CUDA support"
    else
        warn "CUDA build may have failed. Falling back to CPU build..."
        unset CUDACXX CMAKE_ARGS
        $PIP install llama-cpp-python --force-reinstall --no-cache-dir --quiet 2>/dev/null
        success "llama-cpp-python installed (CPU fallback)"
    fi
else
    info "Building llama-cpp-python (CPU only)..."
    $PIP install llama-cpp-python --quiet 2>/dev/null
    success "llama-cpp-python installed (CPU)"
fi

# ----------------------------
# Step 9: Set model directory
# ----------------------------
step "Configuring model directory..."

if [[ -n "$MODEL_DIR" ]]; then
    mkdir -p "$MODEL_DIR" 2>/dev/null
    success "Model directory: $MODEL_DIR"
    info "Set SWIFTLLM_MODEL_DIR=$MODEL_DIR in your shell profile to persist this."
else
    DEFAULT_MODEL_DIR="$HOME/.cache/swiftllm/models"
    mkdir -p "$DEFAULT_MODEL_DIR" 2>/dev/null
    success "Default model directory: $DEFAULT_MODEL_DIR"
    info "Override with: export SWIFTLLM_MODEL_DIR=/your/path"
fi

# ----------------------------
# Step 10: Verify installation
# ----------------------------
step "Verifying installation..."

ERRORS=0

# Check swiftllm CLI
if command_exists swiftllm || [[ -f "$VENV_DIR/bin/swiftllm" ]]; then
    SLLM_VERSION=$($PYTHON -c "import swiftllm; print('0.1.0')" 2>/dev/null || echo "unknown")
    success "swiftllm CLI available (v$SLLM_VERSION)"
else
    warn "swiftllm CLI not found on PATH"
    ERRORS=$((ERRORS + 1))
fi

# Check Python import
if $PYTHON -c "from swiftllm import LLM, SamplingParams, resolve_model" 2>/dev/null; then
    success "Python imports OK"
else
    warn "Python import check failed"
    ERRORS=$((ERRORS + 1))
fi

# Check llama-cpp-python
if $PYTHON -c "from llama_cpp import Llama" 2>/dev/null; then
    success "llama-cpp-python OK"
else
    warn "llama-cpp-python import failed"
    ERRORS=$((ERRORS + 1))
fi

# ----------------------------
# Summary
# ----------------------------
echo ""
echo -e "${BOLD}${CYAN}============================================${NC}"
echo -e "${BOLD}${GREEN}  SwiftLLM installation complete!${NC}"
echo -e "${BOLD}${CYAN}============================================${NC}"
echo ""

if ! $NO_VENV && [[ -n "$VENV_DIR" ]]; then
    echo -e "  ${BOLD}Activate venv:${NC}"
    echo -e "    source ${VENV_DIR}/bin/activate"
    echo ""
fi

echo -e "  ${BOLD}Quick start:${NC}"
echo ""
echo -e "  ${CYAN}# Download a model${NC}"
echo "  swiftllm download -m \"Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf\""
echo ""
echo -e "  ${CYAN}# Run inference${NC}"
echo "  swiftllm generate -m \"Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf\" -p \"Hello!\""
echo ""
echo -e "  ${CYAN}# Interactive chat${NC}"
echo "  swiftllm chat -m \"Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf\""
echo ""

if [[ -n "$MODEL_DIR" ]]; then
    echo -e "  ${BOLD}Model directory:${NC} $MODEL_DIR"
else
    echo -e "  ${BOLD}Model directory:${NC} ~/.cache/swiftllm/models"
fi

if $USE_GPU; then
    echo -e "  ${BOLD}GPU acceleration:${NC} ${GREEN}Enabled${NC} (CUDA $CUDA_VERSION)"
else
    echo -e "  ${BOLD}GPU acceleration:${NC} CPU only"
fi

echo ""

if [[ $ERRORS -gt 0 ]]; then
    warn "$ERRORS verification check(s) had warnings. See above for details."
fi
