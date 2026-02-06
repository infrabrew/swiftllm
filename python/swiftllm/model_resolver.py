"""SwiftLLM Model Resolver

Resolves model identifiers to local filesystem paths.
Handles downloading from HuggingFace Hub when needed.

Supported model formats:
  - Local path: /path/to/model or ./model
  - HuggingFace repo: org/model-name (downloads full repo)
  - HuggingFace file URL: https://huggingface.co/org/repo/blob/main/file.gguf
  - HuggingFace repo + filename: org/repo:file.gguf (downloads single file)
"""

import os
import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "swiftllm", "models")
ENV_MODEL_DIR = "SWIFTLLM_MODEL_DIR"
ENV_HF_TOKEN = "HF_TOKEN"

# Matches: https://huggingface.co/org/repo/blob/revision/filename
# or:      https://huggingface.co/org/repo/resolve/revision/filename
HF_URL_PATTERN = re.compile(
    r"^https?://huggingface\.co/"
    r"(?P<repo>[^/]+/[^/]+)/"
    r"(?:blob|resolve)/"
    r"(?P<revision>[^/]+)/"
    r"(?P<filename>.+)$"
)


def is_local_path(model: str) -> bool:
    """Determine if a model string is a local filesystem path or a HuggingFace ID."""
    if model.startswith(("http://", "https://")):
        return False

    if model.startswith(("/", "./", "../", "~/")):
        return True

    if len(model) >= 2 and model[1] == ":":
        return True

    expanded = os.path.expanduser(model)
    if os.path.exists(expanded):
        return True

    return False


def parse_hf_url(url: str) -> Optional[Tuple[str, str, str]]:
    """Parse a HuggingFace URL into (repo_id, filename, revision).

    Args:
        url: Full HuggingFace URL like
             https://huggingface.co/org/repo/blob/main/file.gguf

    Returns:
        Tuple of (repo_id, filename, revision) or None if not a valid HF URL.
    """
    match = HF_URL_PATTERN.match(url)
    if match:
        return match.group("repo"), match.group("filename"), match.group("revision")
    return None


def parse_repo_filename(model: str) -> Optional[Tuple[str, str]]:
    """Parse org/repo:filename format into (repo_id, filename).

    Args:
        model: String like 'org/repo:file.gguf'

    Returns:
        Tuple of (repo_id, filename) or None if not in that format.
    """
    if ":" not in model:
        return None

    parts = model.rsplit(":", 1)
    repo_id = parts[0]
    filename = parts[1]

    # Validate it looks like a repo ID (org/name)
    if "/" not in repo_id or filename == "":
        return None

    return repo_id, filename


def resolve_download_dir(download_dir: Optional[str] = None) -> str:
    """Determine the directory for storing downloaded models.

    Priority: explicit arg > SWIFTLLM_MODEL_DIR env var > ~/.cache/swiftllm/models
    """
    if download_dir:
        return os.path.abspath(os.path.expanduser(download_dir))

    env_dir = os.environ.get(ENV_MODEL_DIR)
    if env_dir:
        return os.path.abspath(os.path.expanduser(env_dir))

    return os.path.abspath(DEFAULT_CACHE_DIR)


def _get_hf_hub():
    """Import and return huggingface_hub, raising a clear error if missing."""
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models from HuggingFace. "
            "Install with: pip install huggingface-hub"
        )


def _resolve_token(token: Optional[str] = None) -> Optional[str]:
    """Resolve HF token from arg or environment."""
    if token is not None:
        return token
    return os.environ.get(ENV_HF_TOKEN)


def download_file(
    repo_id: str,
    filename: str,
    download_dir: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Download a single file from a HuggingFace repo.

    Args:
        repo_id: HuggingFace repo ID (e.g., 'org/model-name').
        filename: File to download (e.g., 'model.q4_k_m.gguf').
        download_dir: Directory to store the download.
        token: HuggingFace API token.
        revision: Git revision (branch, tag, commit).

    Returns:
        Absolute path to the downloaded file.
    """
    hf_hub = _get_hf_hub()
    cache_dir = resolve_download_dir(download_dir)
    token = _resolve_token(token)

    logger.info(
        "Downloading file '%s' from '%s' to %s", filename, repo_id, cache_dir
    )

    local_path = hf_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        token=token,
        revision=revision,
    )

    logger.info("File downloaded/cached at: %s", local_path)
    return local_path


def download_repo(
    repo_id: str,
    download_dir: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Download a full HuggingFace repo.

    Args:
        repo_id: HuggingFace repo ID (e.g., 'org/model-name').
        download_dir: Directory to store the download.
        token: HuggingFace API token.
        revision: Git revision (branch, tag, commit).

    Returns:
        Absolute path to the downloaded model directory.
    """
    hf_hub = _get_hf_hub()
    cache_dir = resolve_download_dir(download_dir)
    token = _resolve_token(token)

    logger.info("Downloading repo '%s' to %s", repo_id, cache_dir)

    local_path = hf_hub.snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        token=token,
        revision=revision,
    )

    logger.info("Repo downloaded/cached at: %s", local_path)
    return local_path


def resolve_model(
    model: str,
    download_dir: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Resolve a model identifier to a local filesystem path.

    Supports:
      - Local paths: /path/to/model, ./model, ~/models/my-model
      - HuggingFace repo ID: meta-llama/Llama-2-7b-hf (downloads full repo)
      - HuggingFace URL: https://huggingface.co/org/repo/blob/main/file.gguf
      - Repo + filename: org/repo:file.gguf (downloads single file)

    Args:
        model: Model identifier (local path, HF repo ID, HF URL, or repo:filename).
        download_dir: Directory for storing downloaded models.
        token: HuggingFace API token for gated models. Falls back to HF_TOKEN env var.
        revision: Git revision (branch, tag, commit hash) to download.

    Returns:
        Absolute path to the model file or directory on the local filesystem.
    """
    if not model:
        raise ValueError("Model path or HuggingFace model ID must not be empty")

    # 1. Local path
    if is_local_path(model):
        expanded = os.path.abspath(os.path.expanduser(model))
        if not os.path.exists(expanded):
            raise FileNotFoundError(f"Model path does not exist: {expanded}")
        logger.info("Using local model path: %s", expanded)
        return expanded

    # 2. Full HuggingFace URL (single file)
    #    e.g. https://huggingface.co/org/repo/blob/main/file.gguf
    parsed_url = parse_hf_url(model)
    if parsed_url is not None:
        repo_id, filename, url_revision = parsed_url
        return download_file(
            repo_id=repo_id,
            filename=filename,
            download_dir=download_dir,
            token=token,
            revision=revision or url_revision,
        )

    # 3. Repo + filename shorthand (single file)
    #    e.g. org/repo:file.gguf
    parsed_repo_file = parse_repo_filename(model)
    if parsed_repo_file is not None:
        repo_id, filename = parsed_repo_file
        return download_file(
            repo_id=repo_id,
            filename=filename,
            download_dir=download_dir,
            token=token,
            revision=revision,
        )

    # 4. Plain HuggingFace repo ID (full repo)
    #    e.g. meta-llama/Llama-2-7b-hf
    return download_repo(
        repo_id=model,
        download_dir=download_dir,
        token=token,
        revision=revision,
    )
