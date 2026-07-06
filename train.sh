#!/usr/bin/env bash
set -euo pipefail

# Build curl options. SSL certificate verification stays ON by default.
# Set DISABLE_SSL_VERIFY=1 to opt out (e.g. behind a proxy with a self-signed
# cert). WARNING: disabling verification exposes this "curl | sh" install to
# man-in-the-middle tampering -- only use it on a network you trust.
curl_opts=(-LsSf)
if [[ "${DISABLE_SSL_VERIFY:-0}" == "1" ]]; then
    echo "WARNING: SSL certificate verification disabled for curl" >&2
    curl_opts+=(-k)
fi

# Install uv if not available
if ! command -v uv &>/dev/null; then
    echo "uv not found, installing..."
    curl "${curl_opts[@]}" https://astral.sh/uv/install.sh | sh

    # Source the env so uv is available in the current shell
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv installation failed" >&2
        exit 1
    fi
    echo "uv installed successfully: $(uv --version)"
fi

# Run the training script with uv
uv run gpt.py
