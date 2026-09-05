#!/usr/bin/env bash
# Serve the website locally. An HTTP server is required for the model worker
# and sibling model assets.
set -euo pipefail
PORT="${1:-8000}"
cd "$(dirname "$0")"
echo "Website at http://localhost:${PORT}"
python3 -m http.server "${PORT}"
