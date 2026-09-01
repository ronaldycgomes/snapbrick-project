#!/usr/bin/env bash
set -euo pipefail

# Directory where the script is located (data-pipeline)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/ldraw_lib"
ZIP_URL="https://library.ldraw.org/library/updates/complete.zip"
ZIP_FILE="${SCRIPT_DIR}/complete.zip"

echo "=== LDraw 3D Catalog Downloader ==="
echo "Downloading complete.zip silently..."
mkdir -p "${TARGET_DIR}"

# Silent download with progress fail-safe
if command -v curl >/dev/null 2>&1; then
    curl -sSL -o "${ZIP_FILE}" "${ZIP_URL}"
elif command -v wget >/dev/null 2>&1; then
    wget -q -O "${ZIP_FILE}" "${ZIP_URL}"
else
    echo "Error: Neither curl nor wget is installed." >&2
    exit 1
fi

echo "Extracting catalog to ${TARGET_DIR}..."
unzip -q -o "${ZIP_FILE}" -d "${TARGET_DIR}"

echo "Removing zip archive..."
rm -f "${ZIP_FILE}"

echo "LDraw catalog downloaded and extracted successfully to ${TARGET_DIR}."
