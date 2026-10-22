#!/usr/bin/env bash
set -e

SPACE_ID="${1:-N4YuC4/docneedle}"
COMMIT_MSG="${2:-Deploy DocNeedle updates}"

export PATH="$HOME/.local/bin:$PATH"

echo "==> Checking if Space exists: $SPACE_ID..."
if ! hf spaces info "$SPACE_ID" >/dev/null 2>&1; then
  echo "==> Space '$SPACE_ID' not found. Creating brand new Space with ZeroGPU..."
  hf repos create "$SPACE_ID" --type space --sdk gradio --flavor zero-a10g --public -e GRADIO_SSR_MODE=false
  echo "==> Space repository created successfully!"
else
  echo "==> Space '$SPACE_ID' already exists."
fi

echo "==> Deploying project files to $SPACE_ID..."

hf upload "$SPACE_ID" . . --repo-type space \
  --exclude ".git/**" \
  --exclude ".venv/**" \
  --exclude "models/**" \
  --exclude ".pytest_cache/**" \
  --exclude ".ruff_cache/**" \
  --exclude "**/__pycache__/**" \
  --exclude "tests/**" \
  --exclude ".env" \
  --exclude "*.pyc" \
  --commit-message "$COMMIT_MSG"

echo "==> Upload finished. Waiting for Space to build and reach RUNNING status..."
hf spaces wait "$SPACE_ID" --timeout 5m
echo "==> Successfully deployed! Space is live at: https://huggingface.co/spaces/$SPACE_ID"
