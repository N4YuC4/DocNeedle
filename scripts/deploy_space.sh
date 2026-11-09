#!/usr/bin/env bash
set -e

SPACE_ID="${1:-N4YuC4/docneedle}"
COMMIT_MSG="${2:-Deploy DocNeedle updates}"

export PATH="$HOME/.local/bin:$PATH"

echo "==> Checking if Space exists: $SPACE_ID..."
if ! hf spaces info "$SPACE_ID" >/dev/null 2>&1; then
  echo "==> Space '$SPACE_ID' not found. Creating brand new Space with ZeroGPU..."
  hf repos create "$SPACE_ID" --type space --space-sdk gradio --flavor zero-a10g --public -e GRADIO_SSR_MODE=false --exist-ok
  echo "==> Space repository created successfully!"
else
  echo "==> Space '$SPACE_ID' already exists."
fi

echo "==> Deploying project files to $SPACE_ID..."

README_FILE="README.md"
README_BAK="${README_FILE}.hf_bak"

restore_readme() {
  if [ -f "$README_BAK" ]; then
    echo "==> Restoring clean GitHub README.md..."
    mv -f "$README_BAK" "$README_FILE"
  fi
}
trap restore_readme EXIT INT TERM

echo "==> Preparing Hugging Face Spaces YAML frontmatter..."
cp -p "$README_FILE" "$README_BAK"

cat << 'EOF' > "$README_FILE"
---
title: DocNeedle
emoji: 📑
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.0.0
python_version: 3.14
app_file: app.py
pinned: false
license: apache-2.0
---

EOF
cat "$README_BAK" >> "$README_FILE"

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
  --exclude ".python-version" \
  --exclude "**/.fuse_hidden*" \
  --exclude "README.md.hf_bak" \
  --commit-message "$COMMIT_MSG"

restore_readme
trap - EXIT INT TERM

echo "==> Upload finished. Waiting for Space to build and reach RUNNING status..."
hf spaces wait "$SPACE_ID" --timeout 5m
echo "==> Successfully deployed! Space is live at: https://huggingface.co/spaces/$SPACE_ID"
