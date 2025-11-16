#!/bin/bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "❌ uv is not installed. Installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "✅ uv detected: $(uv --version)"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "📌 Pinning Python to 3.11"
uv python pin 3.11

if [ ! -d .venv ]; then
  echo "🔧 Creating virtual environment"
  uv venv
fi

echo "📦 Installing all dependencies"
uv sync --all-groups

echo "🪝 Installing pre-commit hooks"
uv run pre-commit install

cat <<'MSG'
✅ Setup complete!
To activate manually:
  source .venv/bin/activate
Use uv to run commands, e.g.:
  uv run pytest
MSG
