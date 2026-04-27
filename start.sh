#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-human}"

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  echo "Usage: ./start.sh [human|web|ai]"
  echo ""
  echo "Modes:"
  echo "  human  Start local human-playable snake"
  echo "  web    Start Flask web app at http://localhost:8080"
  echo "  ai     Start local AI training loop"
  exit 0
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN is not installed or not in PATH."
  exit 1
fi

if [[ -d "$VENV_DIR" && ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Detected incomplete virtual environment at .venv. Recreating..."
  rm -rf "$VENV_DIR"
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Creating virtual environment in .venv ..."
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    echo ""
    echo "Failed to create venv. Install the system venv package and retry:"
    echo "  sudo apt install python3-venv"
    exit 1
  fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip >/dev/null
python -m pip install -r "$ROOT_DIR/requirements.txt"

case "$MODE" in
  human)
    echo "Starting human mode..."
    exec python "$ROOT_DIR/snake_game_human.py"
    ;;
  web)
    echo "Starting web mode at http://localhost:8080 ..."
    exec python "$ROOT_DIR/web_app.py"
    ;;
  ai)
    echo "Starting AI training mode..."
    if python -c "import tkinter" >/dev/null 2>&1; then
      export MPLBACKEND=TkAgg
    else
      echo ""
      echo "Note: tkinter is not installed, so matplotlib chart windows (scores/Q-values)"
      echo "will not open. The AI game window still runs."
      echo "Install GUI support with:"
      echo "  sudo apt install python3-tk"
      echo ""
    fi
    exec python "$ROOT_DIR/agent.py"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: ./start.sh [human|web|ai]"
    exit 1
    ;;
esac
