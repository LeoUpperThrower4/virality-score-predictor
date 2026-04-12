#!/usr/bin/env bash
set -euo pipefail

echo "╔══════════════════════════════════════════════════╗"
echo "║      Virality Score Predictor — Setup            ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Check Python ──
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found. Install it first."
    exit 1
fi

PYTHON=$(command -v python3)
echo "✓ Python: $($PYTHON --version)"

# ── Check ffmpeg ──
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠ ffmpeg not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo "❌ Please install ffmpeg manually: https://ffmpeg.org/download.html"
        exit 1
    fi
fi
echo "✓ ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# ── Create virtual environment ──
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment: $VENV_DIR"

# ── Install dependencies ──
echo ""
echo "Installing Python dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r backend/requirements.txt

# ── HuggingFace token check ──
echo ""
if [ -z "${HF_TOKEN:-}" ]; then
    echo "⚠ HF_TOKEN not set."
    echo "  You need a HuggingFace token to download the TRIBE v2 model."
    echo "  1. Create a token at https://huggingface.co/settings/tokens"
    echo "  2. Accept the LLaMA 3.2 license at https://huggingface.co/meta-llama/Llama-3.2-3B"
    echo "  3. Export it: export HF_TOKEN=your_token_here"
    echo ""
else
    echo "✓ HF_TOKEN is set"
fi

# ── Done ──
echo ""
echo "══════════════════════════════════════════════════"
echo "  Setup complete! Start the server with:"
echo ""
echo "  source .venv/bin/activate"
echo "  cd backend && python main.py"
echo ""
echo "  Then open http://localhost:8000 in your browser."
echo "══════════════════════════════════════════════════"
