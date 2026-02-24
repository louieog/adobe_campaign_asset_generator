#!/bin/bash
# Campaign Asset Generator — Setup & Launch
# Usage: ./start.sh

set -e

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║   Campaign Asset Generator                        ║"
echo "║   Claude Orchestration + Nano Banana Pro Images   ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3.10+ required. Install from python.org"
  exit 1
fi

# Check API keys
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "❌ ANTHROPIC_API_KEY not set."
  echo "   export ANTHROPIC_API_KEY=sk-ant-..."
  exit 1
fi

if [ -z "$GEMINI_API_KEY" ]; then
  echo "❌ GEMINI_API_KEY not set."
  echo "   export GEMINI_API_KEY=AIza..."
  echo "   Get your key: https://aistudio.google.com/apikey"
  exit 1
fi

echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "✅ Setup complete."
echo ""
echo "   ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:0:12}..."
echo "   GEMINI_API_KEY:    ${GEMINI_API_KEY:0:12}..."
echo "   Image model:       gemini-3-pro-image-preview (Nano Banana Pro)"
echo ""
echo "🚀 Launching at http://localhost:7860"
echo "   Press Ctrl+C to stop."
echo ""

python3 app.py
