#!/bin/bash
# One-command setup script for mini-researcher-agent

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Mini Researcher Agent - Automated Setup                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✓ UV is already installed"
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
uv sync

# Install Tectonic
echo ""
echo "📦 Installing Tectonic LaTeX engine..."
uv run install-tectonic

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✓ Installation Complete!                                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and add your API keys"
echo "2. Run an example: python example.py 5"
echo ""
echo "Important: Make sure ~/.local/bin is in your PATH"
echo "Add this to your ~/.bashrc or ~/.zshrc:"
echo '  export PATH="$HOME/.local/bin:$PATH"'
echo ""
