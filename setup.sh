#!/bin/bash
# Quick setup script for Qode Market Intelligence

echo "🚀 Qode Market Intelligence - Quick Setup"
echo "=========================================="

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.12+"
    exit 1
fi

echo "✓ Python found: $(python --version)"

# Create venv
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

# Install
echo "📥 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Create directories
mkdir -p data output logs output/plots

# Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
python main.py --mode nitter --target 2000 --browser chrome --headless
# In setup.sh, after the other nltk downloads:
python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run: python main.py"