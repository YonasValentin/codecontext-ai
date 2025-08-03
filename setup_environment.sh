#!/bin/bash
# CodeContext AI™ - Environment Setup Script
# Sets up complete development environment with all dependencies

set -e

echo "=========================================="
echo "CodeContext AI™ Environment Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python version: $(python3 --version)"
else
    echo "❌ Python 3.8+ required. Found: $(python3 --version)"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "🔧 Installing core ML dependencies..."
pip install torch transformers pyyaml

# Install additional dependencies if requirements file exists
if [ -f "requirements.txt" ]; then
    echo "🔧 Installing additional dependencies..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found - installing minimal dependencies"
fi

# Install development dependencies
if [ -f "requirements-dev.txt" ]; then
    echo "🔧 Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Install package in development mode
echo "🔧 Installing CodeContext AI™ in development mode..."
pip install -e .

# Test installation
echo "🧪 Testing installation..."
python -c "
try:
    from codecontext_ai.advisory import AdvisoryEngine, AdvisoryType
    from codecontext_ai.inference import InferenceEngine
    import torch, transformers, yaml
    print('✅ All imports successful')
    print(f'   PyTorch: {torch.__version__}')
    print(f'   Transformers: {transformers.__version__}')
except Exception as e:
    print(f'❌ Import test failed: {e}')
    exit(1)
"

# Create necessary directories
echo "🔧 Creating necessary directories..."
mkdir -p data models logs

echo ""
echo "=========================================="
echo "🎉 Setup Complete!"
echo "=========================================="
echo ""
echo "Your CodeContext AI™ environment is ready!"
echo ""
echo "📋 Available commands:"
echo "   source venv/bin/activate     # Activate environment"
echo "   make train MODEL=advisory    # Train advisory model"
echo "   python demo_advisory.py     # Run demo"
echo ""
echo "📚 Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run demo: python demo_advisory.py"
echo "3. Train models: make train MODEL=advisory"
echo ""
echo "For more information, see README.md"