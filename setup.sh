#!/bin/bash

# Feature Discovery System Setup Script

echo "Setting up Feature Discovery System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install pytest for testing
echo "Installing development dependencies..."
pip install pytest pytest-cov

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your API key!"
    echo ""
else
    echo ".env file already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit .env and add your API key (ANTHROPIC_API_KEY or OPENAI_API_KEY)"
echo "3. Run the example: python examples/basic_example.py"
echo "4. Run tests: pytest tests/"
echo ""
