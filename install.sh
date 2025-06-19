#!/bin/bash

echo "=================================================="
echo "Numerai GPU Ensemble Optimizer Installation"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

echo "Python version: $(python3 --version)"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: No NVIDIA GPU detected. The system will run on CPU (very slow)."
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support first
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Test the installation
echo "Testing installation..."
python3 test_quick.py

echo "=================================================="
echo "Installation complete!"
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To run the ensemble optimizer:"
echo "  python main_runner.py --quick-test --trials 5"
echo ""
echo "For full training:"
echo "  python main_runner.py --trials 100"
echo "==================================================" 