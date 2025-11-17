#!/bin/bash
# Script to activate venv and run config_setup.py

cd "$(dirname "$0")"

echo "Activating virtual environment..."
source venv/bin/activate

echo "Checking for required packages..."
python3 -c "import dill" 2>/dev/null || {
    echo "Installing dill..."
    pip install dill || pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org dill
}

python3 -c "import quaternion" 2>/dev/null || {
    echo "Installing numpy-quaternion..."
    pip install numpy-quaternion || pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org numpy-quaternion
}

echo ""
echo "Running config_setup.py..."
python3 config_setup.py

echo ""
echo "Done! You can deactivate the virtual environment with: deactivate"

