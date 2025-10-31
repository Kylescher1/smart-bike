# Virtual Environment Setup

This project uses a Python virtual environment to manage dependencies.

## Initial Setup

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   If you encounter SSL errors, try:
   ```bash
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
   ```

## Daily Usage

**Activate the virtual environment before running any Python scripts:**
```bash
source venv/bin/activate
```

You'll see `(venv)` in your terminal prompt when it's activated.

**Deactivate when done:**
```bash
deactivate
```

## Running Scripts

With the virtual environment activated, you can run scripts normally:
```bash
python main.py
python -m src.hal.cam.depth_tuner
# etc.
```

## Recreating the Virtual Environment

If you need to recreate the virtual environment:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

