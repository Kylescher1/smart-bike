# How to Run config_setup.py

## Step 1: Activate the Virtual Environment

From the project root directory (`/home/radxa/smart-bike`), run:

```bash
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your command prompt, indicating the virtual environment is active.

## Step 2: Install Required Dependencies

The script needs two additional packages that aren't in the main requirements:

```bash
pip install dill numpy-quaternion
```

**Note:** If you encounter SSL errors, try one of these alternatives:

```bash
# Option 1: Use trusted hosts
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org dill numpy-quaternion

# Option 2: Disable SSL verification (use with caution)
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org dill numpy-quaternion
```

## Step 3: Run the Script

With the virtual environment activated, run:

```bash
python config_setup.py
```

Or:

```bash
python3 config_setup.py
```

## What the Script Does

The script:
1. Creates a configuration dictionary for sensors (LIDAR, IMU, etc.)
2. Validates that all required config fields are present
3. Saves the configuration to `config.dill` using dill serialization
4. Loads it back and displays the configuration

## Troubleshooting

### ModuleNotFoundError: No module named 'dill'
- Make sure the virtual environment is activated (you should see `(venv)` in your prompt)
- Install dill: `pip install dill`

### ModuleNotFoundError: No module named 'quaternion'
- Install numpy-quaternion: `pip install numpy-quaternion`
- Note: The package name is `numpy-quaternion` but you import it as `quaternion`

### SSL Errors
- Check your network connection
- Try using `--trusted-host` flags as shown above
- If on a corporate network, you may need to configure proxy settings

## Deactivate When Done

When finished, deactivate the virtual environment:

```bash
deactivate
```

