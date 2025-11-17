# Config Merge Tool - Usage Guide

## Overview

`merge_config_safely.py` allows you to modify `config_setup.py` and update `config.dill` without losing calibration data from `calibrate_maps.py`.

## Why Use This Tool?

When you run `config_setup.py` directly, it overwrites `config.dill` completely, which means you lose:
- Camera calibration maps (`map_x`, `map_y`)
- Camera intrinsics (`K`, `D`, `R`, `P`, `newK`)
- Stereo calibration (`Q`, `stereo` section)
- All other calibration data

This tool intelligently merges new config sections while preserving all calibration data.

## Usage

### Basic Usage

```bash
python "src/Debug Tools/merge_config_safely.py"
```

### Workflow Example: Adding YOLO Config

1. **Edit `config_setup.py`** to add your YOLO section:
   ```python
   config = {
       "camera": {
           # ... existing camera config ...
       },
       "yolo": {
           "model_path": "yolo/models/yolo11n.pt",
           "confidence": 0.5,
           # ... other YOLO settings ...
       }
   }
   ```

2. **Run the merge tool**:
   ```bash
   python "src/Debug Tools/merge_config_safely.py"
   ```

3. **Result**: 
   - YOLO section is added to `config.dill`
   - Camera calibration data is preserved
   - Camera config is updated with any new parameters from `config_setup.py`

## What Gets Preserved

The tool preserves these calibration keys:

**Camera Left/Right:**
- `map_x`, `map_y` (rectification maps)
- `K`, `D`, `R`, `P`, `newK` (camera matrices)
- `rms` (calibration error)
- `map_size` (map dimensions)

**Camera Level:**
- `stereo` (stereo calibration data)
- `resolution` (calibration resolution)
- `Q` (disparity-to-depth mapping matrix)

## What Gets Updated

- New sections from `config_setup.py` are added
- Non-calibration camera parameters are updated
- Missing required fields are validated

## Example Output

```
======================================================================
CONFIG MERGE TOOL - Safe Config Updates
======================================================================
📁 Working directory: C:\smart-bike

📂 Loading existing config from config.dill...
✅ Loaded existing config with sections: ['camera']

📝 Executing config_setup.py to get new config structure...
✅ Got new config structure with sections: ['camera', 'yolo']

======================================================================
MERGING CONFIGS...
======================================================================
🔧 Merging camera section...
🔒 Preserving calibration data from existing config...
   ✓ Preserved camera.Q
   ✓ Preserved camera.left.map_x
   ✓ Preserved camera.left.map_y
   ✓ Preserved camera.left.K
   ✓ Preserved camera.left.D
   ✓ Preserved camera.right.map_x
   ✓ Preserved camera.right.map_y

➕ Adding new section 'yolo'...

======================================================================
VALIDATING CONFIG...
======================================================================
✅ Config validation passed!

======================================================================
SAVING CONFIG...
======================================================================
✅ Saved merged config to config.dill

======================================================================
CONFIG SUMMARY
======================================================================
Device: camera
  - Sections: ['left', 'right', 'who_to_run', ...]
  - Left calibration keys: ['map_x', 'map_y', 'K', 'D', 'R', 'P']
  - Right calibration keys: ['map_x', 'map_y', 'K', 'D', 'R', 'P']
Device: yolo | Type: dict

======================================================================
✅ CONFIG MERGE COMPLETE!
======================================================================
```

## Tips

- Always backup `config.dill` before running if you're unsure
- The tool validates required fields, so you'll get clear error messages if something is missing
- You can modify `config_setup.py` to only include the sections you want to update (e.g., just `{'camera': {...}}` or `{'yolo': {...}}`)

