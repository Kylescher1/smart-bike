"""
Config Merge Tool - Safely merge config_setup.py changes without losing calibration data

This tool allows you to modify config_setup.py and merge new sections/changes
into config.dill without losing calibration data (maps, K, D, R, P, Q, etc.)

Usage:
    python src/Debug Tools/merge_config_safely.py

The tool will:
1. Load existing config.dill (if it exists)
2. Execute config_setup.py to get new config structure
3. Merge them intelligently, preserving calibration data
4. Save the merged config back to config.dill
"""

import dill
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, Any, Set
import importlib.util

# Paths
CONFIG_DILL_PATH = "config.dill"
CONFIG_SETUP_PATH = "config_setup.py"

# Calibration keys that should be preserved (not overwritten)
CALIBRATION_KEYS_LEFT_RIGHT: Set[str] = {
    "map_x", "map_y", "K", "D", "R", "P", "newK", "rms", "map_size"
}

CALIBRATION_KEYS_CAMERA: Set[str] = {
    "stereo", "resolution", "Q"
}


def load_existing_config() -> Dict[str, Any]:
    """Load existing config.dill if it exists, otherwise return empty dict."""
    if os.path.exists(CONFIG_DILL_PATH):
        print(f"📂 Loading existing config from {CONFIG_DILL_PATH}...")
        try:
            with open(CONFIG_DILL_PATH, "rb") as f:
                config = dill.load(f)
            print(f"✅ Loaded existing config with sections: {list(config.keys())}")
            return config
        except Exception as e:
            print(f"⚠️  Warning: Could not load existing config: {e}")
            print("   Starting with empty config...")
            return {}
    else:
        print(f"📝 No existing {CONFIG_DILL_PATH} found. Starting fresh...")
        return {}


def execute_config_setup() -> Dict[str, Any]:
    """Execute config_setup.py and capture the config dictionary it creates."""
    config_setup_abs = os.path.abspath(CONFIG_SETUP_PATH)
    
    if not os.path.exists(config_setup_abs):
        raise FileNotFoundError(f"Could not find {CONFIG_SETUP_PATH} at {config_setup_abs}")
    
    print(f"\n📝 Executing {CONFIG_SETUP_PATH} to get new config structure...")
    
    # Add current directory to path so imports work
    config_setup_dir = os.path.dirname(config_setup_abs)
    if config_setup_dir not in sys.path:
        sys.path.insert(0, config_setup_dir)
    
    # Load config_setup.py as a module
    spec = importlib.util.spec_from_file_location("config_setup", config_setup_abs)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {CONFIG_SETUP_PATH}")
    
    module = importlib.util.module_from_spec(spec)
    
    # Execute the module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error executing {CONFIG_SETUP_PATH}: {e}")
    
    # Get the config dictionary from the module
    if not hasattr(module, 'config'):
        raise RuntimeError(f"{CONFIG_SETUP_PATH} does not define a 'config' variable")
    
    new_config = module.config
    print(f"✅ Got new config structure with sections: {list(new_config.keys())}")
    return new_config


def has_calibration_data(camera_config: Dict[str, Any]) -> bool:
    """Check if camera config contains calibration data."""
    if not isinstance(camera_config, dict):
        return False
    
    # Check for calibration keys in left/right
    for side in ["left", "right"]:
        if side in camera_config:
            side_config = camera_config[side]
            if isinstance(side_config, dict):
                if any(key in side_config for key in CALIBRATION_KEYS_LEFT_RIGHT):
                    # Check if values are not None
                    for key in CALIBRATION_KEYS_LEFT_RIGHT:
                        if key in side_config and side_config[key] is not None:
                            return True
    
    # Check for camera-level calibration keys
    for key in CALIBRATION_KEYS_CAMERA:
        if key in camera_config and camera_config[key] is not None:
            return True
    
    return False


def merge_camera_configs(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Intelligently merge camera configs, preserving calibration data.
    
    Strategy:
    - If existing has calibration data, preserve it
    - Merge non-calibration keys from new config
    - For left/right: preserve calibration keys, merge other keys
    """
    merged = new.copy()
    
    # If existing has calibration data, we need to preserve it
    if has_calibration_data(existing):
        print("🔒 Preserving calibration data from existing config...")
        
        # Preserve camera-level calibration keys
        for key in CALIBRATION_KEYS_CAMERA:
            if key in existing:
                merged[key] = existing[key]
                print(f"   ✓ Preserved camera.{key}")
        
        # Merge left camera config
        if "left" in existing and isinstance(existing["left"], dict):
            if "left" not in merged:
                merged["left"] = {}
            elif not isinstance(merged["left"], dict):
                merged["left"] = {}
            
            # Preserve calibration keys from existing
            for key in CALIBRATION_KEYS_LEFT_RIGHT:
                if key in existing["left"]:
                    merged["left"][key] = existing["left"][key]
                    print(f"   ✓ Preserved camera.left.{key}")
            
            # Merge non-calibration keys from new config
            for key, value in new.get("left", {}).items():
                if key not in CALIBRATION_KEYS_LEFT_RIGHT:
                    merged["left"][key] = value
        
        # Merge right camera config
        if "right" in existing and isinstance(existing["right"], dict):
            if "right" not in merged:
                merged["right"] = {}
            elif not isinstance(merged["right"], dict):
                merged["right"] = {}
            
            # Preserve calibration keys from existing
            for key in CALIBRATION_KEYS_LEFT_RIGHT:
                if key in existing["right"]:
                    merged["right"][key] = existing["right"][key]
                    print(f"   ✓ Preserved camera.right.{key}")
            
            # Merge non-calibration keys from new config
            for key, value in new.get("right", {}).items():
                if key not in CALIBRATION_KEYS_LEFT_RIGHT:
                    merged["right"][key] = value
    else:
        print("ℹ️  No calibration data found in existing config. Using new config as-is.")
    
    return merged


def merge_configs(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge existing and new configs intelligently.
    
    Strategy:
    - For 'camera' section: use special merge that preserves calibration
    - For other sections: new config takes precedence (adds new sections, updates existing)
    """
    merged = existing.copy()
    
    # Handle camera section specially
    if "camera" in new:
        if "camera" in existing:
            print("\n🔧 Merging camera section...")
            merged["camera"] = merge_camera_configs(existing["camera"], new["camera"])
        else:
            print("\n➕ Adding new camera section...")
            merged["camera"] = new["camera"]
    
    # For all other sections, new config takes precedence
    for section_name, section_config in new.items():
        if section_name != "camera":
            if section_name in merged:
                print(f"\n🔄 Updating section '{section_name}'...")
            else:
                print(f"\n➕ Adding new section '{section_name}'...")
            merged[section_name] = section_config
    
    return merged


def validate_config(config: Dict[str, Any]) -> None:
    """Validate that config has required fields."""
    required_keys = {"who_to_run", "port", "position", "z_direction"}
    
    for name, cfg in config.items():
        # Handle nested structure for camera
        if name == "camera" and isinstance(cfg, dict) and "left" in cfg and "right" in cfg:
            # Check camera.left
            left_cfg = cfg["left"]
            left_missing = {"port", "position", "z_direction"} - ({"port", "position", "z_direction"} & left_cfg.keys())
            if left_missing:
                raise KeyError(f"{name}.left is missing required config items: {left_missing}")
            # Check camera.right
            right_cfg = cfg["right"]
            right_missing = {"port", "position", "z_direction"} - ({"port", "position", "z_direction"} & right_cfg.keys())
            if right_missing:
                raise KeyError(f"{name}.right is missing required config items: {right_missing}")
            # Check who_to_run field at camera level
            if "who_to_run" not in cfg:
                raise KeyError(f"{name} is missing required config item: who_to_run")
        else:
            # Standard flat structure
            missing = required_keys - (required_keys & cfg.keys())
            if missing:
                raise KeyError(f"{name} is missing required config items: {missing}")


def main():
    """Main function to merge configs safely."""
    print("=" * 70)
    print("CONFIG MERGE TOOL - Safe Config Updates")
    print("=" * 70)
    
    # Change to project root directory
    script_dir = Path(__file__).parent.parent.parent
    os.chdir(script_dir)
    print(f"📁 Working directory: {os.getcwd()}\n")
    
    try:
        # Load existing config
        existing_config = load_existing_config()
        
        # Execute config_setup.py to get new config
        new_config = execute_config_setup()
        
        # Merge configs
        print("\n" + "=" * 70)
        print("MERGING CONFIGS...")
        print("=" * 70)
        merged_config = merge_configs(existing_config, new_config)
        
        # Validate merged config
        print("\n" + "=" * 70)
        print("VALIDATING CONFIG...")
        print("=" * 70)
        validate_config(merged_config)
        print("✅ Config validation passed!")
        
        # Save merged config
        print("\n" + "=" * 70)
        print("SAVING CONFIG...")
        print("=" * 70)
        with open(CONFIG_DILL_PATH, "wb") as f:
            dill.dump(merged_config, f)
        print(f"✅ Saved merged config to {CONFIG_DILL_PATH}")
        
        # Show summary
        print("\n" + "=" * 70)
        print("CONFIG SUMMARY")
        print("=" * 70)
        for k, v in merged_config.items():
            if k == "camera" and isinstance(v, dict):
                print(f"Device: {k}")
                print(f"  - Sections: {list(v.keys())}")
                if "left" in v and isinstance(v["left"], dict):
                    left_keys = list(v["left"].keys())
                    calib_keys = [k for k in left_keys if k in CALIBRATION_KEYS_LEFT_RIGHT]
                    if calib_keys:
                        print(f"  - Left calibration keys: {calib_keys}")
                if "right" in v and isinstance(v["right"], dict):
                    right_keys = list(v["right"].keys())
                    calib_keys = [k for k in right_keys if k in CALIBRATION_KEYS_LEFT_RIGHT]
                    if calib_keys:
                        print(f"  - Right calibration keys: {calib_keys}")
            else:
                print(f"Device: {k} | Type: {type(v).__name__}")
        
        print("\n" + "=" * 70)
        print("✅ CONFIG MERGE COMPLETE!")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

