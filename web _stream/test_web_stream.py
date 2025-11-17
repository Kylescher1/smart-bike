#!/usr/bin/env python3
"""
Quick test to verify web_stream.py setup is correct.
This doesn't require cameras to be connected.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import flask
        print("  ✅ Flask installed")
    except ImportError:
        print("  ❌ Flask not found - run: pip install -r requirements_web.txt")
        return False
    
    try:
        import cv2
        print("  ✅ OpenCV installed")
    except ImportError:
        print("  ❌ OpenCV not found - run: pip install -r requirements_web.txt")
        return False
    
    try:
        import numpy
        print("  ✅ NumPy installed")
    except ImportError:
        print("  ❌ NumPy not found - run: pip install -r requirements_web.txt")
        return False
    
    try:
        import dill
        print("  ✅ Dill installed")
    except ImportError:
        print("  ❌ Dill not found - run: pip install -r requirements_web.txt")
        return False
    
    return True

def test_files():
    """Test that required files exist."""
    print("\nTesting files...")
    
    files = {
        "web_stream.py": "Main server file",
        "templates/index.html": "Web interface",
        "config.dill": "Camera configuration",
    }
    
    all_exist = True
    for file, description in files.items():
        if Path(file).exists():
            print(f"  ✅ {file} - {description}")
        else:
            print(f"  ❌ {file} - {description} (missing)")
            if file == "config.dill":
                print("     Run config_setup.py to create this file")
            all_exist = False
    
    return all_exist

def test_flask_app():
    """Test that Flask app can be created."""
    print("\nTesting Flask app...")
    try:
        from web_stream import app
        print("  ✅ Flask app created successfully")
        
        # Test that routes exist
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = ['/', '/video_feed', '/set_mode', '/get_parameters', 
                          '/update_parameter', '/save_parameters']
        
        for route in expected_routes:
            if route in routes:
                print(f"  ✅ Route {route} exists")
            else:
                print(f"  ❌ Route {route} missing")
        
        return True
    except Exception as e:
        print(f"  ❌ Error creating Flask app: {e}")
        return False

def main():
    print("="*60)
    print("WEB STREAM TEST")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Files", test_files()))
    results.append(("Flask App", test_flask_app()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = all(result[1] for result in results)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! You're ready to run:")
        print("   python web_stream.py")
    else:
        print("⚠️  Some tests failed. Fix the issues above before running.")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

