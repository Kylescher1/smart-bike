import cv2
import numpy as np
import dill
import sys, os
import importlib.util

config_path = r"config.dill"

def create_trackbars(window_name, disparity_params):
    """Create OpenCV trackbars for all disparity parameters."""
    
    # Stereo block matcher core parameters
    cv2.createTrackbar("minDisparity", window_name, disparity_params["minDisparity"], 100, lambda x: None)
    cv2.createTrackbar("numDisparitiesK", window_name, disparity_params["numDisparitiesK"], 16, lambda x: None)
    cv2.createTrackbar("numDisparities", window_name, disparity_params["numDisparities"], 256, lambda x: None)
    cv2.createTrackbar("blockSize", window_name, max(5, disparity_params["blockSize"]), 51, lambda x: None)
    cv2.createTrackbar("P1", window_name, disparity_params.get("P1", 968), 10000, lambda x: None)
    cv2.createTrackbar("P2", window_name, disparity_params.get("P2", 3872), 50000, lambda x: None)
    cv2.createTrackbar("preFilterCap", window_name, disparity_params["preFilterCap"], 100, lambda x: None)
    cv2.createTrackbar("uniquenessRatio", window_name, disparity_params["uniquenessRatio"], 100, lambda x: None)
    cv2.createTrackbar("speckleWindowSize", window_name, disparity_params["speckleWindowSize"], 500, lambda x: None)
    cv2.createTrackbar("speckleRange", window_name, disparity_params["speckleRange"], 100, lambda x: None)
    cv2.createTrackbar("disp12MaxDiff", window_name, disparity_params["disp12MaxDiff"], 100, lambda x: None)
    cv2.createTrackbar("sgbmMode", window_name, disparity_params.get("sgbmMode", 2), 2, lambda x: None)  # 0=SGBM, 1=HH, 2=SGBM_3WAY
    
    # Pre-processing & scaling
    cv2.createTrackbar("medianBlurK", window_name, disparity_params["medianBlurK"], 21, lambda x: None)
    cv2.createTrackbar("downSample", window_name, disparity_params["downSample"], 100, lambda x: None)
    cv2.createTrackbar("crop", window_name, disparity_params["crop"], 300, lambda x: None)
    cv2.createTrackbar("farEnhance", window_name, disparity_params["farEnhance"], 100, lambda x: None)
    cv2.createTrackbar("nearCutoff %", window_name, disparity_params["nearCutoff"], 100, lambda x: None)
    cv2.createTrackbar("farCutoff %", window_name, disparity_params["farCutoff"], 100, lambda x: None)
    
    # Morphological filtering
    cv2.createTrackbar("useMorph", window_name, int(disparity_params["useMorph"]), 1, lambda x: None)
    cv2.createTrackbar("morphIter", window_name, disparity_params["morphIter"], 20, lambda x: None)
    
    # Bilateral smoothing
    cv2.createTrackbar("useBilateral", window_name, int(disparity_params["useBilateral"]), 1, lambda x: None)
    cv2.createTrackbar("bilateralStrength", window_name, disparity_params["bilateralStrength"], 100, lambda x: None)
    
    # Weighted least squares refinement
    cv2.createTrackbar("useWLS", window_name, int(disparity_params["useWLS"]), 1, lambda x: None)
    cv2.createTrackbar("wlsLambda", window_name, disparity_params["wlsLambda"], 10000, lambda x: None)
    cv2.createTrackbar("wlsSigma x10", window_name, int(disparity_params["wlsSigma"] * 10), 50, lambda x: None)
    
    # Object detection thresholds
    cv2.createTrackbar("objectThresholdMM", window_name, disparity_params["objectThresholdMM"], 5000, lambda x: None)
    cv2.createTrackbar("wsSigma", window_name, disparity_params["wsSigma"], 20, lambda x: None)
    cv2.createTrackbar("wsMinArea", window_name, disparity_params["wsMinArea"], 5000, lambda x: None)
    
    # Edge enhancement
    cv2.createTrackbar("edgeEqualize", window_name, int(disparity_params["edgeEqualize"]), 1, lambda x: None)
    cv2.createTrackbar("edgeBilateralD", window_name, disparity_params["edgeBilateralD"], 31, lambda x: None)
    cv2.createTrackbar("edgeBilateralSigma", window_name, disparity_params["edgeBilateralSigma"], 300, lambda x: None)
    cv2.createTrackbar("edgeCannyKLow x10", window_name, int(disparity_params["edgeCannyKLow"] * 10), 100, lambda x: None)
    cv2.createTrackbar("edgeCannyKHigh x10", window_name, int(disparity_params["edgeCannyKHigh"] * 10), 100, lambda x: None)
    cv2.createTrackbar("edgeUseScharr", window_name, int(disparity_params["edgeUseScharr"]), 1, lambda x: None)
    
    # Color segmentation
    cv2.createTrackbar("colorFocusMM", window_name, disparity_params["colorFocusMM"], 20000, lambda x: None)
    cv2.createTrackbar("colorSpanMM", window_name, disparity_params["colorSpanMM"], 30000, lambda x: None)
    cv2.createTrackbar("segMode", window_name, disparity_params["segMode"], 3, lambda x: None)
    cv2.createTrackbar("kmK", window_name, disparity_params["kmK"], 20, lambda x: None)
    cv2.createTrackbar("kmSpatialX100", window_name, disparity_params["kmSpatialX100"], 200, lambda x: None)
    cv2.createTrackbar("rgTau", window_name, disparity_params["rgTau"], 100, lambda x: None)
    cv2.createTrackbar("rgSeedStep", window_name, disparity_params["rgSeedStep"], 50, lambda x: None)

def get_trackbar_values(window_name):
    """Read all trackbar values and return updated disparity parameters."""
    params = {}
    
    # Stereo block matcher core parameters
    params["minDisparity"] = cv2.getTrackbarPos("minDisparity", window_name)
    params["numDisparitiesK"] = cv2.getTrackbarPos("numDisparitiesK", window_name)
    params["numDisparities"] = cv2.getTrackbarPos("numDisparities", window_name)
    
    # Ensure blockSize is odd and >= 5
    blockSize = cv2.getTrackbarPos("blockSize", window_name)
    params["blockSize"] = max(5, blockSize if blockSize % 2 == 1 else blockSize + 1)
    
    params["P1"] = cv2.getTrackbarPos("P1", window_name)
    params["P2"] = cv2.getTrackbarPos("P2", window_name)
    params["preFilterCap"] = cv2.getTrackbarPos("preFilterCap", window_name)
    params["uniquenessRatio"] = cv2.getTrackbarPos("uniquenessRatio", window_name)
    params["speckleWindowSize"] = cv2.getTrackbarPos("speckleWindowSize", window_name)
    params["speckleRange"] = cv2.getTrackbarPos("speckleRange", window_name)
    params["disp12MaxDiff"] = cv2.getTrackbarPos("disp12MaxDiff", window_name)
    params["sgbmMode"] = cv2.getTrackbarPos("sgbmMode", window_name)
    
    # Pre-processing & scaling
    params["medianBlurK"] = cv2.getTrackbarPos("medianBlurK", window_name)
    params["downSample"] = cv2.getTrackbarPos("downSample", window_name)
    params["crop"] = cv2.getTrackbarPos("crop", window_name)
    params["farEnhance"] = cv2.getTrackbarPos("farEnhance", window_name)
    params["nearCutoff"] = cv2.getTrackbarPos("nearCutoff %", window_name)
    params["farCutoff"] = cv2.getTrackbarPos("farCutoff %", window_name)
    
    # Morphological filtering
    params["useMorph"] = bool(cv2.getTrackbarPos("useMorph", window_name))
    params["morphIter"] = cv2.getTrackbarPos("morphIter", window_name)
    
    # Bilateral smoothing
    params["useBilateral"] = bool(cv2.getTrackbarPos("useBilateral", window_name))
    params["bilateralStrength"] = cv2.getTrackbarPos("bilateralStrength", window_name)
    
    # Weighted least squares refinement
    params["useWLS"] = bool(cv2.getTrackbarPos("useWLS", window_name))
    params["wlsLambda"] = cv2.getTrackbarPos("wlsLambda", window_name)
    params["wlsSigma"] = cv2.getTrackbarPos("wlsSigma x10", window_name) / 10.0
    
    # Object detection thresholds
    params["objectThresholdMM"] = cv2.getTrackbarPos("objectThresholdMM", window_name)
    params["wsSigma"] = cv2.getTrackbarPos("wsSigma", window_name)
    params["wsMinArea"] = cv2.getTrackbarPos("wsMinArea", window_name)
    
    # Edge enhancement
    params["edgeEqualize"] = bool(cv2.getTrackbarPos("edgeEqualize", window_name))
    params["edgeBilateralD"] = cv2.getTrackbarPos("edgeBilateralD", window_name)
    params["edgeBilateralSigma"] = cv2.getTrackbarPos("edgeBilateralSigma", window_name)
    params["edgeCannyKLow"] = cv2.getTrackbarPos("edgeCannyKLow x10", window_name) / 10.0
    params["edgeCannyKHigh"] = cv2.getTrackbarPos("edgeCannyKHigh x10", window_name) / 10.0
    params["edgeUseScharr"] = bool(cv2.getTrackbarPos("edgeUseScharr", window_name))
    
    # Color segmentation
    params["colorFocusMM"] = cv2.getTrackbarPos("colorFocusMM", window_name)
    params["colorSpanMM"] = cv2.getTrackbarPos("colorSpanMM", window_name)
    params["segMode"] = cv2.getTrackbarPos("segMode", window_name)
    params["kmK"] = cv2.getTrackbarPos("kmK", window_name)
    params["kmSpatialX100"] = cv2.getTrackbarPos("kmSpatialX100", window_name)
    params["rgTau"] = cv2.getTrackbarPos("rgTau", window_name)
    params["rgSeedStep"] = cv2.getTrackbarPos("rgSeedStep", window_name)
    
    return params

def run_disparity_calibration(vision, camera_config):
    """Interactive disparity parameter tuning with live preview."""
    
    if not vision.connected:
        print(f"{vision.name}: Starting cameras for calibration...")
        vision.start()
    
    print("\n" + "=" * 60)
    print("DISPARITY CALIBRATION - INTERACTIVE TUNING")
    print("=" * 60)
    print("Instructions:")
    print("  - Adjust trackbars to tune disparity parameters")
    print("  - Press 'w' to toggle WLS on/off quickly")
    print("  - Press 's' to save current parameters")
    print("  - Press 'r' to reset to original parameters")
    print("  - Press 'q' to quit without saving")
    print("  - Press 'Space' to pause/resume live updates")
    print("\n💡 TIP: Turn OFF WLS while tuning core stereo parameters!")
    print("   Then turn it ON to see the final refined result.")
    print("=" * 60 + "\n")
    
    window_name = "Disparity Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 400, 800)
    
    # Store original parameters
    original_params = {
        "minDisparity": vision.minDisparity,
        "numDisparitiesK": vision.numDisparitiesK,
        "numDisparities": vision.numDisparities,
        "blockSize": vision.blockSize,
        "preFilterCap": vision.preFilterCap,
        "uniquenessRatio": vision.uniquenessRatio,
        "speckleWindowSize": vision.speckleWindowSize,
        "speckleRange": vision.speckleRange,
        "disp12MaxDiff": vision.disp12MaxDiff,
        "medianBlurK": vision.medianBlurK,
        "downSample": vision.downSample,
        "crop": vision.crop,
        "farEnhance": vision.farEnhance,
        "nearCutoff": vision.nearCutoff,
        "farCutoff": vision.farCutoff,
        "useMorph": vision.useMorph,
        "morphIter": vision.morphIter,
        "useBilateral": vision.useBilateral,
        "bilateralStrength": vision.bilateralStrength,
        "useWLS": vision.useWLS,
        "wlsLambda": vision.wlsLambda,
        "wlsSigma": vision.wlsSigma,
        "objectThresholdMM": vision.objectThresholdMM,
        "wsSigma": vision.wsSigma,
        "wsMinArea": vision.wsMinArea,
        "edgeEqualize": vision.edgeEqualize,
        "edgeBilateralD": vision.edgeBilateralD,
        "edgeBilateralSigma": vision.edgeBilateralSigma,
        "edgeCannyKLow": vision.edgeCannyKLow,
        "edgeCannyKHigh": vision.edgeCannyKHigh,
        "edgeUseScharr": vision.edgeUseScharr,
        "colorFocusMM": vision.colorFocusMM,
        "colorSpanMM": vision.colorSpanMM,
        "segMode": vision.segMode,
        "kmK": vision.kmK,
        "kmSpatialX100": vision.kmSpatialX100,
        "rgTau": vision.rgTau,
        "rgSeedStep": vision.rgSeedStep,
    }
    
    create_trackbars(window_name, original_params)
    
    paused = False
    should_save = False
    saved_params = None
    
    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                if not should_save:
                    print("\n⚠️ Quitting without saving")
                break
            elif key == ord('s'):
                # Capture the parameters NOW, while window still exists
                saved_params = get_trackbar_values(window_name)
                should_save = True
                print("\n✅ Parameters marked for saving! Press 'q' to exit and save.")
            elif key == ord('r'):
                print("\n🔄 Resetting to original parameters")
                # Reset trackbars to original values
                for param_name, param_value in original_params.items():
                    if param_name in ["wlsSigma", "edgeCannyKLow", "edgeCannyKHigh"]:
                        cv2.setTrackbarPos(f"{param_name} x10", window_name, int(param_value * 10))
                    elif param_name in ["nearCutoff", "farCutoff"]:
                        cv2.setTrackbarPos(f"{param_name} %", window_name, int(param_value))
                    else:
                        cv2.setTrackbarPos(param_name, window_name, int(param_value))
            elif key == ord(' '):
                paused = not paused
                print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
            elif key == ord('w'):
                # Quick toggle WLS on/off
                current_wls = cv2.getTrackbarPos("useWLS", window_name)
                new_wls = 0 if current_wls else 1
                cv2.setTrackbarPos("useWLS", window_name, new_wls)
                print(f"🔄 WLS: {'ON' if new_wls else 'OFF'}")
            
            if not paused:
                # Get current parameter values
                current_params = get_trackbar_values(window_name)
                
                # Update vision system parameters
                for param_name, param_value in current_params.items():
                    setattr(vision, param_name, param_value)
                
                # Recreate the stereo matcher with new parameters
                block_size = current_params['blockSize']
                block_size = block_size if block_size % 2 == 1 else block_size + 1
                num_disparities = max(16, 16 * current_params['numDisparitiesK'])
                
                # Get P1 and P2 with defaults
                P1 = current_params.get('P1', 8 * 1 * block_size * block_size)
                P2 = current_params.get('P2', 32 * 1 * block_size * block_size)
                
                # Map mode integer to OpenCV enum
                sgbm_mode = current_params.get('sgbmMode', 2)
                mode_map = {
                    0: cv2.STEREO_SGBM_MODE_SGBM,
                    1: cv2.STEREO_SGBM_MODE_HH,
                    2: cv2.STEREO_SGBM_MODE_SGBM_3WAY,
                }
                mode = mode_map.get(sgbm_mode, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
                
                vision.stereo = cv2.StereoSGBM_create(
                    minDisparity=current_params['minDisparity'],
                    numDisparities=num_disparities,
                    blockSize=max(3, block_size),
                    P1=P1,
                    P2=P2,
                    preFilterCap=current_params['preFilterCap'],
                    uniquenessRatio=current_params['uniquenessRatio'],
                    speckleWindowSize=current_params['speckleWindowSize'],
                    speckleRange=current_params['speckleRange'],
                    disp12MaxDiff=current_params['disp12MaxDiff'],
                    mode=mode,
                )
                
                # Refresh the depth processor with new stereo matcher
                vision._refresh_depth_processor()
                
                # Get live frames and depth using read() method
                try:
                    result = vision.read()
                    depth_map = result.get('depth_map')
                    metadata = result.get('metadata', {})
                    
                    # Check for errors
                    if 'error' in metadata:
                        print(f"⚠️ Error: {metadata['error']}")
                        continue
                    
                    # Create visualization
                    if depth_map is not None and depth_map.size > 0:
                        # Normalize for visualization
                        depth_viz = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                        depth_viz = depth_viz.astype(np.uint8)
                        depth_colormap = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
                        
                        # Resize for display
                        h, w = depth_colormap.shape[:2]
                        display_width = 800
                        display_height = int(h * display_width / w)
                        depth_display = cv2.resize(depth_colormap, (display_width, display_height))
                        
                        # Add parameter text overlay
                        cv2.putText(depth_display, "Press 'w' to toggle WLS | 's' save | 'r' reset | 'q' quit", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(depth_display, f"blockSize: {current_params['blockSize']}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(depth_display, f"numDisparities: {current_params['numDisparities']}", 
                                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(depth_display, f"uniquenessRatio: {current_params['uniquenessRatio']}", 
                                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # WLS status with color coding
                        wls_color = (0, 255, 0) if current_params['useWLS'] else (0, 0, 255)  # Green if ON, Red if OFF
                        wls_text = "WLS: ON (Final Refinement)" if current_params['useWLS'] else "WLS: OFF (Raw Disparity)"
                        cv2.putText(depth_display, wls_text, 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, wls_color, 2)
                        
                        cv2.imshow("Depth Preview", depth_display)
                    else:
                        print("⚠️ No depth data available")
                except Exception as e:
                    print(f"⚠️ Error getting depth: {e}")
                    import traceback
                    traceback.print_exc()
            
    except KeyboardInterrupt:
        print("\n⚠️ Calibration interrupted by user")
    finally:
        cv2.destroyAllWindows()
    
    # If user pressed 's', update camera_config with saved parameters and return it
    if should_save and saved_params is not None:
        for param_name, param_value in saved_params.items():
            camera_config[param_name] = param_value
        
        print(f"\n💾 Disparity calibration complete")
        return camera_config
    else:
        return None

if __name__ == "__main__":
    # Load config
    print("Loading Config...")
    try:
        with open(config_path, "rb") as f:
            config = dill.load(f)
        print("✅ Loaded whole Dill")
        camera_config = config['camera']
        print("✅ Loaded Camera Config")
    except Exception as e:
        raise KeyError(f"An unexpected error occurred loading config.dill: {e}")
    
    # Instantiate vision system
    print("\n" + "="*60)
    print("Initializing Vision System...")
    print("="*60)
    
    # Import and load the vision class
    module_path, class_name = camera_config['who_to_run'].rsplit(".", 1)
    spec = importlib.util.spec_from_file_location(
        module_path, 
        os.path.join(os.path.dirname(__file__), *module_path.split(".")) + ".py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = module
    spec.loader.exec_module(module)
    VisionClass = getattr(module, class_name)
    
    # Create vision instance
    vision = VisionClass(name="camera", **camera_config)
    
    try:
        # Run disparity calibration
        updated_camera_settings = run_disparity_calibration(vision, camera_config)
        
        if updated_camera_settings is not None:
            # Update the config with new calibration data
            config['camera'] = updated_camera_settings
            
            # Save updated config back to dill file
            print("\n" + "="*60)
            print("Saving updated configuration to config.dill...")
            with open(config_path, "wb") as f:
                dill.dump(config, f)
            print("✅ Configuration saved successfully!")
            print("="*60)
            
            # Print summary of key parameters
            print("\n📊 Key Parameters Summary:")
            print(f"  blockSize: {updated_camera_settings['blockSize']}")
            print(f"  numDisparities: {updated_camera_settings['numDisparities']}")
            print(f"  uniquenessRatio: {updated_camera_settings['uniquenessRatio']}")
            print(f"  useWLS: {updated_camera_settings['useWLS']}")
            print(f"  wlsLambda: {updated_camera_settings['wlsLambda']}")
        else:
            print("\n⚠️ No parameters saved - config.dill unchanged")
        
    except Exception as e:
        print(f"\n❌ Error during disparity calibration: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up
        if vision.connected:
            print("\nStopping vision system...")
            vision.stop()

